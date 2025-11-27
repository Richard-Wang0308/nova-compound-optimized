"""
Enhanced Boltz-2 wrapper using BoltzOptimizationManager and ResidentModelManager.

This replaces the original BoltzWrapper with a more modular, optimized version.
"""

import os
import yaml
import sys
import traceback
import json
import numpy as np
import random
import gc
import shutil
import hashlib
import math
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Environment setup (same as original)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)
sys.path.append(PARENT_DIR)

import bittensor as bt

from src.boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy, is_boltz_safe_smiles
from src.boltz.model.models.boltz2 import Boltz2
from dataclasses import asdict
from src.boltz.main import (
    Boltz2DiffusionParams,
    PairformerArgsV2,
    MSAModuleArgs,
    BoltzSteeringParams,
)
from boltz.precompute_conformers import precompute_conformers_batch
from boltz.quantize_model import quantize_model

# Import our new components
from neurons.miner.boltz_optimizer import BoltzOptimizationManager
from neurons.miner.boltz_model_manager import ResidentModelManager


# ============================================================================
# HELPER FUNCTIONS (from original wrapper)
# ============================================================================

def _snapshot_rng():
    return {
        "py":  random.getstate(),
        "np":  np.random.get_state(),
        "tc":  torch.random.get_rng_state(),
        "tcu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _restore_rng(snap):
    random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.random.set_rng_state(snap["tc"])
    if snap["tcu"] is not None:
        torch.cuda.set_rng_state_all(snap["tcu"])

def _seed_for_record(rec_id, base_seed):
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)


# ============================================================================
# ENHANCED BOLTZ WRAPPER
# ============================================================================

class EnhancedBoltzWrapper:
    """
    Enhanced Boltz-2 wrapper using:
    - BoltzOptimizationManager for monitoring and resource management
    - ResidentModelManager for thread-safe model caching
    
    Drop-in replacement for BoltzWrapper with better performance tracking.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize EnhancedBoltzWrapper.
        
        Args:
            device_id: GPU device ID to use
        """
        # Load config
        config_path = os.path.join(BASE_DIR, "boltz_config.yaml")
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.base_dir = BASE_DIR
        self.device_id = device_id
        
        # Setup directories
        self.tmp_dir = os.path.join(PARENT_DIR, "boltz_tmp_files")
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        os.makedirs(self.input_dir, exist_ok=True)
        
        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.precomputed_dir = os.path.join(self.tmp_dir, "precomputed_conformers")
        os.makedirs(self.precomputed_dir, exist_ok=True)
        
        self.use_precomputed_conformers = self.config.get('use_precomputed_conformers', True)
        
        bt.logging.debug(f"EnhancedBoltzWrapper initialized with device_id={device_id}")
        
        # ═══════════════════════════════════════════════════════════
        # NEW: Initialize BoltzOptimizationManager
        # ═══════════════════════════════════════════════════════════
        self.optimization_manager = BoltzOptimizationManager(
            device_id=device_id,
            quantization=self.config.get('quantization', 'none'),
            base_workers=self.config.get('num_workers', 4),
            max_workers=4,
            target_vram_utilization=self.config.get('max_gpu_memory_usage', 0.90),
            max_ram_usage=self.config.get('max_ram_usage', 0.85),
            enable_monitoring=self.config.get('enable_memory_monitoring', True)
        )
        bt.logging.info("✓ BoltzOptimizationManager initialized")
        
        # ═══════════════════════════════════════════════════════════
        # NEW: Initialize ResidentModelManager
        # ═══════════════════════════════════════════════════════════
        self.model_manager = ResidentModelManager()
        bt.logging.info("✓ ResidentModelManager initialized")
        
        # Storage for per-molecule metrics
        self.per_molecule_metric = {}
        
        # RNG setup for determinism
        self.base_seed = 68
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)
        
        self._rng0 = _snapshot_rng()
        bt.logging.debug("EnhancedBoltzWrapper initialized with deterministic baseline")
    
    # ========================================================================
    # MODEL LOADING (using ResidentModelManager)
    # ========================================================================
    
    def _load_models_if_needed(self):
        """
        Load and cache models using ResidentModelManager.
        Thread-safe and optimized.
        """
        # Check if models already loaded
        structure_model = self.model_manager.get_model("boltz2_structure", self.device_id)
        affinity_model = self.model_manager.get_model("boltz2_affinity", self.device_id)
        
        if structure_model is not None and affinity_model is not None:
            bt.logging.debug(f"Models already cached for GPU {self.device_id}")
            return structure_model, affinity_model
        
        bt.logging.info(f"Loading Boltz-2 models to GPU {self.device_id}...")
        
        cache = Path("~/.boltz").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        
        # ───────────────────────────────────────────────────────────
        # Load structure model using model manager
        # ───────────────────────────────────────────────────────────
        def load_structure_model():
            checkpoint = cache / "boltz2_conf.ckpt"
            diffusion_params = Boltz2DiffusionParams()
            diffusion_params.step_scale = 1.5
            pairformer_args = PairformerArgsV2()
            msa_args = MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
            steering_args = BoltzSteeringParams()
            steering_args.fk_steering = False
            steering_args.physical_guidance_update = False
            
            predict_args_structure = {
                "recycling_steps": self.config['recycling_steps'],
                "sampling_steps": self.config['sampling_steps'],
                "diffusion_samples": self.config['diffusion_samples'],
                "max_parallel_samples": 1,
                "write_confidence_summary": True,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            
            model = Boltz2.load_from_checkpoint(
                checkpoint,
                strict=True,
                predict_args=predict_args_structure,
                map_location=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
                diffusion_process_args=asdict(diffusion_params),
                ema=False,
                use_kernels=not self.config.get('no_kernels', False),
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
            )
            
            if torch.cuda.is_available():
                model = model.to(f"cuda:{self.device_id}")
            model.eval()
            
            return model
        
        structure_model = self.model_manager.load_model(
            model_name="boltz2_structure",
            model_loader_func=load_structure_model,
            device_id=self.device_id
        )
        
        # ───────────────────────────────────────────────────────────
        # Load affinity model using model manager
        # ───────────────────────────────────────────────────────────
        def load_affinity_model():
            affinity_checkpoint = cache / "boltz2_aff.ckpt"
            diffusion_params = Boltz2DiffusionParams()
            diffusion_params.step_scale = 1.5
            pairformer_args = PairformerArgsV2()
            msa_args = MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
            
            predict_args_affinity = {
                "recycling_steps": 5,
                "sampling_steps": self.config['sampling_steps_affinity'],
                "diffusion_samples": self.config['diffusion_samples_affinity'],
                "max_parallel_samples": 1,
                "write_confidence_summary": False,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            
            steering_args_affinity = BoltzSteeringParams()
            steering_args_affinity.fk_steering = False
            steering_args_affinity.guidance_update = False
            steering_args_affinity.physical_guidance_update = False
            steering_args_affinity.contact_guidance_update = False
            
            model = Boltz2.load_from_checkpoint(
                affinity_checkpoint,
                strict=True,
                predict_args=predict_args_affinity,
                map_location=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
                diffusion_process_args=asdict(diffusion_params),
                ema=False,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args_affinity),
                affinity_mw_correction=self.config.get('affinity_mw_correction', False),
            )
            
            if torch.cuda.is_available():
                model = model.to(f"cuda:{self.device_id}")
            model.eval()
            
            return model
        
        affinity_model = self.model_manager.load_model(
            model_name="boltz2_affinity",
            model_loader_func=load_affinity_model,
            device_id=self.device_id
        )
        
        # ───────────────────────────────────────────────────────────
        # Apply quantization if configured
        # ───────────────────────────────────────────────────────────
        quantization = self.config.get('quantization', 'none')
        if quantization == 'fp16':
            bt.logging.info("FP16 quantization enabled - PyTorch Lightning will use AMP")
        elif quantization == 'int8':
            bt.logging.info("Applying INT8 quantization to models...")
            structure_model = quantize_model(structure_model, quantization)
            affinity_model = quantize_model(affinity_model, quantization)
            
            # Update cached models
            self.model_manager.models[("boltz2_structure", self.device_id)] = structure_model
            self.model_manager.models[("boltz2_affinity", self.device_id)] = affinity_model
            
            bt.logging.info("Models quantized to INT8")
        
        bt.logging.info("✓ Boltz-2 models loaded and cached")
        
        return structure_model, affinity_model
    
    # ========================================================================
    # PREPROCESSING (same as original)
    # ========================================================================
    
    def preprocess_data_for_boltz(
        self, 
        valid_molecules_by_uid: dict, 
        score_dict: dict, 
        final_block_hash: str
    ) -> None:
        """Preprocess data for Boltz2 prediction."""
        # Get protein sequence
        self.protein_sequence = get_sequence_from_protein_code(
            self.subnet_config['weekly_target']
        )
        
        # Collect all unique molecules across all UIDs
        self.unique_molecules = {}
        
        bt.logging.info("Preprocessing data for Boltz2")
        for uid, valid_molecules in valid_molecules_by_uid.items():
            # Select subsample
            if self.subnet_config['sample_selection'] == "random":
                seed = int(final_block_hash[2:], 16) + uid
                rng = random.Random(seed)
                unique_indices = rng.sample(
                    range(len(valid_molecules['smiles'])),
                    k=self.subnet_config['num_molecules_boltz']
                )
                boltz_candidates_smiles = [
                    valid_molecules['smiles'][i] for i in unique_indices
                ]
            elif self.subnet_config['sample_selection'] == "first":
                boltz_candidates_smiles = valid_molecules['smiles'][
                    :self.subnet_config['num_molecules_boltz']
                ]
            else:
                bt.logging.error(
                    f"Invalid sample selection method: {self.subnet_config['sample_selection']}"
                )
                return None
            
            # Compute entropy
            if self.subnet_config['num_molecules_boltz'] > 1:
                try:
                    score_dict[uid]["entropy_boltz"] = compute_maccs_entropy(
                        boltz_candidates_smiles
                    )
                except Exception as e:
                    bt.logging.error(f"Error computing entropy for UID={uid}: {e}")
                    score_dict[uid]["entropy_boltz"] = None
            else:
                score_dict[uid]["entropy_boltz"] = None
            
            # Collect unique molecules
            for smiles in boltz_candidates_smiles:
                ok, reason = is_boltz_safe_smiles(smiles)
                if not ok:
                    bt.logging.warning(
                        f"Skipping {smiles} because it is not parseable: {reason}"
                    )
                    continue
                
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                
                rec_id = smiles + self.protein_sequence
                mol_idx = _seed_for_record(rec_id, self.base_seed)
                self.unique_molecules[smiles].append((uid, mol_idx))
        
        bt.logging.info(f"Unique Boltz candidates: {len(self.unique_molecules)}")
        
        # Prepare items for precomputation
        items_for_precompute = []
        for smiles, ids in self.unique_molecules.items():
            product_name = f"mol_{ids[0][1]}"
            items_for_precompute.append((product_name, smiles))
        
        # Precompute conformers
        if self.use_precomputed_conformers:
            index_path = os.path.join(self.precomputed_dir, "index.pt")
            need_precompute = True
            
            if os.path.exists(index_path):
                try:
                    index = torch.load(index_path)
                    all_precomputed = all(
                        f"mol_{ids[0][1]}" in index
                        for ids in self.unique_molecules.values()
                    )
                    if all_precomputed:
                        need_precompute = False
                        bt.logging.info("All conformers already precomputed")
                except Exception:
                    pass
            
            if need_precompute:
                bt.logging.info(
                    f"Precomputing conformers for {len(items_for_precompute)} molecules..."
                )
                stats = precompute_conformers_batch(
                    items_for_precompute,
                    Path(self.precomputed_dir),
                    max_workers=self.config.get('precompute_workers', 32),
                    shard_size=self.config.get('precompute_shard_size', 1000),
                )
                bt.logging.info(f"Precomputation stats: {stats}")
        
        # Write YAML files
        for smiles, ids in self.unique_molecules.items():
            yaml_content = self.create_yaml_content(smiles)
            with open(os.path.join(self.input_dir, f"{ids[0][1]}.yaml"), "w") as f:
                f.write(yaml_content)
        
        bt.logging.debug("Preprocessing complete")
    
    def create_yaml_content(self, ligand_smiles: str) -> str:
        """Create YAML content for Boltz2 prediction."""
        yaml_content = f"""version: 1
sequences:
    - protein:
        id: A
        sequence: {self.protein_sequence}
        msa: empty
    - ligand:
        id: B
        smiles: '{ligand_smiles}'
        """
        
        if self.subnet_config['binding_pocket'] is not None:
            yaml_content += f"""
constraints:
    - pocket:
        binder: B
        contacts: {self.subnet_config['binding_pocket']}
        max_distance: {self.subnet_config['max_distance']}
        force: {self.subnet_config['force']}
        """
        
        yaml_content += f"""
properties:
    - affinity:
        binder: B
        """
        
        return yaml_content
    
    # ========================================================================
    # MAIN SCORING METHOD (enhanced with monitoring)
    # ========================================================================
    
    def score_molecules_target(
        self,
        valid_molecules_by_uid: dict,
        score_dict: dict,
        subnet_config: dict,
        final_block_hash: str
    ) -> None:
        """
        Score molecules with enhanced monitoring and optimization.
        
        This is the main entry point called by the miner.
        """
        self.subnet_config = subnet_config
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: PREPROCESS
        # ═══════════════════════════════════════════════════════════
        preprocess_start = time.time()
        self.preprocess_data_for_boltz(valid_molecules_by_uid, score_dict, final_block_hash)
        preprocess_time = time.time() - preprocess_start
        bt.logging.debug(f"Preprocessing took {preprocess_time:.2f}s")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: LOAD MODELS (using ResidentModelManager)
        # ═══════════════════════════════════════════════════════════
        model_load_start = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
        
        _restore_rng(self._rng0)
        
        structure_model, affinity_model = self._load_models_if_needed()
        
        model_load_time = time.time() - model_load_start
        bt.logging.debug(f"Model loading took {model_load_time:.2f}s")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: ADJUST WORKERS (using BoltzOptimizationManager)
        # ═══════════════════════════════════════════════════════════
        num_workers = self.optimization_manager.get_optimal_workers()
        
        # Cap at 4 to avoid shared memory issues
        num_workers = min(num_workers, 4)
        
        if num_workers > 0:
            bt.logging.info(
                f"Using {num_workers} DataLoader workers "
                f"(capped at 4 to avoid shared memory issues)"
            )
        else:
            bt.logging.info("Using main process only (num_workers=0)")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: CHECK MEMORY BEFORE INFERENCE
        # ═══════════════════════════════════════════════════════════
        mem_stats = self.optimization_manager.get_memory_stats()
        if mem_stats['utilization'] > 0.90:
            bt.logging.warning(
                f"GPU memory usage high ({mem_stats['utilization']:.1%}), "
                "clearing cache..."
            )
            self.optimization_manager.clear_gpu_cache()
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: RUN BOLTZ-2 PREDICT
        # ═══════════════════════════════════════════════════════════
        bt.logging.info(f"Running Boltz2 on GPU {self.device_id}")
        
        inference_start = time.time()
        
        try:
            predict(
                data=self.input_dir,
                out_dir=self.output_dir,
                recycling_steps=self.config['recycling_steps'],
                sampling_steps=self.config['sampling_steps'],
                diffusion_samples=self.config['diffusion_samples'],
                sampling_steps_affinity=self.config['sampling_steps_affinity'],
                diffusion_samples_affinity=self.config['diffusion_samples_affinity'],
                output_format=self.config['output_format'],
                seed=68,
                affinity_mw_correction=self.config['affinity_mw_correction'],
                no_kernels=self.config['no_kernels'],
                batch_predictions=self.config['batch_predictions'],
                override=self.config['override'],
                devices=[self.device_id],
                num_workers=num_workers,
                structure_model=structure_model,
                affinity_model=affinity_model,
                precomputed_conformers_dir=self.precomputed_dir if self.use_precomputed_conformers else None,
                quantization=self.config.get('quantization', 'none'),
            )
            
            inference_time = time.time() - inference_start
            
            # ═══════════════════════════════════════════════════════════
            # STEP 6: RECORD PERFORMANCE METRICS
            # ═══════════════════════════════════════════════════════════
            num_molecules = len(self.unique_molecules)
            self.optimization_manager.record_batch_processing(
                num_molecules, inference_time
            )
            
            bt.logging.info(
                f"Boltz2 predictions complete: {num_molecules} molecules "
                f"in {inference_time:.2f}s "
                f"({num_molecules/inference_time:.2f} mol/s)"
            )
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                bt.logging.error(f"GPU {self.device_id} OOM error!")
                
                # Record OOM event
                self.optimization_manager.record_oom_event()
                
                # Emergency cleanup
                self.optimization_manager.clear_gpu_cache()
                gc.collect()
                
                raise
            else:
                raise
        
        except Exception as e:
            bt.logging.error(f"Error running Boltz2: {e}")
            bt.logging.error(traceback.format_exc())
            return None
        
        # ═══════════════════════════════════════════════════════════
        # STEP 7: POSTPROCESS
        # ═══════════════════════════════════════════════════════════
        postprocess_start = time.time()
        self.postprocess_data(score_dict)
        postprocess_time = time.time() - postprocess_start
        bt.logging.debug(f"Postprocessing took {postprocess_time:.2f}s")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 8: LOG COMPREHENSIVE STATS
        # ═══════════════════════════════════════════════════════════
        total_time = preprocess_time + model_load_time + inference_time + postprocess_time
        bt.logging.info(
            f"Total pipeline time: {total_time:.2f}s "
            f"(preprocess: {preprocess_time:.2f}s, "
            f"model_load: {model_load_time:.2f}s, "
            f"inference: {inference_time:.2f}s, "
            f"postprocess: {postprocess_time:.2f}s)"
        )
    
    # ========================================================================
    # POSTPROCESSING (same as original)
    # ========================================================================
    
    def postprocess_data(self, score_dict: dict) -> None:
        """Postprocess Boltz2 results."""
        # Collect scores
        scores = {}
        for smiles, id_list in self.unique_molecules.items():
            mol_idx = id_list[0][1]
            results_path = os.path.join(
                self.output_dir, 'boltz_results_inputs', 'predictions', f'{mol_idx}'
            )
            
            if mol_idx not in scores:
                scores[mol_idx] = {}
            
            for filepath in os.listdir(results_path):
                if filepath.startswith('affinity'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        affinity_data = json.load(f)
                    scores[mol_idx].update(affinity_data)
                elif filepath.startswith('confidence'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        confidence_data = json.load(f)
                    scores[mol_idx].update(confidence_data)
        
        # Cleanup files
        if self.config['remove_files']:
            bt.logging.info("Removing temporary files")
            results_dir = Path(self.output_dir) / 'boltz_results_inputs'
            if results_dir.exists():
                shutil.rmtree(results_dir)
            
            yaml_files = list(Path(self.input_dir).glob("*.yaml"))
            for yaml_file in yaml_files:
                yaml_file.unlink()
            bt.logging.info("Files removed")
        
        # Distribute results to all UIDs
        self.per_molecule_metric = {}
        final_boltz_scores = {}
        
        for smiles, id_list in self.unique_molecules.items():
            for uid, mol_idx in id_list:
                if uid not in final_boltz_scores:
                    final_boltz_scores[uid] = []
                
                metric_value = scores[mol_idx][self.subnet_config['boltz_metric']]
                final_boltz_scores[uid].append(metric_value)
                
                if uid not in self.per_molecule_metric:
                    self.per_molecule_metric[uid] = {}
                self.per_molecule_metric[uid][smiles] = metric_value
        
        bt.logging.debug(f"final_boltz_scores: {final_boltz_scores}")
        
        for uid, data in score_dict.items():
            if uid in final_boltz_scores:
                data['boltz_score'] = np.mean(final_boltz_scores[uid])
            else:
                data['boltz_score'] = -math.inf
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from optimization manager."""
        return self.optimization_manager.get_performance_stats()
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics from optimization manager."""
        return self.optimization_manager.get_memory_stats()
    
    def get_model_stats(self) -> Dict:
        """Get model statistics from model manager."""
        return self.model_manager.get_stats()
    
    def log_performance_summary(self):
        """Log comprehensive performance summary."""
        self.optimization_manager.log_performance_summary()
        
        # Also log model stats
        model_stats = self.get_model_stats()
        bt.logging.info(f"Models loaded: {model_stats['num_models_loaded']}")
        for model_name, stats in model_stats['models'].items():
            bt.logging.info(
                f"  {model_name}: {stats['size_mb']:.1f} MB, "
                f"load_time={stats['load_time']:.2f}s"
            )
    
    def clear_gpu_memory(self):
        """Clear GPU memory."""
        self.optimization_manager.clear_gpu_cache()
        gc.collect()
    
    def cleanup_model(self):
        """Clean up model and free GPU memory."""
        if hasattr(self, 'unique_molecules'):
            del self.unique_molecules
            self.unique_molecules = None
        if hasattr(self, 'protein_sequence'):
            del self.protein_sequence
            self.protein_sequence = None
        
        self.clear_gpu_memory()
