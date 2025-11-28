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
from typing import Any, Dict, List, Optional, Tuple

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

# IMPORTANT: Add PARENT_DIR first, then BASE_DIR
# This ensures that when boltz package is loaded, it's loaded from BASE_DIR (boltz/)
# and we can import from boltz.enhanced_wrapper, but also access boltz/boltz/ for internal imports
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)

# Clear boltz from module cache if it was already loaded from wrong location
# We need boltz to be loaded from boltz/ (where enhanced_wrapper.py is), not boltz/boltz/
if 'boltz' in sys.modules:
    boltz_module = sys.modules['boltz']
    if hasattr(boltz_module, '__path__') and boltz_module.__path__:
        boltz_path = boltz_module.__path__[0]
        if 'boltz/boltz' in boltz_path.replace('\\', '/'):
            # Boltz was loaded from boltz/boltz/, clear it
            del sys.modules['boltz']

import bittensor as bt


# Import from modules in BASE_DIR directly (avoiding boltz.* package import)
# to prevent conflicts with boltz package from boltz/boltz/
import precompute_conformers
import quantize_model
precompute_conformers_batch = precompute_conformers.precompute_conformers_batch
quantize_model = quantize_model.quantize_model

from boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy, is_boltz_safe_smiles
from boltz.model.models.boltz2 import Boltz2
from dataclasses import asdict
from boltz.main import (
    Boltz2DiffusionParams,
    PairformerArgsV2,
    MSAModuleArgs,
    BoltzSteeringParams,
)

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
        
        # NEW: Initialize subnet_config (will be set by validator)
        self.subnet_config = None
        
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
        # FIXED: Initialize BoltzOptimizationManager with proper batching
        # ═══════════════════════════════════════════════════════════
        
        # Determine optimal batch size based on GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            
            # Adaptive batch sizing based on GPU memory
            if gpu_memory_gb >= 40:  # A100, H100
                optimal_batch_size = 32
                num_workers = 8
            elif gpu_memory_gb >= 24:  # RTX 3090, 4090, A5000
                optimal_batch_size = 16
                num_workers = 6
            elif gpu_memory_gb >= 16:  # RTX 4080, A4000
                optimal_batch_size = 12
                num_workers = 4
            elif gpu_memory_gb >= 12:  # RTX 3080, 4070
                optimal_batch_size = 8
                num_workers = 4
            else:  # Smaller GPUs
                optimal_batch_size = 4
                num_workers = 2
            
            bt.logging.info(f"GPU {device_id}: {gpu_memory_gb:.1f}GB detected, using batch_size={optimal_batch_size}")
        else:
            optimal_batch_size = 4
            num_workers = 2
            bt.logging.warning("No GPU detected, using CPU with small batch size")
        
        # Override with config if specified
        config_batch_size = self.config.get('batch_size', None)
        if config_batch_size is not None:
            optimal_batch_size = config_batch_size
            bt.logging.info(f"Using config-specified batch_size={optimal_batch_size}")
        
        self.optimization_manager = BoltzOptimizationManager(
            device_id=device_id,
            quantization=self.config.get('quantization', 'none'),
            base_workers=self.config.get('num_workers', num_workers),
            max_workers=num_workers,
            target_vram_utilization=self.config.get('max_gpu_memory_usage', 0.85),  # Reduced from 0.90
            max_ram_usage=self.config.get('max_ram_usage', 0.85),
            enable_monitoring=self.config.get('enable_memory_monitoring', True),
            # NEW: Add batch size configuration
            target_batch_size=optimal_batch_size,
            enable_dynamic_batching=True,
            enable_mixed_precision=True,
            enable_memory_optimization=True
        )
        bt.logging.info(f"✓ BoltzOptimizationManager initialized (batch_size={optimal_batch_size}, workers={num_workers})")
        
        # Store batch size for later use
        self.batch_size = optimal_batch_size
        
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
    
    def _download_models_if_needed(self, cache: Path) -> None:
        """Download model checkpoints if they don't exist."""
        import urllib.request
        import tarfile
        
        BOLTZ2_URL_WITH_FALLBACK = [
            "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
        ]
        
        BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
            "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
        ]
        
        # Download CCD data if needed
        mols = cache / "mols"
        tar_mols = cache / "mols.tar"
        MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
        
        if not tar_mols.exists():
            bt.logging.info(f"Downloading CCD data to {tar_mols}...")
            urllib.request.urlretrieve(MOL_URL, str(tar_mols))
        
        if not mols.exists():
            bt.logging.info(f"Extracting CCD data to {mols}...")
            with tarfile.open(str(tar_mols), "r") as tar:
                tar.extractall(cache)
        
        # Download structure model if needed
        model = cache / "boltz2_conf.ckpt"
        if not model.exists():
            bt.logging.info(f"Downloading Boltz-2 structure model to {model}...")
            for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
                try:
                    urllib.request.urlretrieve(url, str(model))
                    bt.logging.info(f"Successfully downloaded structure model from {url}")
                    break
                except Exception as e:
                    if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                        raise RuntimeError(f"Failed to download model from all URLs. Last error: {e}") from e
                    bt.logging.warning(f"Failed to download from {url}, trying next URL...")
                    continue
        
        # Download affinity model if needed
        affinity_model = cache / "boltz2_aff.ckpt"
        if not affinity_model.exists():
            bt.logging.info(f"Downloading Boltz-2 affinity model to {affinity_model}...")
            for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
                try:
                    urllib.request.urlretrieve(url, str(affinity_model))
                    bt.logging.info(f"Successfully downloaded affinity model from {url}")
                    break
                except Exception as e:
                    if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                        raise RuntimeError(f"Failed to download affinity model from all URLs. Last error: {e}") from e
                    bt.logging.warning(f"Failed to download from {url}, trying next URL...")
                    continue
    
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
        
        # Download models if they don't exist
        self._download_models_if_needed(cache)
        
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
    # MAIN SCORING METHOD (enhanced with batching)
    # ========================================================================
    
    def score_molecules_target(
        self,
        valid_molecules_by_uid: dict,
        score_dict: dict,
        final_block_hash: str,
        subnet_config: dict
    ) -> None:
        """
        Score molecules using Boltz-2 with batching support.
        
        Args:
            valid_molecules_by_uid: {uid: {'smiles': [...], 'names': [...]}}
            score_dict: Output dictionary for scores
            final_block_hash: Block hash for determinism
            subnet_config: Subnet configuration
        """
        # Store subnet config
        self.subnet_config = subnet_config
        
        # Preprocess data (same as original)
        self.preprocess_data_for_boltz(
            valid_molecules_by_uid,
            score_dict,
            final_block_hash
        )
        
        if not self.unique_molecules:
            bt.logging.warning("No molecules to score")
            return
        
        # Load models
        structure_model, affinity_model = self._load_models_if_needed()
        
        # ═══════════════════════════════════════════════════════════
        # NEW: Batch the prediction calls
        # ═══════════════════════════════════════════════════════════
        all_input_files = []
        for smiles, id_list in self.unique_molecules.items():
            mol_idx = id_list[0][1]
            yaml_path = os.path.join(self.input_dir, f"{mol_idx}.yaml")
            all_input_files.append((mol_idx, yaml_path))
        
        # Split into batches
        batches = self.optimization_manager.prepare_batch(all_input_files)
        
        bt.logging.info(
            f"Running Boltz-2 on {len(all_input_files)} molecules in {len(batches)} batches "
            f"(batch_size={self.optimization_manager.current_batch_size})"
        )
        
        total_start_time = time.time()
        
        # Process each batch
        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()
            
            bt.logging.info(
                f"Processing batch {batch_idx}/{len(batches)} ({len(batch)} molecules)"
            )
            
            try:
                # Run Boltz-2 on this batch
                self._run_boltz_batch(
                    batch,
                    structure_model,
                    affinity_model
                )
                
                # Record performance
                batch_elapsed = time.time() - batch_start_time
                self.optimization_manager.record_batch_processing(
                    len(batch),
                    batch_elapsed
                )
                
                bt.logging.info(
                    f"✓ Batch {batch_idx}/{len(batches)} complete in {batch_elapsed:.2f}s "
                    f"({len(batch)/batch_elapsed:.2f} mol/s)"
                )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    bt.logging.error(f"OOM in batch {batch_idx}")
                    self.optimization_manager.record_oom_event()
                else:
                    bt.logging.error(f"Error in batch {batch_idx}: {e}")
                    bt.logging.error(traceback.format_exc())
            
            except Exception as e:
                bt.logging.error(f"Unexpected error in batch {batch_idx}: {e}")
                bt.logging.error(traceback.format_exc())
            
            finally:
                # Memory cleanup between batches
                if self.optimization_manager.enable_memory_optimization:
                    self.optimization_manager.clear_gpu_cache()
        
        total_elapsed = time.time() - total_start_time
        
        bt.logging.info(
            f"✓ Boltz-2 inference complete in {total_elapsed:.2f}s "
            f"({len(all_input_files)/total_elapsed:.2f} mol/s)"
        )
        
        # Postprocess results (same as original)
        self.postprocess_data(score_dict)
        
        # Log performance summary
        self.optimization_manager.log_performance_summary()
    
    def _run_boltz_batch(
        self,
        batch: List[Tuple[int, str]],
        structure_model: Any,
        affinity_model: Any
    ) -> None:
        """
        Run Boltz-2 prediction on a batch of input files with parallel processing.
        
        Since predict() only accepts single paths, we process multiple files
        in parallel using ThreadPoolExecutor.
        
        Args:
            batch: List of (mol_idx, yaml_path) tuples
            structure_model: Loaded structure model
            affinity_model: Loaded affinity model
        """
        import concurrent.futures
        import inspect
        
        bt.logging.debug(f"Running Boltz-2 on {len(batch)} molecules (parallel processing)")
        
        # Get predict function signature once
        predict_params = inspect.signature(predict).parameters
        
        def process_single_molecule(mol_idx: int, yaml_path: str) -> Tuple[int, bool, Optional[str]]:
            """
            Process a single molecule.
            
            Returns:
                (mol_idx, success, error_message)
            """
            try:
                # Build kwargs with only supported parameters
                kwargs = {
                    'data': yaml_path,  # ✅ Single path string, not list
                    'out_dir': self.output_dir,
                    'cache': Path("~/.boltz").expanduser(),
                    'devices': [self.device_id],
                    'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
                    'num_workers': 1,  # Use 1 worker per prediction to avoid conflicts
                    'override': False,
                }
                
                # Add optional parameters only if they exist
                if 'checkpoint' in predict_params:
                    kwargs['checkpoint'] = None
                
                if 'recycling_steps' in predict_params:
                    kwargs['recycling_steps'] = self.config['recycling_steps']
                
                if 'sampling_steps' in predict_params:
                    kwargs['sampling_steps'] = self.config['sampling_steps']
                
                if 'diffusion_samples' in predict_params:
                    kwargs['diffusion_samples'] = self.config['diffusion_samples']
                
                if 'output_format' in predict_params:
                    kwargs['output_format'] = "pdb"
                
                if 'use_msa_server' in predict_params:
                    kwargs['use_msa_server'] = False
                
                if 'msa_server_url' in predict_params:
                    kwargs['msa_server_url'] = None
                
                if 'msa_pairing_strategy' in predict_params:
                    kwargs['msa_pairing_strategy'] = "greedy"
                
                if 'write_full_pae' in predict_params:
                    kwargs['write_full_pae'] = False
                
                if 'write_full_pde' in predict_params:
                    kwargs['write_full_pde'] = False
                
                if 'model' in predict_params:
                    kwargs['model'] = structure_model
                
                if 'affinity_model' in predict_params:
                    kwargs['affinity_model'] = affinity_model
                
                # Call predict with only supported parameters
                predict(**kwargs)
                
                return mol_idx, True, None
                
            except Exception as e:
                error_msg = str(e)
                bt.logging.debug(f"Prediction failed for mol_idx={mol_idx}: {error_msg}")
                return mol_idx, False, error_msg
        
        # Determine optimal number of parallel workers
        # Use fewer workers to avoid GPU memory conflicts
        max_parallel = min(4, len(batch))  # Max 4 parallel predictions per GPU
        
        successful = 0
        failed = 0
        errors = []
        
        # Process molecules in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_molecule, mol_idx, yaml_path): (mol_idx, yaml_path)
                for mol_idx, yaml_path in batch
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                mol_idx, yaml_path = futures[future]
                
                try:
                    result_mol_idx, success, error_msg = future.result()
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        if error_msg:
                            errors.append((result_mol_idx, error_msg))
                            
                except Exception as e:
                    bt.logging.warning(f"Unexpected error processing mol_idx={mol_idx}: {e}")
                    failed += 1
                    errors.append((mol_idx, str(e)))
        
        # Log summary
        bt.logging.debug(
            f"Batch complete: {successful}/{len(batch)} successful, {failed} failed"
        )
        
        # Log unique error types (not every error to avoid spam)
        if errors:
            unique_errors = {}
            for mol_idx, error_msg in errors:
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                if error_type not in unique_errors:
                    unique_errors[error_type] = []
                unique_errors[error_type].append(mol_idx)
            
            for error_type, mol_indices in unique_errors.items():
                bt.logging.warning(
                    f"Error '{error_type}' occurred for {len(mol_indices)} molecules"
                )
        
        # Only raise if ALL predictions failed
        if successful == 0 and len(batch) > 0:
            raise RuntimeError(f"All {len(batch)} predictions failed in batch")

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
            
            if not os.path.exists(results_path):
                bt.logging.warning(f"Results path not found for mol_idx={mol_idx}: {results_path}")
                continue
            
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
        if self.config.get('remove_files', False):
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
                
                # Check if we have scores for this molecule
                if mol_idx not in scores:
                    bt.logging.warning(f"No scores found for mol_idx={mol_idx}, uid={uid}")
                    continue
                
                # Check if the metric exists
                if self.subnet_config['boltz_metric'] not in scores[mol_idx]:
                    bt.logging.warning(
                        f"Metric '{self.subnet_config['boltz_metric']}' not found for mol_idx={mol_idx}. "
                        f"Available metrics: {list(scores[mol_idx].keys())}"
                    )
                    continue
                
                metric_value = scores[mol_idx][self.subnet_config['boltz_metric']]
                final_boltz_scores[uid].append(metric_value)
                
                if uid not in self.per_molecule_metric:
                    self.per_molecule_metric[uid] = {}
                self.per_molecule_metric[uid][smiles] = metric_value
        
        bt.logging.debug(f"final_boltz_scores: {final_boltz_scores}")
        
        for uid, data in score_dict.items():
            if uid in final_boltz_scores and len(final_boltz_scores[uid]) > 0:
                data['boltz_score'] = np.mean(final_boltz_scores[uid])
            else:
                data['boltz_score'] = -math.inf
                bt.logging.warning(f"No valid Boltz scores for UID={uid}, setting to -inf")
    
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
