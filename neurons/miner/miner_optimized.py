"""
Optimized miner with three-tier caching and optimized Boltz-2 inference.

Performance improvements:
- Pre-loaded molecule pools (no DB queries)
- Multi-tier caching (permanent, weekly, epoch)
- Optimized Boltz-2 scoring pipeline with dynamic batching and mixed precision
- Resident model management (no reloading)
- Advanced GPU memory management
- RAM management with intelligent caching
- Vectorized elite selection and component weighting
- Comprehensive timing and profiling
- MULTIPROCESSING ENABLED for molecule generation

All critical bugs fixed:
- Fixed shutdown deadlock (graceful cancellation with shield)
- Fixed pool update race condition (single atomic update per iteration)
- Fixed submission timing (background monitor task)
- Fixed blocking operations (run in executors)
- Fixed resource leaks (proper executor cleanup)
- Optimized block queries (caching)
- Added comprehensive timing tracking
- Fixed multiprocessing module import issues
- Fixed WebSocket concurrency errors (thread-safe subtensor access)
- Fixed metagraph.sync TypeError
"""

import os
import sys

# Get project root (2 levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


import multiprocessing as mp

# ============================================================================
# CRITICAL: Multiprocessing Setup - MUST BE FIRST
# ============================================================================

if __name__ == "__main__":
    try:
        import platform
        # Use 'fork' on Linux (faster, no import issues)
        # Use 'spawn' on Windows/Mac
        if platform.system() == 'Linux':
            mp.set_start_method('fork', force=True)
        else:
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# ============================================================================
# FIX: Disable multiprocessing logging BEFORE any other imports
# ============================================================================
import logging
import warnings

# CRITICAL: Disable all multiprocessing logging to prevent EOFError
logging.logMultiprocessing = False
logging.logProcesses = False
logging.logThreads = False

# Completely disable the QueueHandler/QueueListener mechanism
import logging.handlers
_original_queue_handler_init = logging.handlers.QueueHandler.__init__

def _patched_queue_handler_init(self, queue):
    """Patched QueueHandler that doesn't use multiprocessing queues"""
    try:
        _original_queue_handler_init(self, queue)
    except (EOFError, OSError, ValueError):
        # Silently ignore queue errors
        pass

logging.handlers.QueueHandler.__init__ = _patched_queue_handler_init

# Suppress all multiprocessing warnings
warnings.filterwarnings('ignore', category=ResourceWarning, module='multiprocessing')
warnings.filterwarnings('ignore', message='.*EOFError.*')
warnings.filterwarnings('ignore', message='.*multiprocessing.*')
warnings.filterwarnings('ignore', message='.*logging.*')

# Set multiprocessing logger to CRITICAL only (effectively disabled)
mp_logger = logging.getLogger('multiprocessing')
mp_logger.setLevel(logging.CRITICAL)
mp_logger.handlers.clear()
mp_logger.propagate = False

# Also disable the root logger's handlers for multiprocessing
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.handlers.QueueHandler):
        logging.root.removeHandler(handler)


# ============================================================================
# Standard Imports
# ============================================================================
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import shutil
import base64
import hashlib
import time
import logging
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from asyncio import Lock

# Suppress multiprocessing warnings
logging.getLogger('multiprocessing').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=ResourceWarning, module='multiprocessing')

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
import pandas as pd
import torch
import gc

# ============================================================================
# Project Imports
# ============================================================================

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    is_reaction_allowed,
)

from boltz.enhanced_wrapper import EnhancedBoltzWrapper
from btdr import QuicknetBittensorDrandTimelock

# Import optimized cache and generation modules
from neurons.miner.cache_manager import (
    MoleculePoolCache,
    PersistentSMILESCache,
    WeeklyTargetCache,
    EpochAntitargetCache,
    get_reaction_roles,
)
from neurons.miner.integrated_cache import IntegratedCacheManager, create_integrated_cache
from neurons.miner.molecule_generator import (
    generate_valid_molecules_batch_optimized,
    ultra_light_prefilter,
)

# Import timing tracker
try:
    from neurons.miner.timing_tracker import TimingTracker
    TIMING_TRACKER_AVAILABLE = True
except ImportError:
    bt.logging.warning("timing_tracker not available, timing disabled")
    TIMING_TRACKER_AVAILABLE = False

# Import RAM management
try:
    from neurons.miner.ram_manager import (
        IntegratedRAMManager,
        EfficientDataFrameManager,
        optimize_dataframe_memory
    )
    RAM_MANAGER_AVAILABLE = True
except ImportError:
    bt.logging.warning("ram_manager not available, using basic RAM management")
    RAM_MANAGER_AVAILABLE = False

# Import elite optimizer
try:
    from neurons.miner.elite_optimizer import (
        IntegratedEliteOptimizer,
        VectorizedComponentWeightCalculator,
        AdaptiveEliteSelector
    )
    ELITE_OPTIMIZER_AVAILABLE = True
except ImportError:
    bt.logging.warning("elite_optimizer not available, using basic selection")
    ELITE_OPTIMIZER_AVAILABLE = False

# Import GPU memory management
try:
    from neurons.miner.gpu_memory_manager import (
        IntegratedGPUMemoryManager,
        create_gpu_memory_managers,
        gpu_memory_context,
    )
    GPU_MEMORY_MANAGER_AVAILABLE = True
    bt.logging.info("✓ GPU memory manager available")
except ImportError:
    bt.logging.warning("gpu_memory_manager not available, using basic memory management")
    GPU_MEMORY_MANAGER_AVAILABLE = False

# ============================================================================
# Suppress Multiprocessing Logging Errors
# ============================================================================

import logging
import warnings
from logging.handlers import QueueHandler, QueueListener

# Disable multiprocessing queue logging to prevent EOFError
logging.logMultiprocessing = False
logging.logProcesses = False
logging.logThreads = False

# Suppress all multiprocessing warnings
warnings.filterwarnings('ignore', category=ResourceWarning, module='multiprocessing')
warnings.filterwarnings('ignore', message='.*EOFError.*')
warnings.filterwarnings('ignore', message='.*multiprocessing.*')

# Set multiprocessing logger to ERROR only
mp_logger = logging.getLogger('multiprocessing')
mp_logger.setLevel(logging.ERROR)
mp_logger.handlers.clear()
mp_logger.propagate = False

# ============================================================================
# CONSTANTS
# ============================================================================

# Timing constants
SUBMISSION_DEADLINE_BLOCKS = 20
SUBMISSION_WINDOW_START = 5
MAINTENANCE_INTERVAL = 10
AGGRESSIVE_CLEANUP_INTERVAL = 50
CACHE_SAVE_INTERVAL = 300  # seconds
BLOCK_CACHE_UPDATE_INTERVAL = 6  # seconds (1 block)
METAGRAPH_SYNC_INTERVAL = 60  # blocks

# Shutdown constants
GRACEFUL_SHUTDOWN_TIMEOUT = 10.0  # seconds
FORCED_SHUTDOWN_TIMEOUT = 5.0  # seconds

# Generation constants
DEFAULT_MUTATION_PROB = 0.1
DEFAULT_N_SAMPLES_PER_REACTION = 150
DEFAULT_MAX_BATCH_SIZE = 400
DEFAULT_N_ELITES = 50
DEFAULT_ELITE_FRAC = 0.25

# Cache warming constants
CACHE_WARMUP_SAMPLES = 2000

# ============================================================================
# THREAD-SAFE BLOCK CACHING
# ============================================================================

async def get_cached_block(state: Dict[str, Any], cache_duration: int = 12) -> int:
    """
    Get block number with caching to avoid concurrent WebSocket calls.
    
    Args:
        state: Global state dictionary
        cache_duration: Cache validity in seconds
    
    Returns:
        Current block number
    """
    current_time = time.time()
    
    # Initialize cache if needed
    if 'block_cache' not in state:
        state['block_cache'] = {'block': None, 'timestamp': 0}
    
    if 'block_lock' not in state:
        state['block_lock'] = asyncio.Lock()
    
    cache = state['block_cache']
    
    # Return cached value if still valid
    if cache['block'] and (current_time - cache['timestamp']) < cache_duration:
        return cache['block']
    
    # Update cache with new block number
    try:
        async with state['block_lock']:
            # Double-check after acquiring lock
            if cache['block'] and (current_time - cache['timestamp']) < cache_duration:
                return cache['block']
            
            # Get new block number in executor to avoid blocking
            loop = asyncio.get_event_loop()
            block = await loop.run_in_executor(
                state.get('sync_executor'),
                state['subtensor'].get_current_block
            )
            
            cache['block'] = block
            cache['timestamp'] = current_time
            state['last_known_block'] = block
            return block
    
    except Exception as e:
        bt.logging.warning(f"Could not fetch current block: {e}")
        # Return cached value if available, even if expired
        if cache['block']:
            return cache['block']
        # Fallback to last known block
        return state.get('last_known_block', 0)


# ============================================================================
# CONFIG & ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments and merge with config defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    config.update(load_config())

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """Construct the path for GitHub operations from environment variables."""
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH', '')

    if not all([github_repo_name, github_repo_branch, github_repo_owner]):
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    """Set up Bittensor logging."""
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network}")
    bt.logging.info(config)


def validate_config(config: argparse.Namespace) -> None:
    """Validate configuration values."""
    try:
        # Validate numeric ranges
        if hasattr(config, 'prefilter_mw_max'):
            if not isinstance(config.prefilter_mw_max, (int, float)) or config.prefilter_mw_max <= 0:
                raise ValueError(f"prefilter_mw_max must be positive, got {config.prefilter_mw_max}")
        
        if hasattr(config, 'min_rotatable_bonds'):
            if not isinstance(config.min_rotatable_bonds, int) or config.min_rotatable_bonds < 0:
                raise ValueError(f"min_rotatable_bonds must be non-negative")
        
        if hasattr(config, 'max_rotatable_bonds'):
            if not isinstance(config.max_rotatable_bonds, int) or config.max_rotatable_bonds < 0:
                raise ValueError(f"max_rotatable_bonds must be non-negative")
        
        if hasattr(config, 'min_heavy_atoms'):
            if not isinstance(config.min_heavy_atoms, int) or config.min_heavy_atoms < 0:
                raise ValueError(f"min_heavy_atoms must be non-negative")
        
        bt.logging.info("Configuration validated successfully")
    except Exception as e:
        bt.logging.error(f"Configuration validation failed: {e}")
        raise


# ============================================================================
# BITTENSOR & NETWORK SETUP
# ============================================================================

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """Initialize wallet, subtensor, and metagraph."""
    bt.logging.info("Setting up Bittensor objects...")

    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    try:
        subtensor = bt.subtensor(config=config)
        bt.logging.info(f"Connected to subtensor network: {config.subtensor.network}")
        
        metagraph = subtensor.metagraph(config.netuid)
        metagraph.sync(subtensor=subtensor)
        bt.logging.info(f"Metagraph synced successfully")

        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Miner UID: {miner_uid}")

        node = SubstrateInterface(url=config.subtensor.chain_endpoint)
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"Epoch length: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except ValueError as e:
        bt.logging.error(f"Hotkey not registered in metagraph: {e}")
        traceback.print_exc()
        raise
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        traceback.print_exc()
        raise


def get_db_path() -> str:
    """Get the path to the combinatorial database."""
    db_path = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
    
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        bt.logging.error(f"Database file not found or empty at {db_path}")
        bt.logging.info("Attempting to download database from HuggingFace...")
        try:
            import requests
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite"
            response = requests.get(db_url, timeout=30)
            if response.status_code == 200:
                with open(db_path, 'wb') as f:
                    f.write(response.content)
                bt.logging.info(f"Database downloaded successfully to {db_path}")
            else:
                bt.logging.error(f"Failed to download database: HTTP {response.status_code}")
        except Exception as e:
            bt.logging.error(f"Error downloading database: {e}")
            traceback.print_exc()
    
    return db_path


# ============================================================================
# FALLBACK FUNCTIONS (if elite optimizer not available)
# ============================================================================

def build_component_weights_fallback(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """Fallback component weight calculation."""
    from collections import defaultdict
    
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    score_column = 'boltz_score' if 'boltz_score' in top_pool.columns else 'combined_score'
    for _, row in top_pool.iterrows():
        name = row['product_name']
        score = row[score_column]
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1
    
    return weights


def select_diverse_elites_fallback(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """Fallback elite selection."""
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    return top_pool.head(n_elites)


# ============================================================================
# CACHE PREWARMING
# ============================================================================

async def prewarm_caches_for_epoch(
    target_cache: WeeklyTargetCache,
    antitarget_cache: EpochAntitargetCache,
    pool_cache: MoleculePoolCache,
    cache_manager: IntegratedCacheManager,
    n_samples: int = CACHE_WARMUP_SAMPLES
) -> None:
    """
    Pre-warm caches at the start of each epoch.
    Note: Only warms SMILES cache. Target/antitarget scoring done by Boltz-2.
    """
    from neurons.miner.molecule_generator import (
        generate_molecules_from_pools_optimized,
        get_smiles_from_reaction_cached
    )
    
    try:
        bt.logging.info(f"Pre-warming SMILES cache with {n_samples} sample molecules...")
        
        sample_names = []
        for rxn_id in [4, 5]:
            try:
                names = generate_molecules_from_pools_optimized(
                    rxn_id, n_samples // 2, pool_cache, component_weights=None
                )
                sample_names.extend(names)
            except Exception as e:
                bt.logging.warning(f"Failed to generate molecules for rxn {rxn_id}: {e}")
                continue
        
        if not sample_names:
            bt.logging.warning("No molecules generated for cache warming, skipping")
            return
        
        # Generate SMILES to warm the cache
        sample_molecules = []
        for name in sample_names:
            try:
                smiles = get_smiles_from_reaction_cached(name, pool_cache, cache_manager)
                if smiles:
                    sample_molecules.append((name, smiles))
            except Exception as e:
                bt.logging.debug(f"Failed to get SMILES for {name}: {e}")
                continue
        
        if not sample_molecules:
            bt.logging.warning("No valid SMILES generated for cache warming")
            return
        
        bt.logging.info(f"Generated {len(sample_molecules)} sample molecules for SMILES cache warming")
        
        # Log cache stats
        smiles_stats = cache_manager.smiles_cache.get_stats()
        bt.logging.info(f"SMILES cache warming complete:")
        bt.logging.info(f"  SMILES cache: {smiles_stats['size']:,} entries (hit rate: {smiles_stats['hit_rate']:.1%})")
    
    except Exception as e:
        bt.logging.error(f"Cache prewarming failed: {e}")
        traceback.print_exc()


# ============================================================================
# SUBMISSION MONITOR (Background Task)
# ============================================================================

async def submission_monitor(state: Dict[str, Any]) -> None:
    """
    Background task that monitors for submission deadlines.
    
    This runs independently of the main pipeline and ensures submissions
    happen even if the pipeline is busy.
    """
    bt.logging.info("Submission monitor started")
    
    while not state['shutdown_event'].is_set():
        try:
            # Use cached block to avoid network calls
            current_block = await get_cached_block(state)
            if current_block == 0:
                # If no cached block available, skip this iteration
                await asyncio.sleep(2)
                continue
            
            next_epoch = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
            blocks_remaining = next_epoch - current_block
            
            # Submit if within deadline window
            if SUBMISSION_WINDOW_START <= blocks_remaining <= SUBMISSION_DEADLINE_BLOCKS:
                if state['candidate_product'] and state['candidate_product'] != state['last_submitted_product']:
                    bt.logging.info(f"📤 Submission window: {blocks_remaining} blocks remaining")
                    await submit_response(state)
            
            await asyncio.sleep(2)  # Check every 2 seconds
        
        except Exception as e:
            bt.logging.error(f"Submission monitor error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)
    
    bt.logging.info("Submission monitor stopped")


# ============================================================================
# POOL UPDATE (Thread-Safe, Atomic)
# ============================================================================

async def update_top_pool(
    state: Dict[str, Any],
    names: List[str],
    smiles: List[str],
    scores: List[float],
    ram_manager: Optional[Any]
) -> None:
    """
    Thread-safe, atomic pool update.
    
    This is called once per iteration (not per GPU) to avoid race conditions.
    """
    if not names:
        return
    
    pool_update_start = time.time()
    
    boltz_df = pd.DataFrame({
        'product_name': names,
        'smiles': smiles,
        'boltz_score': scores
    })
    
    # Single atomic update with lock
    async with state['pool_lock']:
        current_pool = state.get('boltz_top_pool', pd.DataFrame())
        
        if ram_manager and RAM_MANAGER_AVAILABLE:
            updated_pool = ram_manager.df_manager.efficient_concat([current_pool, boltz_df])
        else:
            updated_pool = pd.concat([current_pool, boltz_df], ignore_index=True)
        
        updated_pool = updated_pool.drop_duplicates(subset=['product_name'], keep='first')
        updated_pool = updated_pool.sort_values('boltz_score', ascending=False)
        updated_pool = updated_pool.head(50)
        
        if ram_manager and RAM_MANAGER_AVAILABLE:
            updated_pool = ram_manager.df_manager.optimize_dataframe(updated_pool, inplace=True)
        
        state['boltz_top_pool'] = updated_pool
        
        # Update best candidate
        if not updated_pool.empty:
            top = updated_pool.iloc[0]
            if top['boltz_score'] > state['best_score']:
                state['best_score'] = top['boltz_score']
                state['candidate_product'] = top['product_name']
                bt.logging.info(f"✨ New best: {top['boltz_score']:.4f} - {top['product_name']}")
    
    pool_update_elapsed = time.time() - pool_update_start
    
    # Record timing
    timing_tracker = state.get('timing_tracker')
    if timing_tracker and TIMING_TRACKER_AVAILABLE:
        timing_tracker._record_time('pool_update', pool_update_elapsed)


# ============================================================================
# OPTIMIZED PIPELINE
# ============================================================================

async def run_optimized_pipeline_with_caching(state: Dict[str, Any]) -> None:
    """Optimized pipeline with all enhancements."""
    pool_cache = state['pool_cache']
    cache_manager = state['cache_manager']
    config = state['config']
    ram_manager = state.get('ram_manager')
    
    iteration = 0
    seen_inchikeys = state['seen_inchikeys']
    
    # Adaptive parameters
    mutation_prob = DEFAULT_MUTATION_PROB
    n_samples_per_reaction = DEFAULT_N_SAMPLES_PER_REACTION
    
    bt.logging.info("Starting optimized pipeline with MULTIPROCESSING ENABLED")
    
    while not state['shutdown_event'].is_set():
        try:
            iteration += 1
            bt.logging.info(f"=== Iteration {iteration} ===")
            
            # Use RAM managed processing context
            if ram_manager and RAM_MANAGER_AVAILABLE:
                with ram_manager.managed_processing(cache_dict=cache_manager.smiles_cache.cache):
                    await process_iteration(state, iteration, pool_cache, cache_manager, config, 
                                          seen_inchikeys, mutation_prob, n_samples_per_reaction)
            else:
                await process_iteration(state, iteration, pool_cache, cache_manager, config,
                                      seen_inchikeys, mutation_prob, n_samples_per_reaction)
            
            # Periodic maintenance
            if iteration % MAINTENANCE_INTERVAL == 0:
                await perform_periodic_maintenance(state, iteration, ram_manager)
            
            # Aggressive cleanup
            if iteration % AGGRESSIVE_CLEANUP_INTERVAL == 0 and ram_manager and RAM_MANAGER_AVAILABLE:
                bt.logging.info("Performing aggressive RAM cleanup...")
                ram_manager.check_and_clean(
                    cache_dict=cache_manager.smiles_cache.cache,
                    force=True
                )
                ram_manager.print_full_report()
            
            # Save SMILES cache periodically
            if iteration % 20 == 0:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    state.get('file_io_executor'),
                    cache_manager.save_all
                )
                bt.logging.debug("Cache saved to disk")
            
            await asyncio.sleep(0.1)
        
        except Exception as e:
            bt.logging.error(f"Pipeline error: {e}")
            traceback.print_exc()
            await asyncio.sleep(2)
    
    bt.logging.info("Pipeline stopped")


async def process_iteration(
    state: Dict[str, Any],
    iteration: int,
    pool_cache: MoleculePoolCache,
    cache_manager: IntegratedCacheManager,
    config: argparse.Namespace,
    seen_inchikeys: set,
    mutation_prob: float,
    n_samples_per_reaction: int
) -> None:
    """Process a single iteration of the pipeline."""
    timing_tracker = state.get('timing_tracker')
    iteration_start = time.time()
    
    try:
        current_pool = state.get('boltz_top_pool', pd.DataFrame())
        ram_manager = state.get('ram_manager')
        elite_optimizer = state.get('elite_optimizer')
        
        # Check deadline before expensive operations
        current_block = await get_cached_block(state)
        next_epoch = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
        blocks_remaining = next_epoch - current_block
        
        # If too close to deadline, skip scoring (submission monitor will handle it)
        if blocks_remaining <= SUBMISSION_DEADLINE_BLOCKS + 5:
            bt.logging.info(f"⏰ Only {blocks_remaining} blocks remaining until epoch {next_epoch}, skipping iteration (waiting for new epoch)")
            return
        
        bt.logging.debug(f"📊 Blocks remaining: {blocks_remaining}, proceeding with iteration")
        
        # Cache empty check
        has_pool = not current_pool.empty
        
        # Process elite selection and component weights
        if has_pool and elite_optimizer and ELITE_OPTIMIZER_AVAILABLE:
            # Use optimized elite selection
            component_weights_dict, elite_df, elite_frac = elite_optimizer.process_top_pool(
                current_pool,
                rxn_ids=[4, 5],
                current_best_score=state['best_score'],
                n_elites=DEFAULT_N_ELITES
            )
            
            elite_names_dict = {
                4: [n for n in elite_df['product_name'] if n.startswith("rxn:4:")],
                5: [n for n in elite_df['product_name'] if n.startswith("rxn:5:")]
            }
            
            bt.logging.debug(
                f"Elite selection: fraction={elite_frac:.3f}, "
                f"selected={len(elite_df)}, "
                f"rxn4={len(elite_names_dict[4])}, "
                f"rxn5={len(elite_names_dict[5])}"
            )
        else:
            # Fallback to basic selection
            component_weights_dict = {}
            elite_names_dict = {}
            elite_frac = DEFAULT_ELITE_FRAC
            
            if has_pool:
                component_weights_dict[4] = build_component_weights_fallback(current_pool, 4)
                component_weights_dict[5] = build_component_weights_fallback(current_pool, 5)
                
                elite_df = select_diverse_elites_fallback(current_pool, n_elites=DEFAULT_N_ELITES)
                elite_names_dict[4] = [n for n in elite_df['product_name'] if n.startswith("rxn:4:")]
                elite_names_dict[5] = [n for n in elite_df['product_name'] if n.startswith("rxn:5:")]
        
        # Generate molecules with timing (MULTIPROCESSING ENABLED)
        generation_start = time.time()
        
        loop = asyncio.get_event_loop()
        executor = state.get('molecule_gen_executor', ThreadPoolExecutor(max_workers=4, thread_name_prefix="molgen"))
        state['molecule_gen_executor'] = executor
        
        # CRITICAL: Pass True as last parameter to enable multiprocessing
        sampler_data = await loop.run_in_executor(
            executor,
            generate_valid_molecules_batch_optimized,
            [4, 5],
            n_samples_per_reaction,
            pool_cache,
            cache_manager,
            config,
            DEFAULT_MAX_BATCH_SIZE,
            elite_names_dict,
            elite_frac,
            mutation_prob,
            seen_inchikeys,
            component_weights_dict,
            # True,  # ← ENABLE MULTIPROCESSING
            False,  # ← DISABLE MULTIPROCESSING
            None   # ← n_workers (None = auto-detect)
        )
        
        generation_elapsed = time.time() - generation_start
        
        if not sampler_data or not sampler_data.get('molecules'):
            bt.logging.error("No molecules generated, skipping iteration")
            return
        
        generated_names = sampler_data['molecules']
        generated_smiles = sampler_data['smiles']
        
        # Record generation timing
        if timing_tracker and TIMING_TRACKER_AVAILABLE:
            timing_tracker.record_generation(generation_elapsed, len(generated_names))
        
        bt.logging.info(f"Generated {len(generated_names)} valid molecules in {generation_elapsed:.2f}s ({len(generated_names)/generation_elapsed:.1f} mol/s)")
        
        # Parallel GPU processing with timing
        inference_start = time.time()
        
        async def process_gpu_sample(gpu_id: int) -> Tuple[List[str], List[str], List[float]]:
            """Each GPU samples and scores independently with memory management."""
            gpu_start = time.time()
            
            wrapper = state['boltz_wrappers'][gpu_id]
            memory_manager = state.get('memory_managers', {}).get(gpu_id)
            opt_manager = wrapper.optimization_manager
            
            loop = asyncio.get_event_loop()
            
            if memory_manager:
                memory_manager.check_memory(force=True)
            
            # Sample molecules
            if len(generated_names) > 100:
                random.seed(iteration * 1000 + gpu_id)
                indices = random.sample(range(len(generated_names)), 100)
                sample_names = [generated_names[i] for i in indices]
                sample_smiles = [generated_smiles[i] for i in indices]
            else:
                start_idx = gpu_id * 100
                end_idx = min(start_idx + 100, len(generated_names))
                sample_names = generated_names[start_idx:end_idx]
                sample_smiles = generated_smiles[start_idx:end_idx]
            
            # Pre-filter
            prefiltered = []
            for name, smiles in zip(sample_names, sample_smiles):
                ok, _ = ultra_light_prefilter(
                    smiles,
                    config.min_rotatable_bonds,
                    config.max_rotatable_bonds,
                    config.min_heavy_atoms,
                    getattr(config, 'prefilter_mw_max', 550.0),
                    getattr(config, 'prefilter_tpsa_max', 140.0)
                )
                if ok:
                    prefiltered.append((name, smiles))
            
            if not prefiltered:
                return [], [], []
            
            names = [p[0] for p in prefiltered]
            smiles = [p[1] for p in prefiltered]
            
            try:
                scoring_start = time.time()
                
                # Use managed context for GPU operations
                if memory_manager and GPU_MEMORY_MANAGER_AVAILABLE:
                    with memory_manager.managed_context():
                        scores = await score_molecules(wrapper, smiles, names, config, iteration, loop, state)
                else:
                    if GPU_MEMORY_MANAGER_AVAILABLE:
                        with gpu_memory_context(gpu_id):
                            scores = await score_molecules(wrapper, smiles, names, config, iteration, loop, state)
                    else:
                        scores = await score_molecules(wrapper, smiles, names, config, iteration, loop, state)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Record batch processing
                elapsed = time.time() - scoring_start
                opt_manager.record_batch_processing(len(smiles), elapsed)
                
                # Record GPU-specific timing
                gpu_elapsed = time.time() - gpu_start
                if timing_tracker and TIMING_TRACKER_AVAILABLE:
                    timing_tracker.record_gpu_inference(gpu_id, gpu_elapsed, len(smiles))
                
                if memory_manager:
                    alert = memory_manager.check_memory(force=True)
                    if alert == 'critical':
                        bt.logging.warning(f"GPU {gpu_id} critical memory, forcing cleanup")
                        memory_manager.cleanup(force=True)
                
                if ram_manager and RAM_MANAGER_AVAILABLE:
                    for name in names:
                        ram_manager.cache_cleaner.record_access(name)
                
                return names, smiles, scores
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    bt.logging.error(f"GPU {gpu_id} OOM error!")
                    opt_manager.record_oom_event()
                    
                    if memory_manager:
                        freed = memory_manager.cleanup(force=True)
                        bt.logging.info(f"Emergency cleanup freed {freed:.1f} MB")
                    else:
                        gc.collect()
                        torch.cuda.empty_cache()
                    return [], [], []
                else:
                    raise
            
            except Exception as e:
                bt.logging.error(f"GPU {gpu_id} scoring error: {e}")
                traceback.print_exc()
                return names, smiles, [-math.inf] * len(names)
        
        # Process all GPUs in parallel
        tasks = [process_gpu_sample(gpu_id) for gpu_id in state['boltz_wrappers'].keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        inference_elapsed = time.time() - inference_start
        
        # Merge results
        all_names, all_smiles, all_scores = [], [], []
        for result in results:
            if isinstance(result, Exception):
                bt.logging.error(f"GPU task error: {result}")
                continue
            names, smiles, scores = result
            all_names.extend(names)
            all_smiles.extend(smiles)
            all_scores.extend(scores)
        
        if not all_names:
            bt.logging.warning("No molecules scored")
            return
        
        # Record inference timing
        if timing_tracker and TIMING_TRACKER_AVAILABLE:
            timing_tracker._record_time('inference', inference_elapsed)
        
        bt.logging.info(f"Scored {len(all_names)} molecules in {inference_elapsed:.2f}s ({len(all_names)/inference_elapsed:.1f} mol/s)")
        
        # Single atomic pool update (not per-GPU)
        await update_top_pool(state, all_names, all_smiles, all_scores, ram_manager)
    
    finally:
        # Record total iteration time
        iteration_elapsed = time.time() - iteration_start
        if timing_tracker and TIMING_TRACKER_AVAILABLE:
            timing_tracker._record_time('iteration', iteration_elapsed)
            timing_tracker.print_iteration_summary(iteration)


async def score_molecules(
    wrapper: Any,
    smiles: List[str],
    names: List[str],
    config: argparse.Namespace,
    iteration: int,
    loop: asyncio.AbstractEventLoop,
    state: Dict[str, Any]
) -> List[float]:
    """Score molecules using EnhancedBoltzWrapper."""
    MINER_UID = 0
    valid_molecules_for_boltz = {
        MINER_UID: {'smiles': smiles, 'names': names}
    }
    score_dict = {
        MINER_UID: {'boltz_score': None, 'entropy_boltz': None}
    }
    
    final_block_hash = "0x" + hashlib.sha256(str(iteration).encode()).hexdigest()[:64]
    boltz_config = {
        'weekly_target': getattr(config, 'weekly_target', None),
        'num_antitargets': getattr(config, 'num_antitargets', 1),
        'binding_pocket': getattr(config, 'binding_pocket', None),
        'max_distance': getattr(config, 'max_distance', None),
        'force': getattr(config, 'force', False),
        'num_molecules_boltz': len(smiles),
        'boltz_metric': getattr(config, 'boltz_metric', 'affinity_probability_binary'),
        'sample_selection': 'first',
    }
    
    # Run scoring in executor
    await loop.run_in_executor(
        state['boltz_executor'],
        wrapper.score_molecules_target,
        valid_molecules_for_boltz,
        score_dict,
        boltz_config,
        final_block_hash
    )
    
    # Extract scores
    per_molecule_scores = wrapper.per_molecule_metric.get(MINER_UID, {})
    scores = []
    for s in smiles:
        if s in per_molecule_scores and per_molecule_scores[s] is not None:
            scores.append(float(per_molecule_scores[s]))
        else:
            avg = score_dict[MINER_UID].get('boltz_score', -math.inf)
            scores.append(float(avg) if avg != -math.inf and avg is not None else -math.inf)
    
    return scores


async def perform_periodic_maintenance(
    state: Dict[str, Any],
    iteration: int,
    ram_manager: Optional[Any]
) -> None:
    """Perform periodic maintenance tasks."""
    
    # Log stats from wrappers
    for gpu_id, wrapper in state['boltz_wrappers'].items():
        perf_stats = wrapper.get_performance_stats()
        mem_stats = wrapper.get_memory_stats()
        
        bt.logging.info(
            f"GPU {gpu_id} Stats: "
            f"throughput={perf_stats['avg_throughput']:.1f} mol/s, "
            f"VRAM={mem_stats['utilization']:.1%} "
            f"({mem_stats['allocated_gb']:.1f}/{mem_stats['reserved_gb']:.1f} GB), "
            f"OOM events={perf_stats['oom_count']}"
        )
    
    # Log GPU memory statistics
    if GPU_MEMORY_MANAGER_AVAILABLE:
        for gpu_id, mem_manager in state.get('memory_managers', {}).items():
            stats = mem_manager.get_comprehensive_stats()
            
            if 'monitoring' in stats:
                current = stats['monitoring']['current']
                bt.logging.info(
                    f"GPU {gpu_id} Memory: "
                    f"{current['utilization']*100:.1f}% used "
                    f"({current['allocated_mb']:.0f}/{stats['monitoring']['total_memory_mb']:.0f} MB), "
                    f"Warnings: {stats['monitoring']['alerts']['warning_count']}, "
                    f"Critical: {stats['monitoring']['alerts']['critical_count']}"
                )
            
            if hasattr(mem_manager, 'monitor') and mem_manager.monitor:
                if mem_manager.monitor.detect_memory_leak(window_size=50, threshold_slope=2.0):
                    bt.logging.warning(f"GPU {gpu_id}: Possible memory leak, forcing cleanup")
                    mem_manager.cleanup(force=True)
    
    # Log RAM statistics
    if ram_manager and RAM_MANAGER_AVAILABLE:
        stats = ram_manager.get_comprehensive_stats()
        memory = stats['cache_cleaner']['current_memory_usage']
        bt.logging.info(
            f"RAM: {memory['used_gb']:.1f}/{memory['total_gb']:.1f} GB "
            f"({memory['percent']*100:.1f}%), "
            f"Process: {memory['process_gb']:.1f} GB"
        )
        
        ram_manager.check_and_clean(
            cache_dict=state['cache_manager'].smiles_cache.cache,
            force=False
        )
    
    # Log elite optimizer statistics
    elite_optimizer = state.get('elite_optimizer')
    if elite_optimizer and ELITE_OPTIMIZER_AVAILABLE:
        stats = elite_optimizer.get_comprehensive_stats()
        
        es_stats = stats['elite_selector']
        bt.logging.info(
            f"Elite Optimizer: "
            f"elite_frac={es_stats['current_elite_frac']:.3f}, "
            f"diversity={es_stats['current_diversity']:.3f}, "
            f"trend={es_stats['score_trend']}"
        )
        
        wc_stats = stats['weight_calculator']
        bt.logging.info(
            f"Weight Calculation: "
            f"avg_time={wc_stats['avg_time_ms']:.2f}ms, "
            f"total={wc_stats['total_calculations']}"
        )
    
    # Print timing report
    timing_tracker = state.get('timing_tracker')
    if timing_tracker and TIMING_TRACKER_AVAILABLE:
        bt.logging.info("")
        bt.logging.info("="*70)
        timing_tracker.print_comprehensive_report()
    
    # Standard cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log cache statistics
    log_cache_statistics(state)


# ============================================================================
# SUBMISSION
# ============================================================================

async def submit_response(state: Dict[str, Any]) -> None:
    """Encrypt and submit the current candidate product."""
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    if not (is_reaction_allowed(candidate_product, "rxn:4") or is_reaction_allowed(candidate_product, "rxn:5")):
        bt.logging.warning(f"Candidate '{candidate_product}' does not use allowed reactions. Skipping.")
        return

    bt.logging.info(f"🚀 Starting submission for: {candidate_product}")
    
    current_block = await get_cached_block(state)
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated")

    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit: {commit_content}")

        try: 
            commitment_status = state['subtensor'].commit(
                wallet=state['wallet'],
                netuid=state['config'].netuid,
                data=commit_content
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
            return

        if commitment_status:
            try:
                bt.logging.info("Attempting GitHub upload...")
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"✅ File uploaded successfully to {commit_content}")
                    state['last_submitted_product'] = candidate_product
                    state['last_submission_time'] = datetime.datetime.now()
                else:
                    bt.logging.error(f"Failed to upload file to GitHub")
            except Exception as e:
                bt.logging.error(f"Failed to upload file: {e}")
                traceback.print_exc()


# ============================================================================
# MAIN EPOCH LOOP
# ============================================================================

def log_cache_statistics(state: Dict[str, Any]) -> None:
    """Log comprehensive cache statistics."""
    cache_stats = state['cache_manager'].get_stats()
    
    bt.logging.info("="*60)
    bt.logging.info("CACHE STATISTICS")
    bt.logging.info("="*60)
    bt.logging.info(f"Cache Type: {cache_stats.get('cache_type', 'N/A')}")
    bt.logging.info(f"Hits: {cache_stats.get('hits', 0):,}")
    bt.logging.info(f"Misses: {cache_stats.get('misses', 0):,}")
    bt.logging.info(f"Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    bt.logging.info(f"SMILES Cache Size: {cache_stats.get('smiles_cache_size', 0):,}")
    
    # Log target/antitarget cache stats
    target_stats = state['target_cache'].get_stats()
    antitarget_stats = state['antitarget_cache'].get_stats()
    bt.logging.info(f"Target Proteins: {target_stats['targets']}")
    bt.logging.info(f"Antitarget Proteins: {antitarget_stats['antitargets']}")
    
    bt.logging.info(f"GPU Workers: {len(state['boltz_wrappers'])}")
    bt.logging.info("="*60)


async def main_epoch_loop(state: Dict[str, Any]) -> None:
    """Main loop with intelligent cache management."""
    target_cache = state['target_cache']
    antitarget_cache = state['antitarget_cache']
    cache_manager = state['cache_manager']
    subtensor = state['subtensor']
    metagraph = state['metagraph']
    epoch_length = state['epoch_length']
    config = state['config']
    
    last_cache_save = time.time()
    last_metagraph_sync = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Use cached block function
            cached_block = await get_cached_block(state)
            
            if cached_block % epoch_length == 0:
                bt.logging.info(f"{'='*60}")
                bt.logging.info(f"🔄 EPOCH BOUNDARY at block {cached_block}")
                bt.logging.info(f"{'='*60}")
                
                # Get block hash in executor
                loop = asyncio.get_event_loop()
                block_hash = await loop.run_in_executor(
                    state.get('sync_executor'),
                    subtensor.get_block_hash,
                    cached_block
                )
                
                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets
                )
                
                if new_proteins:
                    if target_cache.needs_update(new_proteins["targets"]):
                        bt.logging.info("🔄 WEEKLY TARGET UPDATE DETECTED")
                        target_cache.update_targets(new_proteins["targets"])
                        
                        state['boltz_top_pool'] = pd.DataFrame()
                        state['seen_inchikeys'] = set()
                        state['best_score'] = float('-inf')
                        state['candidate_product'] = None
                        
                        cache_manager.clear_week_caches()
                    
                    if antitarget_cache.needs_update(new_proteins["antitargets"]):
                        bt.logging.info("🔄 EPOCH ANTITARGET UPDATE")
                        antitarget_cache.update_antitargets(new_proteins["antitargets"])
                        cache_manager.clear_epoch_caches()
                    
                    await prewarm_caches_for_epoch(
                        target_cache, antitarget_cache,
                        state['pool_cache'], cache_manager,
                        n_samples=CACHE_WARMUP_SAMPLES
                    )
                    
                    # Gracefully stop old inference task
                    if 'inference_task' in state and state['inference_task']:
                        if not state['inference_task'].done():
                            state['shutdown_event'].set()
                            try:
                                await asyncio.wait_for(state['inference_task'], timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
                            except asyncio.TimeoutError:
                                bt.logging.warning("Inference task did not complete within timeout, cancelling...")
                                state['inference_task'].cancel()
                                try:
                                    await asyncio.wait_for(
                                        asyncio.shield(state['inference_task']),
                                        timeout=FORCED_SHUTDOWN_TIMEOUT
                                    )
                                except (asyncio.TimeoutError, asyncio.CancelledError):
                                    bt.logging.warning("Inference task forcefully terminated")
                    
                    state['shutdown_event'] = asyncio.Event()
                    state['inference_task'] = asyncio.create_task(
                        run_optimized_pipeline_with_caching(state)
                    )
                
                log_cache_statistics(state)
            
            # Save cache in executor (non-blocking)
            if time.time() - last_cache_save > CACHE_SAVE_INTERVAL:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    state.get('file_io_executor'),
                    cache_manager.save_all
                )
                last_cache_save = time.time()
                bt.logging.debug("Cache saved to disk")
            
            # Sync metagraph in executor (non-blocking)
            if cached_block - last_metagraph_sync >= METAGRAPH_SYNC_INTERVAL:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    state.get('sync_executor'),
                    lambda: metagraph.sync(subtensor=subtensor)
                )
                last_metagraph_sync = cached_block
                bt.logging.info(
                    f"Block: {cached_block} | "
                    f"Epoch: {cached_block // epoch_length} | "
                    f"Blocks until next: {epoch_length - (cached_block % epoch_length)}"
                )
            
            await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            bt.logging.info("🛑 Shutting down miner...")
            state['shutdown_event'].set()
            
            # Graceful shutdown with shield
            if 'inference_task' in state and state['inference_task']:
                if not state['inference_task'].done():
                    try:
                        await asyncio.wait_for(state['inference_task'], timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
                        bt.logging.info("Inference task completed gracefully")
                    except asyncio.TimeoutError:
                        bt.logging.warning("Inference task did not complete, forcing cancellation...")
                        state['inference_task'].cancel()
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(state['inference_task']),
                                timeout=FORCED_SHUTDOWN_TIMEOUT
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            bt.logging.warning("Inference task forcefully terminated")
            
            # Stop submission monitor
            if 'submission_task' in state and state['submission_task']:
                state['submission_task'].cancel()
                try:
                    await state['submission_task']
                except asyncio.CancelledError:
                    bt.logging.info("Submission monitor stopped")
            
            # Shutdown all executors
            executors_to_shutdown = [
                ('molecule_gen_executor', "Molecule generator"),
                ('boltz_executor', "Boltz"),
                ('sync_executor', "Sync"),
                ('file_io_executor', "File I/O"),
            ]
            
            for executor_name, display_name in executors_to_shutdown:
                if executor_name in state and state[executor_name]:
                    state[executor_name].shutdown(wait=True)
                    bt.logging.info(f"{display_name} executor shut down")
            
            # Export timing data
            timing_tracker = state.get('timing_tracker')
            if timing_tracker and TIMING_TRACKER_AVAILABLE:
                try:
                    os.makedirs('./logs', exist_ok=True)
                    timing_tracker.export_to_csv('./logs/timing_data.csv')
                    bt.logging.info("✓ Timing data exported to logs/timing_data.csv")
                    timing_tracker.print_comprehensive_report()
                except Exception as e:
                    bt.logging.warning(f"Could not export timing data: {e}")
            
            # Print final reports
            for gpu_id, wrapper in state['boltz_wrappers'].items():
                bt.logging.info(f"=== GPU {gpu_id} Final Report ===")
                wrapper.log_performance_summary()
            
            if GPU_MEMORY_MANAGER_AVAILABLE:
                for gpu_id, mem_manager in state.get('memory_managers', {}).items():
                    mem_manager.print_full_report()
            
            if RAM_MANAGER_AVAILABLE and state.get('ram_manager'):
                state['ram_manager'].print_full_report()
            
            elite_optimizer = state.get('elite_optimizer')
            if elite_optimizer and ELITE_OPTIMIZER_AVAILABLE:
                elite_optimizer.print_report()
                
                try:
                    os.makedirs('./logs', exist_ok=True)
                    elite_optimizer.elite_selector.plot_history(
                        save_path='./logs/elite_adaptation_history.png'
                    )
                except Exception as e:
                    bt.logging.debug(f"Could not save elite history plot: {e}")
            
            # Final cleanup
            for gpu_id in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(gpu_id)
            
            cache_manager.save_all()
            cache_manager.close()
            bt.logging.info("✓ Miner shutdown complete")
            break
        
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_miner_optimized(config: argparse.Namespace) -> None:
    """Optimized miner with all performance enhancements."""
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)
    
    bt.logging.info("="*60)
    bt.logging.info("Initializing local cache system...")
    bt.logging.info("="*60)
    
    db_path = get_db_path()
    
    cache_manager = create_integrated_cache(
        smiles_cache_path="./cache/smiles_cache.pkl",
        enable_persistence=True
    )
    
    reaction_roles = get_reaction_roles(db_path)
    pool_cache = MoleculePoolCache(db_path, reaction_roles)
    
    bt.logging.info("✓ Local cache system initialized")
    
    # Initialize EnhancedBoltzWrapper
    bt.logging.info("Initializing Enhanced Boltz-2 wrappers for all GPUs...")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    boltz_wrappers = {}
    for gpu_id in range(gpu_count):
        boltz_wrappers[gpu_id] = EnhancedBoltzWrapper(device_id=gpu_id)
        bt.logging.info(f"✓ EnhancedBoltzWrapper initialized for GPU {gpu_id}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            bt.logging.info(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    bt.logging.info(f"✓ Initialized {len(boltz_wrappers)} GPU workers")
    
    # Initialize GPU memory managers
    memory_managers = {}
    if GPU_MEMORY_MANAGER_AVAILABLE:
        bt.logging.info("Initializing GPU memory managers...")
        memory_managers = create_gpu_memory_managers(
            num_gpus=gpu_count,
            enable_monitoring=True,
            enable_aggressive_cleanup=True
        )
        bt.logging.info(f"✓ Initialized {len(memory_managers)} GPU memory managers")
    
    # Initialize RAM manager
    ram_manager = None
    if RAM_MANAGER_AVAILABLE:
        bt.logging.info("Initializing RAM manager...")
        ram_manager = IntegratedRAMManager(
            memory_threshold=0.80,
            chunk_size=1000,
            warmup_size=1000,
            auto_cleanup=True
        )
        bt.logging.info("✓ RAM manager initialized")
    
    # Initialize elite optimizer
    elite_optimizer = None
    if ELITE_OPTIMIZER_AVAILABLE:
        bt.logging.info("Initializing elite optimizer...")
        elite_optimizer = IntegratedEliteOptimizer(
            min_elite_frac=0.05,
            max_elite_frac=0.40,
            target_diversity=0.5,
            smoothing=0.1
        )
        bt.logging.info("✓ Elite optimizer initialized")
    
    # Initialize timing tracker
    timing_tracker = None
    if TIMING_TRACKER_AVAILABLE:
        bt.logging.info("Initializing timing tracker...")
        timing_tracker = TimingTracker(history_size=100)
        bt.logging.info("✓ Timing tracker initialized")
    
    # Create state
    state = {
        'config': config,
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,
        
        # Caches
        'pool_cache': pool_cache,
        'reaction_roles': reaction_roles,
        'target_cache': cache_manager.local_target,
        'antitarget_cache': cache_manager.local_antitarget,
        'cache_manager': cache_manager,
        
        # Boltz-2
        'boltz_wrappers': boltz_wrappers,
        'boltz_executor': ThreadPoolExecutor(max_workers=len(boltz_wrappers), thread_name_prefix="boltz"),
        
        # Memory Management
        'memory_managers': memory_managers,
        'ram_manager': ram_manager,
        
        # Elite Optimizer
        'elite_optimizer': elite_optimizer,
        
        # Timing
        'timing_tracker': timing_tracker,
        
        # Additional executors
        'sync_executor': ThreadPoolExecutor(max_workers=1, thread_name_prefix="sync"),
        'file_io_executor': ThreadPoolExecutor(max_workers=1, thread_name_prefix="fileio"),
        
        # Thread safety
        'pool_lock': Lock(),
        
        # Evolutionary state
        'boltz_top_pool': pd.DataFrame(),
        'seen_inchikeys': set(),
        'best_score': float('-inf'),
        'candidate_product': None,
        'last_submitted_product': None,
        'shutdown_event': asyncio.Event(),
        
        # Block caching
        'cached_block': 0,
        'last_known_block': 0,
        
        # BDT
        'bdt': QuicknetBittensorDrandTimelock(),
        'github_path': load_github_path(),
    }
    
    # Get initial proteins
    current_block = await get_cached_block(state)
    state['cached_block'] = current_block
    last_boundary = (current_block // epoch_length) * epoch_length
    
    # Get block hash in executor
    loop = asyncio.get_event_loop()
    block_hash = await loop.run_in_executor(
        state['sync_executor'],
        subtensor.get_block_hash,
        last_boundary
    )
    
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )
    
    if startup_proteins:
        state['target_cache'].update_targets(startup_proteins["targets"])
        state['antitarget_cache'].update_antitargets(startup_proteins["antitargets"])
        
        await prewarm_caches_for_epoch(
            state['target_cache'], state['antitarget_cache'], 
            pool_cache, cache_manager, n_samples=CACHE_WARMUP_SAMPLES
        )
        
        # Start inference task
        state['inference_task'] = asyncio.create_task(
            run_optimized_pipeline_with_caching(state)
        )
        
        # Start submission monitor
        state['submission_task'] = asyncio.create_task(
            submission_monitor(state)
        )
    
    try:
        await main_epoch_loop(state)
    finally:
        cache_manager.close()
        bt.logging.info("Miner shutdown complete")


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    validate_config(config)
    await run_miner_optimized(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
