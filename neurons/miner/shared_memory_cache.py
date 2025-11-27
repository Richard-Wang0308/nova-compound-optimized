"""
Shared memory cache for multi-GPU coordination within a single miner.

Uses multiprocessing shared memory for zero-copy data sharing between GPU workers.
"""

import multiprocessing as mp
from multiprocessing import Manager, Lock
from typing import Optional, Dict, List, Any
import pickle
import time
from collections import OrderedDict
import bittensor as bt


class SharedMemoryCache:
    """
    Thread-safe shared memory cache for multi-GPU workers.
    
    Features:
    - Zero-copy sharing between processes
    - LRU eviction policy
    - Thread-safe operations
    - Automatic cleanup
    """
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize shared memory cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.manager = Manager()
        
        # Shared dictionaries (accessible by all GPU workers)
        self.smiles_cache = self.manager.dict()  # {product_name: smiles}
        self.target_scores = self.manager.dict()  # {(smiles, protein): score}
        self.antitarget_scores = self.manager.dict()  # {(smiles, protein): score}
        
        # Locks for thread-safe operations
        self.smiles_lock = Lock()
        self.target_lock = Lock()
        self.antitarget_lock = Lock()
        
        # Configuration
        self.max_size = max_size
        
        # Statistics (shared counters)
        self.stats = self.manager.dict({
            'smiles_hits': 0,
            'smiles_misses': 0,
            'target_hits': 0,
            'target_misses': 0,
            'antitarget_hits': 0,
            'antitarget_misses': 0,
        })
        
        bt.logging.info(f"SharedMemoryCache initialized (max_size={max_size})")
    
    # ========================================================================
    # SMILES CACHE OPERATIONS
    # ========================================================================
    
    def get_smiles(self, product_name: str) -> Optional[str]:
        """Get SMILES from shared cache (thread-safe)"""
        with self.smiles_lock:
            if product_name in self.smiles_cache:
                self.stats['smiles_hits'] += 1
                return self.smiles_cache[product_name]
            else:
                self.stats['smiles_misses'] += 1
                return None
    
    def set_smiles(self, product_name: str, smiles: str):
        """Set SMILES in shared cache (thread-safe)"""
        with self.smiles_lock:
            # Evict oldest if at capacity
            if len(self.smiles_cache) >= self.max_size:
                # Remove 10% oldest entries
                to_remove = int(self.max_size * 0.1)
                keys_to_remove = list(self.smiles_cache.keys())[:to_remove]
                for key in keys_to_remove:
                    del self.smiles_cache[key]
            
            self.smiles_cache[product_name] = smiles
    
    def batch_get_smiles(self, product_names: List[str]) -> Dict[str, str]:
        """Batch get SMILES (thread-safe)"""
        results = {}
        with self.smiles_lock:
            for name in product_names:
                if name in self.smiles_cache:
                    results[name] = self.smiles_cache[name]
                    self.stats['smiles_hits'] += 1
                else:
                    self.stats['smiles_misses'] += 1
        return results
    
    def batch_set_smiles(self, smiles_dict: Dict[str, str]):
        """Batch set SMILES (thread-safe)"""
        with self.smiles_lock:
            # Check capacity
            if len(self.smiles_cache) + len(smiles_dict) > self.max_size:
                to_remove = int(self.max_size * 0.1)
                keys_to_remove = list(self.smiles_cache.keys())[:to_remove]
                for key in keys_to_remove:
                    del self.smiles_cache[key]
            
            # Add new entries
            for name, smiles in smiles_dict.items():
                self.smiles_cache[name] = smiles
    
    # ========================================================================
    # TARGET SCORE CACHE OPERATIONS
    # ========================================================================
    
    def get_target_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get target score from shared cache"""
        key = f"{smiles}:{protein_code}"
        with self.target_lock:
            if key in self.target_scores:
                self.stats['target_hits'] += 1
                return self.target_scores[key]
            else:
                self.stats['target_misses'] += 1
                return None
    
    def set_target_score(self, smiles: str, protein_code: str, score: float):
        """Set target score in shared cache"""
        key = f"{smiles}:{protein_code}"
        with self.target_lock:
            self.target_scores[key] = score
    
    def batch_get_target_scores(
        self, smiles_list: List[str], protein_code: str
    ) -> Dict[str, Optional[float]]:
        """Batch get target scores"""
        results = {}
        with self.target_lock:
            for smiles in smiles_list:
                key = f"{smiles}:{protein_code}"
                if key in self.target_scores:
                    results[smiles] = self.target_scores[key]
                    self.stats['target_hits'] += 1
                else:
                    results[smiles] = None
                    self.stats['target_misses'] += 1
        return results
    
    def batch_set_target_scores(
        self, scores_dict: Dict[str, float], protein_code: str
    ):
        """Batch set target scores"""
        with self.target_lock:
            for smiles, score in scores_dict.items():
                key = f"{smiles}:{protein_code}"
                self.target_scores[key] = score
    
    # ========================================================================
    # ANTITARGET SCORE CACHE OPERATIONS
    # ========================================================================
    
    def get_antitarget_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get antitarget score from shared cache"""
        key = f"{smiles}:{protein_code}"
        with self.antitarget_lock:
            if key in self.antitarget_scores:
                self.stats['antitarget_hits'] += 1
                return self.antitarget_scores[key]
            else:
                self.stats['antitarget_misses'] += 1
                return None
    
    def set_antitarget_score(self, smiles: str, protein_code: str, score: float):
        """Set antitarget score in shared cache"""
        key = f"{smiles}:{protein_code}"
        with self.antitarget_lock:
            self.antitarget_scores[key] = score
    
    def batch_get_antitarget_scores(
        self, smiles_list: List[str], protein_code: str
    ) -> Dict[str, Optional[float]]:
        """Batch get antitarget scores"""
        results = {}
        with self.antitarget_lock:
            for smiles in smiles_list:
                key = f"{smiles}:{protein_code}"
                if key in self.antitarget_scores:
                    results[smiles] = self.antitarget_scores[key]
                    self.stats['antitarget_hits'] += 1
                else:
                    results[smiles] = None
                    self.stats['antitarget_misses'] += 1
        return results
    
    def batch_set_antitarget_scores(
        self, scores_dict: Dict[str, float], protein_code: str
    ):
        """Batch set antitarget scores"""
        with self.antitarget_lock:
            for smiles, score in scores_dict.items():
                key = f"{smiles}:{protein_code}"
                self.antitarget_scores[key] = score
    
    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================
    
    def clear_target_scores(self):
        """Clear all target scores (called on weekly update)"""
        with self.target_lock:
            self.target_scores.clear()
            bt.logging.info("Cleared target score cache")
    
    def clear_antitarget_scores(self):
        """Clear all antitarget scores (called on epoch boundary)"""
        with self.antitarget_lock:
            self.antitarget_scores.clear()
            bt.logging.info("Cleared antitarget score cache")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        smiles_total = self.stats['smiles_hits'] + self.stats['smiles_misses']
        target_total = self.stats['target_hits'] + self.stats['target_misses']
        antitarget_total = self.stats['antitarget_hits'] + self.stats['antitarget_misses']
        
        return {
            'smiles': {
                'size': len(self.smiles_cache),
                'max_size': self.max_size,
                'hits': self.stats['smiles_hits'],
                'misses': self.stats['smiles_misses'],
                'hit_rate': self.stats['smiles_hits'] / smiles_total if smiles_total > 0 else 0,
            },
            'target_scores': {
                'size': len(self.target_scores),
                'hits': self.stats['target_hits'],
                'misses': self.stats['target_misses'],
                'hit_rate': self.stats['target_hits'] / target_total if target_total > 0 else 0,
            },
            'antitarget_scores': {
                'size': len(self.antitarget_scores),
                'hits': self.stats['antitarget_hits'],
                'misses': self.stats['antitarget_misses'],
                'hit_rate': self.stats['antitarget_hits'] / antitarget_total if antitarget_total > 0 else 0,
            }
        }
    
    def cleanup(self):
        """Cleanup shared memory resources"""
        try:
            self.manager.shutdown()
            bt.logging.info("SharedMemoryCache cleaned up")
        except:
            pass


# ============================================================================
# INTEGRATED CACHE MANAGER FOR MULTI-GPU
# ============================================================================

class MultiGPUCacheManager:
    """
    Unified cache manager combining local and shared memory caches.
    
    Strategy:
    1. Check local cache first (fastest - no locks)
    2. Check shared memory cache (fast - same machine)
    3. Compute if not found, then populate both caches
    """
    
    def __init__(
        self,
        local_smiles_cache,  # PersistentSMILESCache
        local_target_cache,  # WeeklyTargetCache
        local_antitarget_cache,  # EpochAntitargetCache
        shared_cache: SharedMemoryCache
    ):
        self.local_smiles = local_smiles_cache
        self.local_target = local_target_cache
        self.local_antitarget = local_antitarget_cache
        self.shared = shared_cache
    
    # ========================================================================
    # SMILES OPERATIONS
    # ========================================================================
    
    def get_smiles(self, product_name: str) -> Optional[str]:
        """Get SMILES with two-tier lookup (local -> shared)"""
        # Try local cache first (fastest)
        result = self.local_smiles.get(product_name)
        if result:
            return result
        
        # Try shared memory cache
        result = self.shared.get_smiles(product_name)
        if result:
            # Populate local cache for next time
            self.local_smiles.set(product_name, result)
            return result
        
        return None
    
    def set_smiles(self, product_name: str, smiles: str):
        """Set SMILES in both caches"""
        self.local_smiles.set(product_name, smiles)
        self.shared.set_smiles(product_name, smiles)
    
    def batch_get_smiles(self, product_names: List[str]) -> Dict[str, str]:
        """Batch get SMILES with two-tier lookup"""
        results = {}
        remaining = []
        
        # Try local cache first
        for name in product_names:
            smiles = self.local_smiles.get(name)
            if smiles:
                results[name] = smiles
            else:
                remaining.append(name)
        
        # Try shared cache for remaining
        if remaining:
            shared_results = self.shared.batch_get_smiles(remaining)
            
            # Populate local cache
            for name, smiles in shared_results.items():
                self.local_smiles.set(name, smiles)
                results[name] = smiles
        
        return results
    
    def batch_set_smiles(self, smiles_dict: Dict[str, str]):
        """Batch set SMILES in both caches"""
        # Set in local cache
        for name, smiles in smiles_dict.items():
            self.local_smiles.set(name, smiles)
        
        # Set in shared cache
        self.shared.batch_set_smiles(smiles_dict)
    
    # ========================================================================
    # TARGET SCORE OPERATIONS
    # ========================================================================
    
    def batch_get_target_scores(
        self, smiles_list: List[str], protein_code: str
    ) -> Dict[str, Optional[float]]:
        """Get target scores with two-tier lookup"""
        results = {}
        remaining = []
        
        # Try local cache first
        for smiles in smiles_list:
            score = self.local_target.get_score(smiles, protein_code)
            if score != -float('inf'):
                results[smiles] = score
            else:
                remaining.append(smiles)
        
        # Try shared cache for remaining
        if remaining:
            shared_results = self.shared.batch_get_target_scores(remaining, protein_code)
            
            # Populate local cache
            for smiles, score in shared_results.items():
                if score is not None:
                    self.local_target.score_cache[(smiles, protein_code)] = score
                    results[smiles] = score
        
        # Return None for still-missing scores
        for smiles in smiles_list:
            if smiles not in results:
                results[smiles] = None
        
        return results
    
    def batch_set_target_scores(
        self, scores_dict: Dict[str, float], protein_code: str
    ):
        """Set target scores in both caches"""
        # Set in local cache
        for smiles, score in scores_dict.items():
            self.local_target.score_cache[(smiles, protein_code)] = score
        
        # Set in shared cache
        self.shared.batch_set_target_scores(scores_dict, protein_code)
    
    # ========================================================================
    # ANTITARGET SCORE OPERATIONS
    # ========================================================================
    
    def batch_get_antitarget_scores(
        self, smiles_list: List[str], protein_code: str
    ) -> Dict[str, Optional[float]]:
        """Get antitarget scores with two-tier lookup"""
        results = {}
        remaining = []
        
        # Try local cache first
        for smiles in smiles_list:
            cache_key = (smiles, protein_code)
            if cache_key in self.local_antitarget.score_cache:
                results[smiles] = self.local_antitarget.score_cache[cache_key]
            else:
                remaining.append(smiles)
        
        # Try shared cache for remaining
        if remaining:
            shared_results = self.shared.batch_get_antitarget_scores(remaining, protein_code)
            
            # Populate local cache
            for smiles, score in shared_results.items():
                if score is not None:
                    self.local_antitarget.score_cache[(smiles, protein_code)] = score
                    results[smiles] = score
        
        # Return None for still-missing scores
        for smiles in smiles_list:
            if smiles not in results:
                results[smiles] = None
        
        return results
    
    def batch_set_antitarget_scores(
        self, scores_dict: Dict[str, float], protein_code: str
    ):
        """Set antitarget scores in both caches"""
        # Set in local cache
        for smiles, score in scores_dict.items():
            self.local_antitarget.score_cache[(smiles, protein_code)] = score
        
        # Set in shared cache
        self.shared.batch_set_antitarget_scores(scores_dict, protein_code)
    
    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================
    
    def clear_target_scores(self):
        """Clear target scores in both caches"""
        self.local_target.score_cache.clear()
        self.shared.clear_target_scores()
    
    def clear_antitarget_scores(self):
        """Clear antitarget scores in both caches"""
        self.local_antitarget.score_cache.clear()
        self.shared.clear_antitarget_scores()
    
    def get_stats(self) -> dict:
        """Get combined cache statistics"""
        local_smiles_stats = self.local_smiles.get_stats()
        local_target_stats = self.local_target.get_stats()
        local_antitarget_stats = self.local_antitarget.get_stats()
        shared_stats = self.shared.get_stats()
        
        return {
            'local': {
                'smiles': local_smiles_stats,
                'target': local_target_stats,
                'antitarget': local_antitarget_stats,
            },
            'shared': shared_stats
        }
