"""
Integrated cache management system for local-only caching.

This replaces the distributed cache system with a simpler local-only approach.
"""

import pickle
import os
from typing import Dict, Optional, Set
import bittensor as bt


class IntegratedCacheManager:
    """
    Integrated cache manager for local-only caching.
    
    Manages three types of caches:
    1. SMILES cache (permanent) - Never cleared
    2. Target scores cache (weekly) - Cleared when targets change
    3. Antitarget scores cache (epoch) - Cleared every epoch
    """
    
    def __init__(
        self,
        smiles_cache_path: str = "./cache/smiles_cache.pkl",
        enable_persistence: bool = True
    ):
        self.smiles_cache_path = smiles_cache_path
        self.enable_persistence = enable_persistence
        
        # Initialize caches
        from neurons.miner.cache_manager import (
            PersistentSMILESCache,
            WeeklyTargetCache,
            EpochAntitargetCache
        )
        
        # Permanent SMILES cache
        cache_dir = os.path.dirname(smiles_cache_path) or "./cache"
        self.smiles_cache = PersistentSMILESCache(
            cache_dir=cache_dir,
            max_memory_size=100000
        )
        
        # Weekly target cache (cleared when targets change)
        self.local_target = WeeklyTargetCache()
        
        # Epoch antitarget cache (cleared every epoch)
        self.local_antitarget = EpochAntitargetCache()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        bt.logging.info(f"Initialized IntegratedCacheManager (local-only)")
        bt.logging.info(f"  SMILES cache: {smiles_cache_path}")
        bt.logging.info(f"  Persistence: {enable_persistence}")
    
    def get_smiles(self, product_name: str) -> Optional[str]:
        """Get SMILES from cache."""
        smiles = self.smiles_cache.get(product_name)
        if smiles:
            self.hits += 1
        else:
            self.misses += 1
        return smiles
    
    def set_smiles(self, product_name: str, smiles: str):
        """Cache SMILES."""
        self.smiles_cache.set(product_name, smiles)
    
    def get_target_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get target score from cache."""
        score = self.local_target.get_score(smiles, protein_code)
        if score is not None:
            self.hits += 1
        else:
            self.misses += 1
        return score
    
    def set_target_score(self, smiles: str, protein_code: str, score: float):
        """Cache target score."""
        self.local_target.set_score(smiles, protein_code, score)
    
    def get_antitarget_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get antitarget score from cache."""
        score = self.local_antitarget.get_score(smiles, protein_code)
        if score is not None:
            self.hits += 1
        else:
            self.misses += 1
        return score
    
    def set_antitarget_score(self, smiles: str, protein_code: str, score: float):
        """Cache antitarget score."""
        self.local_antitarget.set_score(smiles, protein_code, score)
    
    def clear_week_caches(self):
        """Clear weekly caches (when targets change)."""
        bt.logging.info("Clearing weekly target caches...")
        self.local_target.clear_cache()
    
    def clear_epoch_caches(self):
        """Clear epoch caches (every epoch)."""
        bt.logging.info("Clearing epoch antitarget caches...")
        self.local_antitarget.clear_cache()
    
    def save_all(self):
        """Save all persistent caches to disk."""
        if self.enable_persistence:
            try:
                self.smiles_cache.save()
                bt.logging.debug("Saved SMILES cache to disk")
            except Exception as e:
                bt.logging.error(f"Failed to save SMILES cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        target_stats = self.local_target.get_stats()
        antitarget_stats = self.local_antitarget.get_stats()
        
        return {
            'cache_type': 'local',
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'smiles_cache_size': len(self.smiles_cache.cache),
            'target_cache_size': target_stats.get('cached_scores', 0),
            'antitarget_cache_size': antitarget_stats.get('cached_scores', 0),
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        
        bt.logging.info("="*60)
        bt.logging.info("INTEGRATED CACHE STATISTICS")
        bt.logging.info("="*60)
        bt.logging.info(f"Cache Type: {stats['cache_type']}")
        bt.logging.info(f"Total Hits: {stats['hits']:,}")
        bt.logging.info(f"Total Misses: {stats['misses']:,}")
        bt.logging.info(f"Hit Rate: {stats['hit_rate']:.1%}")
        bt.logging.info(f"SMILES Cache Size: {stats['smiles_cache_size']:,}")
        bt.logging.info(f"Target Cache Size: {stats['target_cache_size']:,}")
        bt.logging.info(f"Antitarget Cache Size: {stats['antitarget_cache_size']:,}")
        bt.logging.info("="*60)
    
    def close(self):
        """Close cache manager and save data."""
        bt.logging.info("Closing IntegratedCacheManager...")
        self.save_all()
        bt.logging.info("Cache manager closed")


def create_integrated_cache(
    smiles_cache_path: str = "./cache/smiles_cache.pkl",
    enable_persistence: bool = True
) -> IntegratedCacheManager:
    """
    Factory function to create an integrated cache manager.
    
    Args:
        smiles_cache_path: Path to SMILES cache file
        enable_persistence: Enable disk persistence
    
    Returns:
        IntegratedCacheManager instance
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(smiles_cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        bt.logging.info(f"Created cache directory: {cache_dir}")
    
    return IntegratedCacheManager(
        smiles_cache_path=smiles_cache_path,
        enable_persistence=enable_persistence
    )
