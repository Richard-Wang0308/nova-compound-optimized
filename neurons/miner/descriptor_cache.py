"""
Molecular descriptor caching system for performance optimization.
"""

import time
from typing import Dict, Optional
from threading import Lock
from rdkit import Chem
from rdkit.Chem import Descriptors
import bittensor as bt


class MolecularDescriptorCache:
    """
    Cache for molecular descriptors to avoid repeated RDKit calculations.
    
    Descriptors are expensive to compute, so we cache them by SMILES.
    This is especially useful when the same molecules appear multiple times
    across iterations.
    
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.cache = {}  # {smiles: descriptors_dict}
        self.hits = 0
        self.misses = 0
        self.lock = Lock()  # Thread safety
    
    def get(self, smiles: str) -> Optional[Dict[str, float]]:
        """Get cached descriptors for a SMILES string (thread-safe)."""
        with self.lock:
            if smiles in self.cache:
                self.hits += 1
                return self.cache[smiles].copy()  # Return copy to prevent mutation
            else:
                self.misses += 1
                return None
    
    def set(self, smiles: str, descriptors: Dict[str, float]):
        """Cache descriptors for a SMILES string (thread-safe)."""
        with self.lock:
            # Implement LRU-like behavior: remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Remove 10% of oldest entries (simple FIFO)
                remove_count = self.max_size // 10
                keys_to_remove = list(self.cache.keys())[:remove_count]
                for key in keys_to_remove:
                    del self.cache[key]
            
            self.cache[smiles] = descriptors.copy()  # Store copy to prevent mutation
    
    def compute_and_cache(self, smiles: str, mol: Chem.Mol = None) -> Optional[Dict[str, float]]:
        """
        Compute descriptors and cache them.
        
        Args:
            smiles: SMILES string
            mol: Pre-computed RDKit Mol object (optional, for efficiency)
        
        Returns:
            Dictionary of descriptors or None if invalid
        """
        # Check cache first
        cached = self.get(smiles)
        if cached:
            return cached
        
        try:
            # Convert SMILES to Mol if not provided
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    return None
            
            # Compute all descriptors at once
            descriptors = {
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'mol_weight': Descriptors.MolWt(mol),
                'tpsa': Descriptors.TPSA(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'num_rings': Descriptors.RingCount(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
            }
            
            # Cache the result
            self.set(smiles, descriptors)
            return descriptors
        
        except Exception as e:
            bt.logging.debug(f"Error computing descriptors for {smiles}: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get cache statistics (thread-safe)."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'usage': len(self.cache) / self.max_size
            }
    
    def clear(self):
        """Clear the cache (thread-safe)."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics (thread-safe)."""
        import sys
        
        with self.lock:
            cache_size_bytes = sys.getsizeof(self.cache)
            
            # Estimate size of all entries
            for key, value in self.cache.items():
                cache_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
            
            return {
                'cache_size_mb': cache_size_bytes / (1024 * 1024),
                'num_entries': len(self.cache),
                'avg_entry_size_bytes': cache_size_bytes / len(self.cache) if self.cache else 0
            }


# Global descriptor cache instance
_descriptor_cache = None
_cache_lock = Lock()


def get_descriptor_cache() -> MolecularDescriptorCache:
    """
    Get the global descriptor cache instance (thread-safe singleton).
    
    Uses double-check locking pattern for thread safety.
    """
    global _descriptor_cache
    if _descriptor_cache is None:
        with _cache_lock:
            if _descriptor_cache is None:  # Double-check locking
                _descriptor_cache = MolecularDescriptorCache(max_size=50000)
    return _descriptor_cache


def reset_descriptor_cache():
    """
    Reset the global descriptor cache (useful for testing).
    """
    global _descriptor_cache
    with _cache_lock:
        if _descriptor_cache is not None:
            _descriptor_cache.clear()
        _descriptor_cache = None
