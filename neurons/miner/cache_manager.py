"""
Multi-tier caching system for molecular generation and scoring.

Tier 1: Permanent cache (reactions, molecules, SMILES)
Tier 2: Weekly cache (target proteins) - WITHOUT PSICHIC
Tier 3: Epoch cache (antitarget proteins) - WITHOUT PSICHIC
"""

import os
import pickle
import sqlite3
import math
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import OrderedDict
from threading import Lock
import gc

import bittensor as bt
from rdkit import Chem


# ============================================================================
# TIER 1: PERMANENT CACHES (Reactions & Molecules)
# ============================================================================

class MoleculePoolCache:
    """
    In-memory cache for molecule pools used in reactions 4 and 5.
    Eliminates repeated SQLite queries during generation.
    """
    
    def __init__(self, db_path: str, reaction_roles: dict):
        # Validate database path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.db_path = os.path.abspath(db_path)  # Use absolute path
        self.pools = {}  # {role_mask: [(mol_id, smiles, role_mask), ...]}
        self.role_to_reactions = {}  # {role_mask: [rxn_ids that use it]}
        self.reaction_roles = reaction_roles
        self.lock = Lock()  # Thread safety
        
        # Pre-load all required roles
        self._preload_pools(reaction_roles)
        
        bt.logging.info(f"MoleculePoolCache initialized with {len(self.pools)} role pools")
        for role_mask, molecules in self.pools.items():
            bt.logging.info(f"  Role {role_mask}: {len(molecules):,} molecules")
    
    def _preload_pools(self, reaction_roles: dict):
        """Load all molecule pools for reactions 4 and 5 into memory"""
        required_roles = set()
        
        # Collect all unique role masks
        for rxn_id, roles in reaction_roles.items():
            required_roles.add(roles['roleA'])
            required_roles.add(roles['roleB'])
            if roles['roleC']:
                required_roles.add(roles['roleC'])
            
            # Track which reactions use each role
            for role in [roles['roleA'], roles['roleB'], roles['roleC']]:
                if role:
                    if role not in self.role_to_reactions:
                        self.role_to_reactions[role] = []
                    self.role_to_reactions[role].append(rxn_id)
        
        bt.logging.info(f"Loading {len(required_roles)} unique role pools: {sorted(required_roles)}")
        
        # Load each role pool from database
        conn = None
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            
            for role_mask in required_roles:
                cursor.execute(
                    "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
                    (role_mask, role_mask)
                )
                self.pools[role_mask] = cursor.fetchall()
                bt.logging.debug(f"Loaded {len(self.pools[role_mask]):,} molecules for role {role_mask}")
        
        except Exception as e:
            bt.logging.error(f"Failed to preload molecule pools: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def get_pool(self, role_mask: int) -> list:
        """Get pre-loaded molecule pool for a role (thread-safe)"""
        with self.lock:
            return self.pools.get(role_mask, []).copy()
    
    def get_pool_size(self, role_mask: int) -> int:
        """Get size of a role pool"""
        with self.lock:
            return len(self.pools.get(role_mask, []))
    
    def find_molecule(self, mol_id: int, possible_roles: list) -> Optional[Tuple]:
        """
        Find a molecule in pre-loaded pools by ID.
        Returns: (mol_id, smiles, role_mask) or None
        """
        with self.lock:
            for role in possible_roles:
                if role is None:
                    continue
                pool = self.pools.get(role, [])
                for mol in pool:
                    if mol[0] == mol_id:  # mol[0] is mol_id
                        return mol
        return None
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'total_roles': len(self.pools),
                'total_molecules': sum(len(pool) for pool in self.pools.values()),
                'role_sizes': {role: len(pool) for role, pool in self.pools.items()}
            }


class PersistentSMILESCache:
    """
    Persistent cache for reaction products with disk backup.
    Survives miner restarts. Uses LRU eviction.
    """
    
    def __init__(self, cache_dir: str = "./cache", max_memory_size: int = 100000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "smiles_cache.pkl"
        
        # Use OrderedDict for efficient LRU
        self.cache = OrderedDict()
        self.max_memory_size = max_memory_size
        self.hits = 0
        self.misses = 0
        self.lock = Lock()
        
        # Auto-save tracking
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes
        self.unsaved_changes = 0
        
        # Load from disk if exists
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load cache from disk on startup with corruption handling"""
        if not self.cache_file.exists():
            bt.logging.info("No existing SMILES cache found, starting fresh")
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
            
            # Validate loaded cache
            if not isinstance(loaded_cache, dict):
                raise ValueError("Cache file contains invalid data type")
            
            # Convert to OrderedDict if needed
            if isinstance(loaded_cache, OrderedDict):
                self.cache = loaded_cache
            else:
                self.cache = OrderedDict(loaded_cache)
            
            bt.logging.info(f"Loaded {len(self.cache):,} cached SMILES from disk")
        
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            bt.logging.error(f"Cache file corrupted: {e}")
            
            # Backup corrupted file
            backup_file = self.cache_file.with_suffix('.pkl.corrupted')
            try:
                import shutil
                shutil.move(str(self.cache_file), str(backup_file))
                bt.logging.info(f"Backed up corrupted cache to {backup_file}")
            except Exception as backup_error:
                bt.logging.warning(f"Could not backup corrupted cache: {backup_error}")
                # Try to delete corrupted file
                try:
                    self.cache_file.unlink()
                except:
                    pass
            
            # Start with fresh cache
            self.cache = OrderedDict()
        
        except Exception as e:
            bt.logging.error(f"Unexpected error loading cache: {e}")
            import traceback
            traceback.print_exc()
            self.cache = OrderedDict()
    
    def _save_to_disk(self):
        """Save cache to disk with atomic write"""
        try:
            # Write to temporary file first (atomic operation)
            temp_file = self.cache_file.with_suffix('.pkl.tmp')
            
            with self.lock:
                cache_copy = self.cache.copy()
            
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.replace(self.cache_file)
            
            bt.logging.debug(f"Saved {len(cache_copy):,} SMILES to disk cache")
        
        except Exception as e:
            bt.logging.error(f"Failed to save SMILES cache: {e}")
            import traceback
            traceback.print_exc()
    
    def get(self, product_name: str) -> Optional[str]:
        """Get SMILES from cache (LRU - moves to end)"""
        with self.lock:
            if product_name in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(product_name)
                return self.cache[product_name]
            else:
                self.misses += 1
                return None
    
    def set(self, product_name: str, smiles: str):
        """Add SMILES to cache with LRU eviction"""
        with self.lock:
            if product_name in self.cache:
                # Update and move to end
                self.cache.move_to_end(product_name)
            
            self.cache[product_name] = smiles
            self.unsaved_changes += 1
            
            # Evict oldest entries if cache too large
            while len(self.cache) > self.max_memory_size:
                # Remove oldest (first) item
                evicted_key, _ = self.cache.popitem(last=False)
                bt.logging.debug(f"Evicted {evicted_key} from SMILES cache")
            
            # Auto-save if needed
            if (time.time() - self.last_save_time > self.save_interval and 
                self.unsaved_changes > 100):
                self._save_to_disk()
                self.last_save_time = time.time()
                self.unsaved_changes = 0
    
    def save(self):
        """Explicitly save to disk"""
        self._save_to_disk()
        with self.lock:
            self.unsaved_changes = 0
            self.last_save_time = time.time()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_memory_size,
                'usage': len(self.cache) / self.max_memory_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'unsaved_changes': self.unsaved_changes
            }
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
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
    
    def get_cache_health(self) -> dict:
        """Get cache health metrics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'is_healthy': hit_rate > 0.7,  # 70% hit rate threshold
                'recommendation': 'good' if hit_rate > 0.7 else 'needs_warming',
                'total_requests': total
            }


# ============================================================================
# TIER 2: WEEKLY TARGET CACHE (Simplified - No PSICHIC)
# ============================================================================

class WeeklyTargetCache:
    """
    Cache for weekly target proteins.
    Stores target information and provides score caching interface.
    """
    
    def __init__(self):
        self.current_targets = []
        self.week_hash = None
        
        # Score cache: {(smiles, protein_code): score}
        self.score_cache = {}
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        bt.logging.info("WeeklyTargetCache initialized (simplified - no PSICHIC)")
    
    def needs_update(self, new_targets: list) -> bool:
        """Check if targets have changed"""
        new_hash = hash(tuple(sorted(new_targets)))
        if new_hash != self.week_hash:
            bt.logging.info(f"Target proteins changed: {self.current_targets} -> {new_targets}")
            return True
        return False
    
    def update_targets(self, new_targets: list):
        """Update target proteins (called weekly)"""
        with self.lock:
            self.week_hash = hash(tuple(sorted(new_targets)))
            
            # Clear score cache (new week = new targets)
            self.score_cache.clear()
            self.current_targets = new_targets
            self.hits = 0
            self.misses = 0
        
        bt.logging.info(f"Weekly target cache updated: {len(new_targets)} targets")
    
    def get_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get cached score (returns None if not cached)"""
        cache_key = (smiles, protein_code)
        
        with self.lock:
            if cache_key in self.score_cache:
                self.hits += 1
                return self.score_cache[cache_key]
            else:
                self.misses += 1
                return None
    
    def set_score(self, smiles: str, protein_code: str, score: float):
        """Cache a score"""
        cache_key = (smiles, protein_code)
        with self.lock:
            self.score_cache[cache_key] = score
    
    def batch_score(self, smiles_list: list, protein_code: str) -> list:
        """
        Get cached scores for multiple molecules.
        Returns list with cached scores or None for uncached entries.
        """
        results = []
        
        with self.lock:
            for smiles in smiles_list:
                cache_key = (smiles, protein_code)
                if cache_key in self.score_cache:
                    self.hits += 1
                    results.append(self.score_cache[cache_key])
                else:
                    self.misses += 1
                    results.append(None)
        
        return results
    
    def batch_set_scores(self, smiles_list: list, protein_code: str, scores: list):
        """Cache multiple scores"""
        with self.lock:
            for smiles, score in zip(smiles_list, scores):
                cache_key = (smiles, protein_code)
                self.score_cache[cache_key] = score
    
    def clear_cache(self):
        """Clear all cached scores"""
        with self.lock:
            self.score_cache.clear()
            self.hits = 0
            self.misses = 0
        bt.logging.info("Weekly target cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'targets': self.current_targets,
                'cached_scores': len(self.score_cache),
                'week_hash': self.week_hash,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


# ============================================================================
# TIER 3: EPOCH ANTITARGET CACHE (Simplified - No PSICHIC)
# ============================================================================

class EpochAntitargetCache:
    """
    Cache for epoch-based antitarget proteins.
    Reloads every 72 minutes.
    """
    
    def __init__(self):
        self.current_antitargets = []
        self.epoch_hash = None
        
        # Score cache (cleared each epoch)
        self.score_cache = {}
        self.lock = Lock()
        
        # Track epoch timing
        self.epoch_start_time = None
        self.epoch_duration = 72 * 60  # 72 minutes in seconds
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        bt.logging.info("EpochAntitargetCache initialized (simplified - no PSICHIC)")
    
    def needs_update(self, new_antitargets: list) -> bool:
        """Check if antitargets have changed"""
        new_hash = hash(tuple(sorted(new_antitargets)))
        if new_hash != self.epoch_hash:
            bt.logging.info(f"Antitarget proteins changed: {self.current_antitargets} -> {new_antitargets}")
            return True
        return False
    
    def update_antitargets(self, new_antitargets: list):
        """Update antitarget proteins (called every epoch)"""
        with self.lock:
            self.epoch_hash = hash(tuple(sorted(new_antitargets)))
            self.epoch_start_time = time.time()
            
            # Clear score cache
            self.score_cache.clear()
            self.current_antitargets = new_antitargets
            self.hits = 0
            self.misses = 0
        
        bt.logging.info(f"Epoch antitarget cache updated: {len(new_antitargets)} antitargets")
    
    def get_score(self, smiles: str, protein_code: str) -> Optional[float]:
        """Get cached score (returns None if not cached)"""
        cache_key = (smiles, protein_code)
        
        with self.lock:
            if cache_key in self.score_cache:
                self.hits += 1
                return self.score_cache[cache_key]
            else:
                self.misses += 1
                return None
    
    def set_score(self, smiles: str, protein_code: str, score: float):
        """Cache a score"""
        cache_key = (smiles, protein_code)
        with self.lock:
            self.score_cache[cache_key] = score
    
    def batch_score(self, smiles_list: list, protein_code: str) -> list:
        """
        Get cached scores for multiple molecules.
        Returns list with cached scores or None for uncached entries.
        """
        results = []
        
        with self.lock:
            for smiles in smiles_list:
                cache_key = (smiles, protein_code)
                if cache_key in self.score_cache:
                    self.hits += 1
                    results.append(self.score_cache[cache_key])
                else:
                    self.misses += 1
                    results.append(None)
        
        return results
    
    def batch_set_scores(self, smiles_list: list, protein_code: str, scores: list):
        """Cache multiple scores"""
        with self.lock:
            for smiles, score in zip(smiles_list, scores):
                cache_key = (smiles, protein_code)
                self.score_cache[cache_key] = score
    
    def clear_cache(self):
        """Clear all cached scores"""
        with self.lock:
            self.score_cache.clear()
            self.hits = 0
            self.misses = 0
        bt.logging.info("Epoch antitarget cache cleared")
    
    def get_time_remaining(self) -> float:
        """Get time remaining in current epoch (seconds)"""
        if self.epoch_start_time is None:
            return 0
        elapsed = time.time() - self.epoch_start_time
        return max(0, self.epoch_duration - elapsed)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'antitargets': self.current_antitargets,
                'cached_scores': len(self.score_cache),
                'epoch_hash': self.epoch_hash,
                'time_remaining_minutes': self.get_time_remaining() / 60,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_reaction_roles(db_path: str) -> dict:
    """Get role masks for reactions 4 and 5"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    roles = {}
    conn = None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for rxn_id in [4, 5]:
            cursor.execute(
                "SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?",
                (rxn_id,)
            )
            result = cursor.fetchone()
            if result:
                smarts, roleA, roleB, roleC = result
                roles[rxn_id] = {
                    'smarts': smarts,
                    'roleA': roleA,
                    'roleB': roleB,
                    'roleC': roleC,
                    'is_three_component': roleC is not None and roleC != 0
                }
                bt.logging.info(f"Reaction {rxn_id}: roleA={roleA}, roleB={roleB}, roleC={roleC}")
            else:
                bt.logging.warning(f"Reaction {rxn_id} not found in database")
    
    except Exception as e:
        bt.logging.error(f"Failed to get reaction roles: {e}")
        raise
    
    finally:
        if conn:
            conn.close()
    
    if not roles:
        raise ValueError("No reaction roles found in database")
    
    return roles
