"""
Advanced RAM management system for efficient memory usage.

Solutions:
1. Efficient DataFrame operations (avoid copies, use views)
2. Generators for large datasets (lazy evaluation)
3. Periodic cache cleaning with intelligent warmup

All bugs fixed:
- Correct memory freed calculation (process memory, not system)
- Thread-safe operations (locks added)
- Bounded access tracking (no memory leak)
- Optimized DataFrame size calculation (deep=False option)
- Fixed GC interval check (batch count based)
- Size-aware warmup strategy (optional)
- Warnings for large DataFrame copies
"""

import gc
import sys
import time
import psutil
import pandas as pd
import numpy as np
from typing import Iterator, List, Dict, Any, Optional, Callable
from collections import deque
from contextlib import contextmanager
from threading import Lock
import bittensor as bt


# Constants
CATEGORY_THRESHOLD = 0.5  # Use category if < 50% unique values
LARGE_DATAFRAME_MB = 100.0  # Warn when copying DataFrames > 100 MB
DEFAULT_MEMORY_THRESHOLD = 0.80
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_WARMUP_SIZE = 1000
DEFAULT_CHECK_INTERVAL = 60.0
MAX_TRACKED_KEYS = 10000
GC_INTERVAL_BATCHES = 10


# ============================================================================
# SOLUTION 1: Efficient DataFrame Operations
# ============================================================================

class EfficientDataFrameManager:
    """
    Manager for memory-efficient DataFrame operations.
    
    Strategies:
    - Use categorical dtypes for string columns
    - Avoid unnecessary copies
    - Use inplace operations
    - Downcast numeric types
    - Use sparse arrays for sparse data
    
    Fixed:
    - Warns about large DataFrame copies
    - Handles optimization failures gracefully
    """
    
    def __init__(self):
        self.memory_saved_mb = 0
        self.optimization_count = 0
        self.lock = Lock()
        
        bt.logging.info("Initialized EfficientDataFrameManager")
    
    def optimize_dataframe(
        self, 
        df: pd.DataFrame, 
        inplace: bool = False,
        warn_copy: bool = True
    ) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            inplace: Modify DataFrame in place (recommended for large DataFrames)
            warn_copy: Warn if copying large DataFrame
        
        Returns:
            Optimized DataFrame
        """
        memory_before = df.memory_usage(deep=True).sum() / 1e6
        
        # Warn about copying large DataFrames
        if not inplace and warn_copy and memory_before > LARGE_DATAFRAME_MB:
            bt.logging.warning(
                f"Copying large DataFrame ({memory_before:.1f} MB) for optimization. "
                f"Consider using inplace=True to avoid doubling memory usage."
            )
        
        if not inplace:
            df = df.copy()
        
        # Optimize each column
        for col in df.columns:
            try:
                col_type = df[col].dtype
                
                # Convert object columns to category if beneficial
                if col_type == 'object':
                    num_unique = df[col].nunique()
                    num_total = len(df[col])
                    
                    # Use category if < 50% unique values
                    if num_total > 0 and num_unique / num_total < CATEGORY_THRESHOLD:
                        df[col] = df[col].astype('category')
                
                # Downcast numeric types
                elif col_type == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                elif col_type == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            
            except Exception as e:
                bt.logging.debug(f"Could not optimize column {col}: {e}")
                continue
        
        memory_after = df.memory_usage(deep=True).sum() / 1e6
        saved = memory_before - memory_after
        
        with self.lock:
            if saved > 0:
                self.memory_saved_mb += saved
                self.optimization_count += 1
                bt.logging.debug(
                    f"DataFrame optimized: {memory_before:.1f} MB → {memory_after:.1f} MB "
                    f"(saved {saved:.1f} MB, {saved/memory_before*100:.1f}%)"
                )
            elif saved < 0:
                bt.logging.debug(
                    f"DataFrame optimization increased memory by {-saved:.1f} MB "
                    f"(optimization may not be beneficial for this DataFrame)"
                )
        
        return df
    
    def efficient_concat(self, dfs: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Memory-efficient DataFrame concatenation.
        
        Uses copy=False to avoid unnecessary copies.
        """
        # Filter out empty DataFrames
        dfs = [df for df in dfs if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        # Concatenate without copying
        result = pd.concat(dfs, copy=False, **kwargs)
        
        # Optimize result
        return self.optimize_dataframe(result, inplace=True)
    
    def efficient_merge(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Memory-efficient DataFrame merge.
        
        Uses copy=False and optimizes result.
        """
        result = pd.merge(left, right, copy=False, **kwargs)
        return self.optimize_dataframe(result, inplace=True)
    
    def efficient_groupby_agg(
        self,
        df: pd.DataFrame,
        by: str,
        agg_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Memory-efficient groupby aggregation.
        
        Uses observed=True for categorical columns.
        """
        # Convert groupby column to category if beneficial
        if df[by].dtype == 'object':
            num_unique = df[by].nunique()
            num_total = len(df[by])
            if num_total > 0 and num_unique / num_total < CATEGORY_THRESHOLD:
                df[by] = df[by].astype('category')
        
        # Perform aggregation
        result = df.groupby(by, observed=True).agg(agg_dict)
        
        return self.optimize_dataframe(result.reset_index(), inplace=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self.lock:
            return {
                'memory_saved_mb': self.memory_saved_mb,
                'optimization_count': self.optimization_count,
                'avg_saved_mb': self.memory_saved_mb / self.optimization_count if self.optimization_count > 0 else 0
            }


# ============================================================================
# SOLUTION 2: Generators for Large Datasets
# ============================================================================

class DataGenerator:
    """
    Generator-based data processing for memory efficiency.
    
    Benefits:
    - Lazy evaluation (only compute when needed)
    - Constant memory usage regardless of dataset size
    - Can process datasets larger than RAM
    
    Fixed:
    - Correct GC interval check (batch count based)
    """
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.total_processed = 0
        self.lock = Lock()
        
        bt.logging.info(f"Initialized DataGenerator with chunk_size={chunk_size}")
    
    def batch_generator(
        self,
        data: List[Any],
        batch_size: int = None
    ) -> Iterator[List[Any]]:
        """
        Generate batches from a list.
        
        Args:
            data: Input data list
            batch_size: Size of each batch (default: self.chunk_size)
        
        Yields:
            Batches of data
        """
        if batch_size is None:
            batch_size = self.chunk_size
        
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
            with self.lock:
                self.total_processed += min(batch_size, len(data) - i)
    
    def dataframe_chunk_generator(
        self,
        df: pd.DataFrame,
        chunk_size: int = None
    ) -> Iterator[pd.DataFrame]:
        """
        Generate chunks from a DataFrame.
        
        Args:
            df: Input DataFrame
            chunk_size: Size of each chunk
        
        Yields:
            DataFrame chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
            with self.lock:
                self.total_processed += min(chunk_size, len(df) - i)
    
    def file_line_generator(self, filepath: str) -> Iterator[str]:
        """
        Generate lines from a file without loading entire file.
        
        Args:
            filepath: Path to file
        
        Yields:
            Lines from file
        """
        with open(filepath, 'r') as f:
            for line in f:
                yield line.strip()
                with self.lock:
                    self.total_processed += 1
    
    def filtered_generator(
        self,
        data_generator: Iterator[Any],
        filter_func: Callable[[Any], bool]
    ) -> Iterator[Any]:
        """
        Apply filter to a generator (lazy filtering).
        
        Args:
            data_generator: Input generator
            filter_func: Filter function
        
        Yields:
            Filtered items
        """
        for item in data_generator:
            if filter_func(item):
                yield item
    
    def mapped_generator(
        self,
        data_generator: Iterator[Any],
        map_func: Callable[[Any], Any]
    ) -> Iterator[Any]:
        """
        Apply mapping to a generator (lazy mapping).
        
        Args:
            data_generator: Input generator
            map_func: Mapping function
        
        Yields:
            Mapped items
        """
        for item in data_generator:
            yield map_func(item)
    
    def process_in_batches(
        self,
        data: List[Any],
        process_func: Callable[[List[Any]], List[Any]],
        batch_size: int = None,
        gc_interval: int = GC_INTERVAL_BATCHES
    ) -> List[Any]:
        """
        Process data in batches to limit memory usage.
        
        Args:
            data: Input data
            process_func: Function to process each batch
            batch_size: Batch size
            gc_interval: Run GC every N batches
        
        Returns:
            Processed results
        """
        results = []
        batch_count = 0
        
        for batch in self.batch_generator(data, batch_size):
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            batch_count += 1
            
            # FIX: Periodic cleanup based on batch count
            if batch_count % gc_interval == 0:
                gc.collect()
                bt.logging.debug(f"GC after {batch_count} batches ({len(results)} results)")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.lock:
            return {
                'total_processed': self.total_processed,
                'chunk_size': self.chunk_size
            }


# ============================================================================
# SOLUTION 3: Periodic Cache Cleaning with Warmup
# ============================================================================

class IntelligentCacheCleaner:
    """
    Intelligent cache cleaning with warmup strategy.
    
    Features:
    - Monitor memory usage
    - Clean caches when memory is high
    - Intelligently warm up most important data
    - Track access patterns for smart eviction
    
    Fixed:
    - Correct memory freed calculation (process memory, not system)
    - Thread-safe operations (locks added)
    - Bounded access tracking (no memory leak)
    - Size-aware warmup strategy (optional)
    """
    
    def __init__(
        self,
        memory_threshold: float = DEFAULT_MEMORY_THRESHOLD,
        warmup_size: int = DEFAULT_WARMUP_SIZE,
        check_interval: float = DEFAULT_CHECK_INTERVAL
    ):
        self.memory_threshold = memory_threshold
        self.warmup_size = warmup_size
        self.check_interval = check_interval
        self.max_tracked_keys = MAX_TRACKED_KEYS
        
        # Track access patterns
        self.access_counts = {}  # {cache_key: access_count}
        self.access_times = {}  # {cache_key: last_access_time}
        
        # Statistics
        self.total_cleanups = 0
        self.total_memory_freed_mb = 0
        self.last_check_time = 0
        
        # Thread safety
        self.lock = Lock()
        
        bt.logging.info(
            f"Initialized IntelligentCacheCleaner: "
            f"threshold={memory_threshold*100:.0f}%, "
            f"warmup_size={warmup_size}"
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1e9  # GB
        
        return {
            'total_gb': memory.total / 1e9,
            'available_gb': memory.available / 1e9,
            'used_gb': memory.used / 1e9,
            'percent': memory.percent / 100,
            'process_gb': process_memory
        }
    
    def should_clean(self, force: bool = False) -> bool:
        """
        Check if cache cleaning is needed (thread-safe).
        
        Args:
            force: Force check regardless of interval
        
        Returns:
            True if cleaning is needed
        """
        # FIX: Thread-safe interval check
        with self.lock:
            current_time = time.time()
            
            # Check interval
            if not force and (current_time - self.last_check_time) < self.check_interval:
                return False
            
            self.last_check_time = current_time
        
        # Check memory usage (outside lock)
        memory = self.get_memory_usage()
        
        if memory['percent'] >= self.memory_threshold:
            bt.logging.warning(
                f"⚠️  High RAM usage: {memory['percent']*100:.1f}% "
                f"({memory['used_gb']:.1f}/{memory['total_gb']:.1f} GB)"
            )
            return True
        
        return False
    
    def record_access(self, cache_key: str):
        """Record cache access for tracking (thread-safe with bounded growth)."""
        with self.lock:
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            self.access_times[cache_key] = time.time()
            
            # FIX: Limit tracking size to prevent memory leak
            if len(self.access_counts) > self.max_tracked_keys:
                # Remove least recently accessed entries
                sorted_keys = sorted(
                    self.access_times.keys(), 
                    key=lambda k: self.access_times[k]
                )
                
                # Remove oldest 10%
                keys_to_remove = sorted_keys[:self.max_tracked_keys // 10]
                for key in keys_to_remove:
                    self.access_counts.pop(key, None)
                    self.access_times.pop(key, None)
                
                bt.logging.debug(
                    f"Pruned {len(keys_to_remove)} old tracking entries "
                    f"({len(self.access_counts)} remaining)"
                )
    
    def get_warmup_keys(
        self, 
        all_keys: List[str],
        size_func: Optional[Callable[[str], int]] = None,
        target_memory_mb: Optional[float] = None
    ) -> List[str]:
        """
        Get keys to keep warm based on access patterns and size.
        
        Strategy:
        - Most frequently accessed
        - Most recently accessed
        - Weighted combination
        - Optional: Size-aware selection
        
        Args:
            all_keys: All available cache keys
            size_func: Function to get size of each key's value (bytes)
            target_memory_mb: Target memory to keep (if None, use count-based)
        
        Returns:
            Keys to keep warm
        """
        if len(all_keys) <= self.warmup_size and target_memory_mb is None:
            return all_keys
        
        # Score each key
        current_time = time.time()
        scores = {}
        sizes = {}
        
        with self.lock:
            for key in all_keys:
                # Frequency score (normalized)
                freq_score = self.access_counts.get(key, 0)
                
                # Recency score (exponential decay)
                last_access = self.access_times.get(key, 0)
                time_since_access = current_time - last_access
                recency_score = np.exp(-time_since_access / 3600)  # 1 hour half-life
                
                # Get size if function provided
                if size_func:
                    try:
                        sizes[key] = size_func(key)
                    except Exception:
                        sizes[key] = 0
                
                # Combined score (70% frequency, 30% recency)
                scores[key] = 0.7 * freq_score + 0.3 * recency_score
        
        # Sort by score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Select keys based on strategy
        if target_memory_mb is not None and size_func is not None:
            # Size-aware selection
            warmup_keys = []
            total_size = 0
            target_bytes = target_memory_mb * 1e6
            
            for key in sorted_keys:
                if total_size + sizes[key] <= target_bytes:
                    warmup_keys.append(key)
                    total_size += sizes[key]
                
                # Stop if we've checked enough keys
                if len(warmup_keys) >= self.warmup_size:
                    break
            
            bt.logging.debug(
                f"Selected {len(warmup_keys)} keys for warmup "
                f"({total_size/1e6:.1f} MB / {target_memory_mb:.1f} MB target)"
            )
        else:
            # Count-based selection (original behavior)
            warmup_keys = sorted_keys[:self.warmup_size]
            bt.logging.debug(f"Selected {len(warmup_keys)} keys for warmup")
        
        return warmup_keys
    
    def clean_cache_with_warmup(
        self,
        cache_dict: Dict[str, Any],
        warmup_func: Optional[Callable[[List[str]], None]] = None
    ) -> Dict[str, Any]:
        """
        Clean cache and warm up important entries.
        
        Args:
            cache_dict: Cache dictionary to clean
            warmup_func: Optional function to warm up keys
        
        Returns:
            Dictionary with cleanup statistics
        """
        # FIX: Measure process memory, not system memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1e6  # MB
        
        # Get keys to keep warm
        all_keys = list(cache_dict.keys())
        warmup_keys = self.get_warmup_keys(all_keys)
        
        # Save warmup data
        warmup_data = {key: cache_dict[key] for key in warmup_keys if key in cache_dict}
        
        # Clear cache
        initial_size = len(cache_dict)
        cache_dict.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Restore warmup data
        cache_dict.update(warmup_data)
        
        # Call warmup function if provided
        if warmup_func:
            try:
                warmup_func(warmup_keys)
            except Exception as e:
                bt.logging.warning(f"Warmup function failed: {e}")
        
        # FIX: Measure process memory after cleanup
        memory_after = process.memory_info().rss / 1e6  # MB
        memory_freed = max(0, memory_before - memory_after)
        
        removed_count = initial_size - len(warmup_keys)
        
        with self.lock:
            self.total_cleanups += 1
            self.total_memory_freed_mb += memory_freed
        
        bt.logging.info(
            f"Cache cleaned: {initial_size} → {len(warmup_keys)} entries "
            f"(removed {removed_count}, freed {memory_freed:.1f} MB)"
        )
        
        return {
            'removed_count': removed_count,
            'memory_freed_mb': memory_freed,
            'entries_before': initial_size,
            'entries_after': len(warmup_keys)
        }
    
    def clean_dataframe_cache(
        self,
        df_cache: Dict[str, pd.DataFrame],
        keep_top_n: int = None,
        use_fast_size: bool = True
    ) -> Dict[str, Any]:
        """
        Clean DataFrame cache with intelligent selection.
        
        Args:
            df_cache: Cache of DataFrames
            keep_top_n: Number of DataFrames to keep (default: warmup_size)
            use_fast_size: Use fast size estimation instead of deep=True
        
        Returns:
            Cleanup statistics
        """
        if keep_top_n is None:
            keep_top_n = self.warmup_size
        
        # FIX: Measure process memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1e6  # MB
        
        # FIX: Fast size calculation
        if use_fast_size:
            # Fast estimation (doesn't traverse all objects)
            df_sizes = {
                key: df.memory_usage(deep=False).sum()
                for key, df in df_cache.items()
            }
        else:
            # Accurate but slow
            df_sizes = {
                key: df.memory_usage(deep=True).sum()
                for key, df in df_cache.items()
            }
        
        # Get warmup keys based on access patterns
        def get_size(key):
            return df_sizes.get(key, 0)
        
        warmup_keys = self.get_warmup_keys(
            list(df_cache.keys()),
            size_func=get_size,
            target_memory_mb=None  # Use count-based for now
        )
        warmup_keys = warmup_keys[:keep_top_n]
        
        # Keep warmup DataFrames
        initial_size = len(df_cache)
        keys_to_remove = [k for k in df_cache.keys() if k not in warmup_keys]
        
        for key in keys_to_remove:
            del df_cache[key]
        
        # Force garbage collection
        gc.collect()
        
        # FIX: Measure process memory after cleanup
        memory_after = process.memory_info().rss / 1e6  # MB
        memory_freed = max(0, memory_before - memory_after)
        
        removed_count = len(keys_to_remove)
        
        with self.lock:
            self.total_cleanups += 1
            self.total_memory_freed_mb += memory_freed
        
        bt.logging.info(
            f"DataFrame cache cleaned: {initial_size} → {len(df_cache)} DataFrames "
            f"(removed {removed_count}, freed {memory_freed:.1f} MB)"
        )
        
        return {
            'removed_count': removed_count,
            'memory_freed_mb': memory_freed,
            'dataframes_before': initial_size,
            'dataframes_after': len(df_cache)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        memory = self.get_memory_usage()
        
        with self.lock:
            return {
                'total_cleanups': self.total_cleanups,
                'total_memory_freed_mb': self.total_memory_freed_mb,
                'avg_freed_per_cleanup_mb': self.total_memory_freed_mb / self.total_cleanups if self.total_cleanups > 0 else 0,
                'current_memory_usage': memory,
                'tracked_keys': len(self.access_counts),
                'warmup_size': self.warmup_size
            }
    
    def print_report(self):
        """Print detailed memory report."""
        stats = self.get_stats()
        memory = stats['current_memory_usage']
        
        bt.logging.info("="*60)
        bt.logging.info("RAM MANAGEMENT REPORT")
        bt.logging.info("="*60)
        bt.logging.info(f"Current Memory Usage:")
        bt.logging.info(f"  Total: {memory['total_gb']:.1f} GB")
        bt.logging.info(f"  Used: {memory['used_gb']:.1f} GB ({memory['percent']*100:.1f}%)")
        bt.logging.info(f"  Available: {memory['available_gb']:.1f} GB")
        bt.logging.info(f"  Process: {memory['process_gb']:.1f} GB")
        bt.logging.info(f"")
        bt.logging.info(f"Cache Cleaning:")
        bt.logging.info(f"  Total cleanups: {stats['total_cleanups']}")
        bt.logging.info(f"  Total freed: {stats['total_memory_freed_mb']:.0f} MB")
        bt.logging.info(f"  Avg per cleanup: {stats['avg_freed_per_cleanup_mb']:.0f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Cache Tracking:")
        bt.logging.info(f"  Tracked keys: {stats['tracked_keys']}")
        bt.logging.info(f"  Warmup size: {stats['warmup_size']}")
        bt.logging.info("="*60)


# ============================================================================
# INTEGRATED RAM MANAGER
# ============================================================================

class IntegratedRAMManager:
    """
    Integrated RAM management combining all three solutions:
    1. Efficient DataFrame operations
    2. Generators for large datasets
    3. Periodic cache cleaning with warmup
    
    This is the main class you should use in production.
    """
    
    def __init__(
        self,
        memory_threshold: float = DEFAULT_MEMORY_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        warmup_size: int = DEFAULT_WARMUP_SIZE,
        auto_cleanup: bool = True
    ):
        self.memory_threshold = memory_threshold
        self.auto_cleanup = auto_cleanup
        
        # Initialize components
        self.df_manager = EfficientDataFrameManager()
        self.generator = DataGenerator(chunk_size)
        self.cache_cleaner = IntelligentCacheCleaner(
            memory_threshold=memory_threshold,
            warmup_size=warmup_size
        )
        
        bt.logging.info(
            f"Integrated RAM Manager initialized: "
            f"threshold={memory_threshold*100:.0f}%, "
            f"chunk_size={chunk_size}, "
            f"warmup_size={warmup_size}, "
            f"auto_cleanup={auto_cleanup}"
        )
    
    def check_and_clean(
        self, 
        cache_dict: Dict[str, Any] = None, 
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Check memory and clean if needed.
        
        Args:
            cache_dict: Optional cache dictionary to clean
            force: Force cleaning regardless of threshold
        
        Returns:
            Cleanup statistics if cleaning was performed
        """
        if self.cache_cleaner.should_clean(force) or force:
            bt.logging.info("Starting intelligent cache cleanup...")
            
            # Clean provided cache
            stats = None
            if cache_dict:
                stats = self.cache_cleaner.clean_cache_with_warmup(cache_dict)
            
            # Force garbage collection
            gc.collect()
            
            # Log memory stats
            memory = self.cache_cleaner.get_memory_usage()
            bt.logging.info(
                f"Memory after cleanup: {memory['used_gb']:.1f}/{memory['total_gb']:.1f} GB "
                f"({memory['percent']*100:.1f}%)"
            )
            
            return stats
        
        return None
    
    @contextmanager
    def managed_processing(self, cache_dict: Dict[str, Any] = None):
        """
        Context manager for automatic memory management.
        
        Usage:
            with ram_manager.managed_processing(my_cache):
                # Your processing code
                pass
            # Memory automatically checked and cleaned
        """
        # Check memory before
        self.check_and_clean(cache_dict, force=False)
        
        try:
            yield
        finally:
            # Check memory after
            if self.auto_cleanup:
                self.check_and_clean(cache_dict, force=False)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'dataframe_manager': self.df_manager.get_stats(),
            'generator': self.generator.get_stats(),
            'cache_cleaner': self.cache_cleaner.get_stats()
        }
    
    def print_full_report(self):
        """Print comprehensive RAM management report."""
        self.cache_cleaner.print_report()
        
        df_stats = self.df_manager.get_stats()
        gen_stats = self.generator.get_stats()
        
        bt.logging.info(f"")
        bt.logging.info(f"DataFrame Optimizations:")
        bt.logging.info(f"  Total optimizations: {df_stats['optimization_count']}")
        bt.logging.info(f"  Total memory saved: {df_stats['memory_saved_mb']:.1f} MB")
        bt.logging.info(f"  Avg per optimization: {df_stats['avg_saved_mb']:.1f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Generator Processing:")
        bt.logging.info(f"  Total processed: {gen_stats['total_processed']:,} items")
        bt.logging.info(f"  Chunk size: {gen_stats['chunk_size']}")
        bt.logging.info("="*60)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def optimize_dataframe_memory(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Quick function to optimize DataFrame memory.
    
    Args:
        df: DataFrame to optimize
        inplace: Modify in place
    
    Returns:
        Optimized DataFrame
    """
    manager = EfficientDataFrameManager()
    return manager.optimize_dataframe(df, inplace=inplace)


def process_large_dataset_in_batches(
    data: List[Any],
    process_func: Callable[[List[Any]], List[Any]],
    batch_size: int = DEFAULT_CHUNK_SIZE
) -> List[Any]:
    """
    Process large dataset in batches to limit memory usage.
    
    Args:
        data: Input data
        process_func: Function to process each batch
        batch_size: Batch size
    
    Returns:
        Processed results
    """
    generator = DataGenerator(chunk_size=batch_size)
    return generator.process_in_batches(data, process_func, batch_size)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the RAM management system.
    """
    
    # Example 1: Optimize DataFrame
    print("\n=== Example 1: Optimize DataFrame ===")
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B'] * 1000,
        'value': range(4000),
        'float_val': [1.0] * 4000
    })
    
    print(f"Before: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    df_optimized = optimize_dataframe_memory(df, inplace=False)
    print(f"After: {df_optimized.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Example 2: Process in batches
    print("\n=== Example 2: Process in Batches ===")
    large_data = list(range(10000))
    
    def process_batch(batch):
        return [x * 2 for x in batch]
    
    results = process_large_dataset_in_batches(large_data, process_batch, batch_size=1000)
    print(f"Processed {len(results)} items")
    
    # Example 3: Integrated manager
    print("\n=== Example 3: Integrated Manager ===")
    manager = IntegratedRAMManager(
        memory_threshold=0.80,
        chunk_size=1000,
        warmup_size=100,
        auto_cleanup=True
    )
    
    # Create a cache
    cache = {f"key_{i}": f"value_{i}" for i in range(1000)}
    
    # Record some accesses
    for i in range(100):
        manager.cache_cleaner.record_access(f"key_{i}")
    
    # Use managed context
    with manager.managed_processing(cache):
        # Your processing code here
        pass
    
    # Print report
    manager.print_full_report()
