"""
Advanced GPU memory management system.

Solutions:
1. Aggressive memory cleanup
2. Context managers for automatic cleanup
3. Proactive memory monitoring

All bugs fixed:
- Memory leak in history tracking (now uses deque)
- Correct memory freed calculation (measures reserved + allocated)
- Thread-safe operations (locks added)
- Accurate memory leak detection (linear regression + R²)
- Efficient statistics calculation (cached averages)
"""

import gc
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps
from collections import deque
from threading import Lock
import bittensor as bt


# Constants
MB_TO_GB = 1024.0  # Binary GB (GiB)
CLEANUP_THRESHOLD_MB = 100.0
DEFAULT_WARNING_THRESHOLD = 0.8
DEFAULT_CRITICAL_THRESHOLD = 0.95
DEFAULT_CHECK_INTERVAL = 1.0
MAX_HISTORY_SIZE = 1000


# ============================================================================
# SOLUTION 1: Aggressive Memory Cleanup
# ============================================================================

class AggressiveMemoryManager:
    """
    Aggressive GPU memory cleanup manager.
    
    Strategies:
    - Clear CUDA cache after every operation
    - Force garbage collection
    - Clear unused tensors
    - Reset peak memory stats
    
    Fixed:
    - Correctly measures both cache and tensor memory freed
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.cleanup_count = 0
        self.total_freed_mb = 0
        self.total_cache_freed_mb = 0
        self.total_tensors_freed_mb = 0
        
        if torch.cuda.is_available():
            bt.logging.info(f"GPU {device_id}: Aggressive memory management enabled")
    
    def cleanup(self, force: bool = False) -> float:
        """
        Perform aggressive memory cleanup.
        
        Args:
            force: Force cleanup even if not much memory is used
        
        Returns:
            Amount of memory freed (MB) - includes both cache and tensors
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Measure reserved (cached) and allocated memory before cleanup
        reserved_before = torch.cuda.memory_reserved(self.device_id) / 1e6
        allocated_before = torch.cuda.memory_allocated(self.device_id) / 1e6
        
        # Step 1: Python garbage collection (frees unreferenced tensors)
        gc.collect()
        
        # Step 2: Clear PyTorch cache (frees cached allocations)
        torch.cuda.empty_cache()
        
        # Step 3: Synchronize to ensure all operations complete
        torch.cuda.synchronize(self.device_id)
        
        # Step 4: Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(self.device_id)
        
        # Measure memory after cleanup
        reserved_after = torch.cuda.memory_reserved(self.device_id) / 1e6
        allocated_after = torch.cuda.memory_allocated(self.device_id) / 1e6
        
        # Calculate freed memory
        cache_freed = max(0, reserved_before - reserved_after)
        tensors_freed = max(0, allocated_before - allocated_after)
        total_freed = cache_freed + tensors_freed
        
        if total_freed > 0:
            self.cleanup_count += 1
            self.total_freed_mb += total_freed
            self.total_cache_freed_mb += cache_freed
            self.total_tensors_freed_mb += tensors_freed
            
            bt.logging.debug(
                f"GPU {self.device_id}: Freed {total_freed:.1f} MB "
                f"(cache: {cache_freed:.1f} MB, tensors: {tensors_freed:.1f} MB)"
            )
        
        return total_freed
    
    def cleanup_if_needed(self, threshold_mb: float = CLEANUP_THRESHOLD_MB) -> bool:
        """
        Cleanup only if allocated memory exceeds threshold.
        
        Args:
            threshold_mb: Cleanup if more than this amount is allocated
        
        Returns:
            True if cleanup was performed
        """
        if not torch.cuda.is_available():
            return False
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1e6
        
        if allocated > threshold_mb:
            self.cleanup(force=True)
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return {
            'device_id': self.device_id,
            'cleanup_count': self.cleanup_count,
            'total_freed_mb': self.total_freed_mb,
            'total_cache_freed_mb': self.total_cache_freed_mb,
            'total_tensors_freed_mb': self.total_tensors_freed_mb,
            'avg_freed_mb': self.total_freed_mb / self.cleanup_count if self.cleanup_count > 0 else 0
        }


# ============================================================================
# SOLUTION 2: Context Managers for Automatic Cleanup
# ============================================================================

@contextmanager
def gpu_memory_context(device_id: int = 0, cleanup_on_exit: bool = True):
    """
    Context manager for automatic GPU memory cleanup.
    
    Usage:
        with gpu_memory_context(device_id=0):
            # Your GPU operations here
            output = model(input)
        # Memory automatically cleaned up here
    
    Args:
        device_id: GPU device ID
        cleanup_on_exit: Whether to cleanup on exit
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Record initial memory state
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_id)
        allocated_before = torch.cuda.memory_allocated(device_id) / 1e6
        reserved_before = torch.cuda.memory_reserved(device_id) / 1e6
    else:
        allocated_before = 0
        reserved_before = 0
    
    try:
        yield device
    finally:
        if cleanup_on_exit and torch.cuda.is_available():
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_id)
            
            # Measure memory after cleanup
            allocated_after = torch.cuda.memory_allocated(device_id) / 1e6
            reserved_after = torch.cuda.memory_reserved(device_id) / 1e6
            
            allocated_change = allocated_after - allocated_before
            cache_freed = max(0, reserved_before - reserved_after)
            
            bt.logging.debug(
                f"GPU {device_id}: Context cleanup - "
                f"allocated change: {allocated_change:+.1f} MB, "
                f"cache freed: {cache_freed:.1f} MB, "
                f"current: {allocated_after:.1f} MB"
            )


@contextmanager
def inference_memory_context(device_id: int = 0, model: Optional[torch.nn.Module] = None):
    """
    Context manager optimized for inference operations.
    
    Features:
    - Sets model to eval mode
    - Disables gradient computation
    - Automatic cleanup on exit
    
    Usage:
        with inference_memory_context(device_id=0, model=my_model) as device:
            with torch.no_grad():
                output = model(input)
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Set model to eval mode if provided
    was_training = False
    if model is not None:
        was_training = model.training
        model.eval()
    
    # Record memory
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated(device_id) / 1e6
        reserved_before = torch.cuda.memory_reserved(device_id) / 1e6
    else:
        allocated_before = 0
        reserved_before = 0
    
    try:
        yield device
    finally:
        # Restore model state
        if model is not None and was_training:
            model.train()
        
        # Aggressive cleanup
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_id)
            
            allocated_after = torch.cuda.memory_allocated(device_id) / 1e6
            reserved_after = torch.cuda.memory_reserved(device_id) / 1e6
            
            allocated_change = allocated_after - allocated_before
            cache_freed = max(0, reserved_before - reserved_after)
            
            bt.logging.debug(
                f"GPU {device_id}: Inference used {allocated_change:+.1f} MB, "
                f"cache freed: {cache_freed:.1f} MB, "
                f"current: {allocated_after:.1f} MB"
            )


def auto_cleanup_decorator(cleanup_threshold_mb: float = 500.0):
    """
    Decorator for automatic memory cleanup after function execution.
    
    Usage:
        @auto_cleanup_decorator(cleanup_threshold_mb=500.0)
        def my_gpu_function(data):
            # GPU operations
            return result
    
    Args:
        cleanup_threshold_mb: Cleanup if allocated memory exceeds this
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Check if cleanup needed
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(device_id) / 1e6
                    if allocated > cleanup_threshold_mb:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device_id)
                        bt.logging.debug(f"GPU {device_id}: Auto-cleanup after {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# SOLUTION 3: Proactive Memory Monitoring
# ============================================================================

class GPUMemoryMonitor:
    """
    Proactive GPU memory monitoring system.
    
    Features:
    - Real-time memory tracking
    - Automatic alerts on high usage
    - Memory leak detection (with linear regression)
    - Historical statistics
    - Thread-safe operations
    
    Fixed:
    - Uses deque to prevent memory leak in history tracking
    - Thread-safe with locks
    - Accurate leak detection with R² validation
    - Efficient statistics calculation
    """
    
    def __init__(
        self,
        device_id: int = 0,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        check_interval: float = DEFAULT_CHECK_INTERVAL
    ):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        
        # Memory tracking (using deque to prevent memory leak)
        self.memory_history = deque(maxlen=MAX_HISTORY_SIZE)
        
        # Statistics
        self.warning_count = 0
        self.critical_count = 0
        self.last_check_time = 0
        
        # Thread safety
        self.lock = Lock()
        
        # Cached statistics (for efficient calculation)
        self._cached_avg = None
        self._last_history_size = 0
        
        # Get total memory
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            self.total_memory = props.total_memory / 1e6  # MB
            bt.logging.info(
                f"GPU {device_id} Memory Monitor initialized: "
                f"{self.total_memory:.0f} MB total, "
                f"warning at {warning_threshold*100:.0f}%, "
                f"critical at {critical_threshold*100:.0f}%"
            )
        else:
            self.total_memory = 0
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0,
                'utilization': 0,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0
            }
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1e6
        reserved = torch.cuda.memory_reserved(self.device_id) / 1e6
        free = self.total_memory - reserved
        utilization = reserved / self.total_memory if self.total_memory > 0 else 0
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'utilization': utilization,
            'allocated_gb': allocated / MB_TO_GB,
            'reserved_gb': reserved / MB_TO_GB,
            'free_gb': free / MB_TO_GB
        }
    
    def check_and_alert(self, force: bool = False) -> Optional[str]:
        """
        Check memory usage and alert if thresholds exceeded.
        Thread-safe implementation.
        
        Args:
            force: Force check even if interval hasn't elapsed
        
        Returns:
            Alert level ('warning', 'critical', or None)
        """
        # Thread-safe interval check
        with self.lock:
            current_time = time.time()
            
            # Check interval
            if not force and (current_time - self.last_check_time) < self.check_interval:
                return None
            
            self.last_check_time = current_time
        
        if not torch.cuda.is_available():
            return None
        
        # Get current usage
        usage = self.get_current_usage()
        utilization = usage['utilization']
        
        # Record history (deque automatically handles size limit)
        with self.lock:
            self.memory_history.append((
                current_time,
                usage['allocated_mb'],
                usage['reserved_mb']
            ))
            
            # Invalidate cached average
            self._cached_avg = None
        
        # Check thresholds
        alert_level = None
        
        if utilization >= self.critical_threshold:
            with self.lock:
                self.critical_count += 1
            alert_level = 'critical'
            bt.logging.error(
                f"🔴 GPU {self.device_id} CRITICAL: {utilization*100:.1f}% memory used "
                f"({usage['reserved_mb']:.0f}/{self.total_memory:.0f} MB)"
            )
        elif utilization >= self.warning_threshold:
            with self.lock:
                self.warning_count += 1
            alert_level = 'warning'
            bt.logging.warning(
                f"⚠️  GPU {self.device_id} WARNING: {utilization*100:.1f}% memory used "
                f"({usage['reserved_mb']:.0f}/{self.total_memory:.0f} MB)"
            )
        
        return alert_level
    
    def detect_memory_leak(
        self, 
        window_size: int = 100, 
        threshold_slope: float = 1.0
    ) -> bool:
        """
        Detect potential memory leaks using linear regression.
        
        Uses R² to validate that the trend is real (not just noise).
        
        Args:
            window_size: Number of samples to analyze
            threshold_slope: Alert if memory increases by this much per sample (MB)
        
        Returns:
            True if potential leak detected (consistent upward trend)
        """
        with self.lock:
            if len(self.memory_history) < window_size:
                return False
            
            # Get recent history (create copy to release lock quickly)
            recent = list(self.memory_history)[-window_size:]
        
        # Extract allocated memory values
        allocated_values = np.array([h[1] for h in recent])
        
        # Handle constant values
        if np.std(allocated_values) < 1e-6:
            return False
        
        # Fit linear trend
        x = np.arange(len(allocated_values))
        
        try:
            slope, intercept = np.polyfit(x, allocated_values, 1)
        except np.linalg.LinAlgError:
            return False
        
        # Check if slope exceeds threshold (consistent increase)
        if slope > threshold_slope:
            # Additional check: R² (how well trend fits)
            y_pred = slope * x + intercept
            ss_res = np.sum((allocated_values - y_pred) ** 2)
            ss_tot = np.sum((allocated_values - np.mean(allocated_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Only alert if trend is strong (R² > 0.7)
            if r_squared > 0.7:
                total_increase = slope * len(allocated_values)
                bt.logging.warning(
                    f"⚠️  GPU {self.device_id} Potential memory leak detected: "
                    f"Memory increasing at {slope:.2f} MB/sample "
                    f"(total: {total_increase:.1f} MB over {window_size} samples, R²={r_squared:.2f})"
                )
                return True
        
        return False
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage since last reset."""
        if not torch.cuda.is_available():
            return {'peak_allocated_mb': 0, 'peak_reserved_mb': 0}
        
        peak_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1e6
        peak_reserved = torch.cuda.max_memory_reserved(self.device_id) / 1e6
        
        return {
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'peak_allocated_gb': peak_allocated / MB_TO_GB,
            'peak_reserved_gb': peak_reserved / MB_TO_GB
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics (with efficient caching)."""
        current = self.get_current_usage()
        peak = self.get_peak_usage()
        
        # Calculate average with caching (efficient)
        with self.lock:
            history_size = len(self.memory_history)
            
            # Recalculate only if history changed
            if self._cached_avg is None or history_size != self._last_history_size:
                if self.memory_history:
                    allocated_sum = sum(h[1] for h in self.memory_history)
                    reserved_sum = sum(h[2] for h in self.memory_history)
                    self._cached_avg = {
                        'allocated': allocated_sum / history_size,
                        'reserved': reserved_sum / history_size
                    }
                    self._last_history_size = history_size
                else:
                    self._cached_avg = {'allocated': 0, 'reserved': 0}
            
            avg_allocated = self._cached_avg['allocated']
            avg_reserved = self._cached_avg['reserved']
            warning_count = self.warning_count
            critical_count = self.critical_count
        
        return {
            'device_id': self.device_id,
            'total_memory_mb': self.total_memory,
            'current': current,
            'peak': peak,
            'average': {
                'allocated_mb': avg_allocated,
                'reserved_mb': avg_reserved
            },
            'alerts': {
                'warning_count': warning_count,
                'critical_count': critical_count
            },
            'history_size': history_size
        }
    
    def reset_stats(self):
        """Reset monitoring statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
        
        with self.lock:
            self.memory_history.clear()
            self.warning_count = 0
            self.critical_count = 0
            self._cached_avg = None
            self._last_history_size = 0
        
        bt.logging.info(f"GPU {self.device_id}: Memory monitor stats reset")
    
    def print_report(self):
        """Print detailed memory report."""
        stats = self.get_stats()
        
        bt.logging.info("="*60)
        bt.logging.info(f"GPU {self.device_id} MEMORY REPORT")
        bt.logging.info("="*60)
        bt.logging.info(f"Total Memory: {stats['total_memory_mb']:.0f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Current Usage:")
        bt.logging.info(f"  Allocated: {stats['current']['allocated_mb']:.1f} MB ({stats['current']['utilization']*100:.1f}%)")
        bt.logging.info(f"  Reserved:  {stats['current']['reserved_mb']:.1f} MB")
        bt.logging.info(f"  Free:      {stats['current']['free_mb']:.1f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Peak Usage:")
        bt.logging.info(f"  Allocated: {stats['peak']['peak_allocated_mb']:.1f} MB")
        bt.logging.info(f"  Reserved:  {stats['peak']['peak_reserved_mb']:.1f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Average Usage:")
        bt.logging.info(f"  Allocated: {stats['average']['allocated_mb']:.1f} MB")
        bt.logging.info(f"  Reserved:  {stats['average']['reserved_mb']:.1f} MB")
        bt.logging.info(f"")
        bt.logging.info(f"Alerts:")
        bt.logging.info(f"  Warnings:  {stats['alerts']['warning_count']}")
        bt.logging.info(f"  Critical:  {stats['alerts']['critical_count']}")
        bt.logging.info("="*60)


# ============================================================================
# INTEGRATED MEMORY MANAGER
# ============================================================================

class IntegratedGPUMemoryManager:
    """
    Integrated GPU memory manager combining all three solutions:
    1. Aggressive cleanup
    2. Context managers
    3. Proactive monitoring
    
    This is the main class you should use in production.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_monitoring: bool = True,
        enable_aggressive_cleanup: bool = True,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
    ):
        self.device_id = device_id
        self.enable_monitoring = enable_monitoring
        self.enable_aggressive_cleanup = enable_aggressive_cleanup
        
        # Initialize components
        self.cleanup_manager = AggressiveMemoryManager(device_id) if enable_aggressive_cleanup else None
        self.monitor = GPUMemoryMonitor(
            device_id,
            warning_threshold,
            critical_threshold
        ) if enable_monitoring else None
        
        bt.logging.info(
            f"Integrated GPU Memory Manager initialized for GPU {device_id}: "
            f"monitoring={enable_monitoring}, aggressive_cleanup={enable_aggressive_cleanup}"
        )
    
    def cleanup(self, force: bool = False) -> float:
        """Perform memory cleanup."""
        if self.cleanup_manager:
            return self.cleanup_manager.cleanup(force)
        return 0.0
    
    def check_memory(self, force: bool = False) -> Optional[str]:
        """Check memory and alert if needed."""
        if self.monitor:
            alert = self.monitor.check_and_alert(force)
            
            # Auto-cleanup on critical alert
            if alert == 'critical' and self.cleanup_manager:
                freed = self.cleanup_manager.cleanup(force=True)
                bt.logging.info(f"Auto-cleanup freed {freed:.1f} MB after critical alert")
            
            return alert
        return None
    
    @contextmanager
    def managed_context(self, model: Optional[torch.nn.Module] = None):
        """
        Combined context manager with monitoring and cleanup.
        
        Usage:
            with manager.managed_context(model=my_model):
                output = model(input)
        """
        # Check memory before
        if self.monitor:
            self.monitor.check_and_alert(force=True)
        
        # Use inference context
        with inference_memory_context(self.device_id, model) as device:
            yield device
        
        # Check memory after
        if self.monitor:
            self.monitor.check_and_alert(force=True)
        
        # Cleanup if needed
        if self.cleanup_manager:
            self.cleanup_manager.cleanup_if_needed(threshold_mb=CLEANUP_THRESHOLD_MB)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        stats = {
            'device_id': self.device_id,
            'monitoring_enabled': self.enable_monitoring,
            'aggressive_cleanup_enabled': self.enable_aggressive_cleanup
        }
        
        if self.cleanup_manager:
            stats['cleanup'] = self.cleanup_manager.get_stats()
        
        if self.monitor:
            stats['monitoring'] = self.monitor.get_stats()
        
        return stats
    
    def print_full_report(self):
        """Print comprehensive memory report."""
        if self.monitor:
            self.monitor.print_report()
        
        if self.cleanup_manager:
            cleanup_stats = self.cleanup_manager.get_stats()
            bt.logging.info(f"")
            bt.logging.info(f"Cleanup Statistics:")
            bt.logging.info(f"  Total cleanups: {cleanup_stats['cleanup_count']}")
            bt.logging.info(f"  Total freed: {cleanup_stats['total_freed_mb']:.1f} MB")
            bt.logging.info(f"  Cache freed: {cleanup_stats['total_cache_freed_mb']:.1f} MB")
            bt.logging.info(f"  Tensors freed: {cleanup_stats['total_tensors_freed_mb']:.1f} MB")
            bt.logging.info(f"  Avg per cleanup: {cleanup_stats['avg_freed_mb']:.1f} MB")
            bt.logging.info("="*60)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_gpu_memory_managers(
    num_gpus: int = None,
    enable_monitoring: bool = True,
    enable_aggressive_cleanup: bool = True,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
) -> Dict[int, IntegratedGPUMemoryManager]:
    """
    Create integrated memory managers for all GPUs.
    
    Args:
        num_gpus: Number of GPUs (None = all available)
        enable_monitoring: Enable proactive monitoring
        enable_aggressive_cleanup: Enable aggressive cleanup
        warning_threshold: Warning threshold (0.0-1.0)
        critical_threshold: Critical threshold (0.0-1.0)
    
    Returns:
        Dictionary mapping GPU ID to IntegratedGPUMemoryManager
    """
    if not torch.cuda.is_available():
        bt.logging.warning("No CUDA GPUs available")
        return {}
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    managers = {}
    for gpu_id in range(num_gpus):
        manager = IntegratedGPUMemoryManager(
            device_id=gpu_id,
            enable_monitoring=enable_monitoring,
            enable_aggressive_cleanup=enable_aggressive_cleanup,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )
        managers[gpu_id] = manager
    
    bt.logging.info(f"✓ Created {len(managers)} GPU memory managers")
    
    return managers


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the GPU memory management system.
    """
    
    # Example 1: Simple context manager
    print("\n=== Example 1: Simple Context Manager ===")
    with gpu_memory_context(device_id=0):
        # Your GPU operations here
        tensor = torch.randn(1000, 1000, device='cuda:0')
        result = tensor @ tensor.T
    # Memory automatically cleaned up
    
    # Example 2: Integrated manager
    print("\n=== Example 2: Integrated Manager ===")
    manager = IntegratedGPUMemoryManager(
        device_id=0,
        enable_monitoring=True,
        enable_aggressive_cleanup=True
    )
    
    with manager.managed_context():
        tensor = torch.randn(1000, 1000, device='cuda:0')
        result = tensor @ tensor.T
    
    manager.print_full_report()
    
    # Example 3: Multiple GPUs
    print("\n=== Example 3: Multiple GPUs ===")
    managers = create_gpu_memory_managers(
        num_gpus=2,
        enable_monitoring=True,
        enable_aggressive_cleanup=True
    )
    
    for gpu_id, manager in managers.items():
        print(f"\nGPU {gpu_id} stats:")
        stats = manager.get_comprehensive_stats()
        print(f"  Monitoring: {stats['monitoring_enabled']}")
        print(f"  Cleanup: {stats['aggressive_cleanup_enabled']}")
