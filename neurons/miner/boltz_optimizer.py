"""
Optimized Boltz-2 inference with advanced GPU optimizations.

Integrates with existing BoltzWrapper without breaking changes.
"""

import os
import time
from typing import List, Dict, Optional, Any
from collections import deque
from threading import Lock
import torch
import psutil
import bittensor as bt


# ============================================================================
# VRAM MONITORING (Thread-Safe)
# ============================================================================

class VRAMMonitor:
    """
    Monitor GPU VRAM usage and dynamically adjust batch sizes.
    Thread-safe implementation.
    """
    
    def __init__(self, device_id: int = 0, target_utilization: float = 0.85):
        self.device_id = device_id
        self.target_utilization = target_utilization
        self.device = torch.device(f'cuda:{device_id}')
        self.lock = Lock()  # Thread safety
        
        # Track batch size history
        self.batch_size_history = deque(maxlen=10)
        self.oom_count = 0
        
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
            bt.logging.info(f"GPU {device_id} total VRAM: {self.total_memory / 1e9:.2f} GB")
        else:
            self.total_memory = 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'utilization': 0}
        
        with self.lock:
            allocated = torch.cuda.memory_allocated(self.device_id)
            reserved = torch.cuda.memory_reserved(self.device_id)
            free = self.total_memory - reserved
            utilization = reserved / self.total_memory if self.total_memory > 0 else 0
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'utilization': utilization,
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9,
                'free_gb': free / 1e9
            }
    
    def get_optimal_batch_size(
        self, 
        current_batch_size: int,
        min_batch: int = 1,
        max_batch: int = 100
    ) -> int:
        """
        Calculate optimal batch size based on current VRAM usage.
        Thread-safe.
        """
        with self.lock:
            stats = self.get_memory_stats()
            utilization = stats['utilization']
            
            # If we've had recent OOM errors, be conservative
            if self.oom_count > 0:
                new_batch_size = max(min_batch, current_batch_size // 2)
                self.oom_count = max(0, self.oom_count - 1)
                bt.logging.warning(f"OOM detected, reducing batch size: {current_batch_size} → {new_batch_size}")
                return new_batch_size
            
            # Adjust based on utilization
            if utilization < self.target_utilization * 0.7:
                new_batch_size = min(max_batch, int(current_batch_size * 1.5))
            elif utilization < self.target_utilization:
                new_batch_size = min(max_batch, int(current_batch_size * 1.2))
            elif utilization > self.target_utilization * 1.1:
                new_batch_size = max(min_batch, int(current_batch_size * 0.8))
            else:
                new_batch_size = current_batch_size
            
            self.batch_size_history.append(new_batch_size)
            return new_batch_size
    
    def report_oom(self):
        """Report an out-of-memory error (thread-safe)."""
        with self.lock:
            self.oom_count += 1
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device_id)


# ============================================================================
# MIXED PRECISION WRAPPER
# ============================================================================

class MixedPrecisionConfig:
    """
    Configuration for mixed precision inference.
    
    Compatible with PyTorch Lightning's precision settings.
    """
    
    def __init__(self, device_id: int = 0, quantization: str = 'none'):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.quantization = quantization
        
        # Determine precision based on quantization setting
        if quantization == 'fp16':
            self.use_mixed_precision = True
            self.dtype = torch.float16
            self.precision_str = "16-mixed"  # For PyTorch Lightning
            bt.logging.info(f"GPU {device_id}: FP16 mixed precision enabled")
        elif quantization == 'bf16':
            # Check if BF16 is supported (Ampere+ GPUs)
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability(device_id)
                if compute_capability[0] >= 8:  # Ampere or newer
                    self.use_mixed_precision = True
                    self.dtype = torch.bfloat16
                    self.precision_str = "bf16-mixed"
                    bt.logging.info(f"GPU {device_id}: BF16 mixed precision enabled")
                else:
                    bt.logging.warning(f"GPU {device_id}: BF16 not supported, falling back to FP16")
                    self.use_mixed_precision = True
                    self.dtype = torch.float16
                    self.precision_str = "16-mixed"
            else:
                self.use_mixed_precision = False
                self.dtype = torch.float32
                self.precision_str = "32"
        else:
            self.use_mixed_precision = False
            self.dtype = torch.float32
            self.precision_str = "32"
            bt.logging.info(f"GPU {device_id}: Using FP32 (no mixed precision)")
    
    def get_lightning_precision(self) -> str:
        """Get precision string for PyTorch Lightning Trainer."""
        return self.precision_str
    
    def should_use_amp(self) -> bool:
        """Check if AMP should be used."""
        return self.use_mixed_precision and self.quantization == 'fp16'


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Monitor and log performance metrics.
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.metrics = {
            'total_molecules': 0,
            'total_time': 0.0,
            'batch_times': deque(maxlen=100),
            'throughput_history': deque(maxlen=100),
            'oom_count': 0,
        }
        self.lock = Lock()
    
    def record_batch(self, num_molecules: int, elapsed_time: float):
        """Record batch processing metrics."""
        with self.lock:
            self.metrics['total_molecules'] += num_molecules
            self.metrics['total_time'] += elapsed_time
            self.metrics['batch_times'].append(elapsed_time)
            
            throughput = num_molecules / elapsed_time if elapsed_time > 0 else 0
            self.metrics['throughput_history'].append(throughput)
    
    def record_oom(self):
        """Record OOM event."""
        with self.lock:
            self.metrics['oom_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            avg_batch_time = (
                sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
                if self.metrics['batch_times'] else 0
            )
            avg_throughput = (
                sum(self.metrics['throughput_history']) / len(self.metrics['throughput_history'])
                if self.metrics['throughput_history'] else 0
            )
            overall_throughput = (
                self.metrics['total_molecules'] / self.metrics['total_time']
                if self.metrics['total_time'] > 0 else 0
            )
            
            return {
                'device_id': self.device_id,
                'total_molecules': self.metrics['total_molecules'],
                'total_time': self.metrics['total_time'],
                'avg_batch_time': avg_batch_time,
                'avg_throughput': avg_throughput,
                'overall_throughput': overall_throughput,
                'oom_count': self.metrics['oom_count'],
            }
    
    def log_stats(self):
        """Log performance statistics."""
        stats = self.get_stats()
        bt.logging.info("="*60)
        bt.logging.info(f"PERFORMANCE STATS (GPU {self.device_id})")
        bt.logging.info("="*60)
        bt.logging.info(f"Total molecules: {stats['total_molecules']}")
        bt.logging.info(f"Total time: {stats['total_time']:.2f}s")
        bt.logging.info(f"Avg batch time: {stats['avg_batch_time']:.2f}s")
        bt.logging.info(f"Avg throughput: {stats['avg_throughput']:.2f} mol/s")
        bt.logging.info(f"Overall throughput: {stats['overall_throughput']:.2f} mol/s")
        bt.logging.info(f"OOM events: {stats['oom_count']}")
        bt.logging.info("="*60)


# ============================================================================
# ADAPTIVE WORKER MANAGER
# ============================================================================

class AdaptiveWorkerManager:
    """
    Dynamically adjust DataLoader workers based on system resources.
    """
    
    def __init__(
        self,
        base_workers: int = 4,
        max_workers: int = 4,  # Cap at 4 to avoid shared memory issues
        max_ram_usage: float = 0.85,
        device_id: int = 0
    ):
        self.base_workers = min(base_workers, max_workers)
        self.max_workers = max_workers
        self.max_ram_usage = max_ram_usage
        self.device_id = device_id
        self.lock = Lock()
        
        bt.logging.info(
            f"AdaptiveWorkerManager initialized: base={self.base_workers}, "
            f"max={self.max_workers}, max_ram={self.max_ram_usage:.1%}"
        )
    
    def get_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on current system state.
        Thread-safe.
        """
        with self.lock:
            try:
                # Check RAM usage
                memory = psutil.virtual_memory()
                ram_usage = memory.percent / 100.0
                available_memory_mb = memory.available / (1024 * 1024)
                
                # Estimate worker memory (each worker ~75MB)
                estimated_worker_memory_mb = self.base_workers * 75
                
                # If RAM usage is high, reduce workers
                if ram_usage > self.max_ram_usage:
                    reduction_factor = (1.0 - ram_usage) / (1.0 - self.max_ram_usage)
                    adjusted_workers = max(0, int(self.base_workers * reduction_factor))
                    bt.logging.warning(
                        f"RAM usage high ({ram_usage:.1%}), reducing workers: "
                        f"{self.base_workers} → {adjusted_workers}"
                    )
                    return adjusted_workers
                
                # If not enough available memory, reduce workers
                if available_memory_mb < estimated_worker_memory_mb * 1.5:
                    max_safe_workers = int(available_memory_mb / (75 * 1.5))
                    adjusted_workers = max(0, min(self.base_workers, max_safe_workers))
                    if adjusted_workers < self.base_workers:
                        bt.logging.warning(
                            f"Limited RAM ({available_memory_mb:.0f}MB), reducing workers: "
                            f"{self.base_workers} → {adjusted_workers}"
                        )
                    return adjusted_workers
                
                # All good, use base workers
                return self.base_workers
                
            except Exception as e:
                bt.logging.warning(f"Error calculating optimal workers: {e}, using base={self.base_workers}")
                return self.base_workers


# ============================================================================
# OPTIMIZATION MANAGER (Integrates All Optimizations)
# ============================================================================

class BoltzOptimizationManager:
    """
    Central manager for all Boltz optimizations.
    
    Integrates:
    - VRAM monitoring
    - Mixed precision
    - Performance tracking
    - Adaptive workers
    """
    
    def __init__(
        self,
        device_id: int = 0,
        quantization: str = 'none',
        base_workers: int = 4,
        max_workers: int = 4,
        target_vram_utilization: float = 0.85,
        max_ram_usage: float = 0.85,
        enable_monitoring: bool = True
    ):
        self.device_id = device_id
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.vram_monitor = VRAMMonitor(device_id, target_vram_utilization)
        self.mixed_precision = MixedPrecisionConfig(device_id, quantization)
        self.performance_monitor = PerformanceMonitor(device_id)
        self.worker_manager = AdaptiveWorkerManager(
            base_workers, max_workers, max_ram_usage, device_id
        )
        
        bt.logging.info("="*60)
        bt.logging.info(f"BoltzOptimizationManager initialized (GPU {device_id})")
        bt.logging.info(f"  Quantization: {quantization}")
        bt.logging.info(f"  Mixed precision: {self.mixed_precision.use_mixed_precision}")
        bt.logging.info(f"  Base workers: {base_workers} (max: {max_workers})")
        bt.logging.info(f"  VRAM monitoring: {enable_monitoring}")
        bt.logging.info("="*60)
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of DataLoader workers."""
        if not self.enable_monitoring:
            return self.worker_manager.base_workers
        return self.worker_manager.get_optimal_workers()
    
    def get_lightning_precision(self) -> str:
        """Get precision string for PyTorch Lightning."""
        return self.mixed_precision.get_lightning_precision()
    
    def record_batch_processing(self, num_molecules: int, elapsed_time: float):
        """Record batch processing metrics."""
        self.performance_monitor.record_batch(num_molecules, elapsed_time)
    
    def record_oom_event(self):
        """Record OOM event."""
        self.vram_monitor.report_oom()
        self.performance_monitor.record_oom()
    
    def clear_gpu_cache(self):
        """Clear GPU cache."""
        self.vram_monitor.clear_cache()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        return self.vram_monitor.get_memory_stats()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_monitor.get_stats()
    
    def log_performance_summary(self):
        """Log performance summary."""
        self.performance_monitor.log_stats()
        
        # Also log memory stats
        mem_stats = self.get_memory_stats()
        bt.logging.info(f"GPU {self.device_id} Memory:")
        bt.logging.info(f"  Allocated: {mem_stats['allocated_gb']:.2f} GB")
        bt.logging.info(f"  Reserved: {mem_stats['reserved_gb']:.2f} GB")
        bt.logging.info(f"  Free: {mem_stats['free_gb']:.2f} GB")
        bt.logging.info(f"  Utilization: {mem_stats['utilization']:.1%}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_optimization_managers(
    num_gpus: int = None,
    quantization: str = 'none',
    base_workers: int = 4,
    max_workers: int = 4,
    enable_monitoring: bool = True
) -> Dict[int, BoltzOptimizationManager]:
    """
    Create optimization managers for all GPUs.
    
    Args:
        num_gpus: Number of GPUs (None = all available)
        quantization: Quantization mode ('none', 'fp16', 'bf16', 'int8')
        base_workers: Base number of DataLoader workers
        max_workers: Maximum number of workers (capped at 4)
        enable_monitoring: Enable resource monitoring
    
    Returns:
        Dictionary mapping GPU ID to BoltzOptimizationManager
    """
    if torch.cuda.is_available():
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
        bt.logging.warning("No CUDA GPUs available, using CPU")
    
    managers = {}
    for gpu_id in range(num_gpus):
        manager = BoltzOptimizationManager(
            device_id=gpu_id,
            quantization=quantization,
            base_workers=base_workers,
            max_workers=max_workers,
            enable_monitoring=enable_monitoring
        )
        managers[gpu_id] = manager
        
        # Log GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(gpu_id)
            bt.logging.info(
                f"GPU {gpu_id}: {props.name} "
                f"({props.total_memory / 1e9:.1f} GB, "
                f"compute {props.major}.{props.minor})"
            )
    
    return managers
