"""
Comprehensive timing and profiling system for the miner.

Tracks:
- Iteration time (total)
- Molecule generation time
- Model inference time (per GPU)
- Pool update time
- Cache operations
- Maintenance tasks
"""

import time
import statistics
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from contextlib import contextmanager
import bittensor as bt


class TimingTracker:
    """
    Track timing for all miner operations.
    
    Features:
    - Per-operation timing
    - Rolling statistics (last N operations)
    - Percentile calculations
    - Performance reports
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # Timing histories (use deque for efficient rolling window)
        self.iteration_times = deque(maxlen=history_size)
        self.generation_times = deque(maxlen=history_size)
        self.inference_times = deque(maxlen=history_size)
        self.pool_update_times = deque(maxlen=history_size)
        self.cache_times = deque(maxlen=history_size)
        self.maintenance_times = deque(maxlen=history_size)
        
        # Per-GPU inference times
        self.gpu_inference_times = defaultdict(lambda: deque(maxlen=history_size))
        
        # Molecule counts
        self.molecules_generated = deque(maxlen=history_size)
        self.molecules_scored = deque(maxlen=history_size)
        
        # Current operation start times (for nested timing)
        self._current_starts = {}
        
        # Total counters
        self.total_iterations = 0
        self.total_molecules_generated = 0
        self.total_molecules_scored = 0
        
        bt.logging.info("TimingTracker initialized")
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Usage:
            with timing_tracker.time_operation('generation'):
                generate_molecules()
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._record_time(operation_name, elapsed)
    
    def _record_time(self, operation_name: str, elapsed: float):
        """Record timing for an operation."""
        if operation_name == 'iteration':
            self.iteration_times.append(elapsed)
            self.total_iterations += 1
        elif operation_name == 'generation':
            self.generation_times.append(elapsed)
        elif operation_name == 'inference':
            self.inference_times.append(elapsed)
        elif operation_name == 'pool_update':
            self.pool_update_times.append(elapsed)
        elif operation_name == 'cache':
            self.cache_times.append(elapsed)
        elif operation_name == 'maintenance':
            self.maintenance_times.append(elapsed)
        elif operation_name.startswith('gpu_'):
            gpu_id = int(operation_name.split('_')[1])
            self.gpu_inference_times[gpu_id].append(elapsed)
    
    def record_gpu_inference(self, gpu_id: int, elapsed: float, num_molecules: int):
        """Record GPU inference timing."""
        self.gpu_inference_times[gpu_id].append(elapsed)
        self.molecules_scored.append(num_molecules)
        self.total_molecules_scored += num_molecules
    
    def record_generation(self, elapsed: float, num_molecules: int):
        """Record molecule generation timing."""
        self.generation_times.append(elapsed)
        self.molecules_generated.append(num_molecules)
        self.total_molecules_generated += num_molecules
    
    def get_stats(self, times: deque) -> Dict[str, float]:
        """Calculate statistics for a timing deque."""
        if not times:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'std': 0.0
            }
        
        times_list = list(times)
        sorted_times = sorted(times_list)
        n = len(sorted_times)
        
        return {
            'count': n,
            'mean': statistics.mean(times_list),
            'median': statistics.median(times_list),
            'min': min(times_list),
            'max': max(times_list),
            'p95': sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            'p99': sorted_times[int(n * 0.99)] if n > 0 else 0.0,
            'std': statistics.stdev(times_list) if n > 1 else 0.0
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all operations."""
        stats = {
            'iteration': self.get_stats(self.iteration_times),
            'generation': self.get_stats(self.generation_times),
            'inference': self.get_stats(self.inference_times),
            'pool_update': self.get_stats(self.pool_update_times),
            'cache': self.get_stats(self.cache_times),
            'maintenance': self.get_stats(self.maintenance_times),
        }
        
        # Per-GPU stats
        stats['gpu_inference'] = {}
        for gpu_id, times in self.gpu_inference_times.items():
            stats['gpu_inference'][gpu_id] = self.get_stats(times)
        
        # Throughput stats
        if self.molecules_generated:
            avg_generated = statistics.mean(self.molecules_generated)
        else:
            avg_generated = 0
        
        if self.molecules_scored:
            avg_scored = statistics.mean(self.molecules_scored)
        else:
            avg_scored = 0
        
        if self.generation_times:
            avg_gen_time = statistics.mean(self.generation_times)
            generation_throughput = avg_generated / avg_gen_time if avg_gen_time > 0 else 0
        else:
            generation_throughput = 0
        
        if self.inference_times:
            avg_inf_time = statistics.mean(self.inference_times)
            inference_throughput = avg_scored / avg_inf_time if avg_inf_time > 0 else 0
        else:
            inference_throughput = 0
        
        stats['throughput'] = {
            'generation_mol_per_sec': generation_throughput,
            'inference_mol_per_sec': inference_throughput,
            'avg_molecules_generated': avg_generated,
            'avg_molecules_scored': avg_scored,
        }
        
        # Totals
        stats['totals'] = {
            'iterations': self.total_iterations,
            'molecules_generated': self.total_molecules_generated,
            'molecules_scored': self.total_molecules_scored,
        }
        
        return stats
    
    def print_iteration_summary(self, iteration: int):
        """Print summary for the current iteration."""
        if not self.iteration_times:
            return
        
        last_iter_time = self.iteration_times[-1]
        last_gen_time = self.generation_times[-1] if self.generation_times else 0
        last_inf_time = self.inference_times[-1] if self.inference_times else 0
        last_pool_time = self.pool_update_times[-1] if self.pool_update_times else 0
        
        last_gen_count = self.molecules_generated[-1] if self.molecules_generated else 0
        last_scored_count = self.molecules_scored[-1] if self.molecules_scored else 0
        
        bt.logging.info(f"⏱️  Iteration {iteration} Timing:")
        bt.logging.info(f"  Total:      {last_iter_time:.2f}s")
        
        # FIX: Safe division with zero checks
        if last_gen_time > 0:
            if last_gen_count > 0:
                gen_throughput = last_gen_count / last_gen_time
                bt.logging.info(f"  Generation: {last_gen_time:.2f}s ({last_gen_count} molecules, {gen_throughput:.1f} mol/s)")
            else:
                bt.logging.info(f"  Generation: {last_gen_time:.2f}s (0 molecules)")
        else:
            if last_gen_count > 0:
                bt.logging.info(f"  Generation: <0.01s ({last_gen_count} molecules)")
            else:
                bt.logging.info(f"  Generation: skipped")
        
        if last_inf_time > 0:
            if last_scored_count > 0:
                inf_throughput = last_scored_count / last_inf_time
                bt.logging.info(f"  Inference:  {last_inf_time:.2f}s ({last_scored_count} molecules, {inf_throughput:.1f} mol/s)")
            else:
                bt.logging.info(f"  Inference:  {last_inf_time:.2f}s (0 molecules)")
        else:
            if last_scored_count > 0:
                bt.logging.info(f"  Inference:  <0.01s ({last_scored_count} molecules)")
            else:
                bt.logging.info(f"  Inference:  skipped")
        
        if last_pool_time > 0:
            bt.logging.info(f"  Pool Update: {last_pool_time:.3f}s")
        
        # Per-GPU breakdown
        if self.gpu_inference_times:
            has_gpu_data = any(len(times) > 0 for times in self.gpu_inference_times.values())
            if has_gpu_data:
                bt.logging.info(f"  GPU Breakdown:")
                for gpu_id in sorted(self.gpu_inference_times.keys()):
                    if self.gpu_inference_times[gpu_id]:
                        gpu_time = self.gpu_inference_times[gpu_id][-1]
                        bt.logging.info(f"    GPU {gpu_id}: {gpu_time:.2f}s")

    def print_comprehensive_report(self):
        """Print comprehensive timing report."""
        stats = self.get_comprehensive_stats()
        
        bt.logging.info("="*70)
        bt.logging.info("COMPREHENSIVE TIMING REPORT")
        bt.logging.info("="*70)
        
        # Iteration stats
        iter_stats = stats['iteration']
        bt.logging.info(f"Iteration Timing (n={iter_stats['count']}):")
        bt.logging.info(f"  Mean:   {iter_stats['mean']:.2f}s")
        bt.logging.info(f"  Median: {iter_stats['median']:.2f}s")
        bt.logging.info(f"  Min:    {iter_stats['min']:.2f}s")
        bt.logging.info(f"  Max:    {iter_stats['max']:.2f}s")
        bt.logging.info(f"  P95:    {iter_stats['p95']:.2f}s")
        bt.logging.info(f"  P99:    {iter_stats['p99']:.2f}s")
        bt.logging.info(f"  Std:    {iter_stats['std']:.2f}s")
        
        # Generation stats
        gen_stats = stats['generation']
        bt.logging.info(f"")
        bt.logging.info(f"Molecule Generation (n={gen_stats['count']}):")
        bt.logging.info(f"  Mean:   {gen_stats['mean']:.2f}s")
        bt.logging.info(f"  Median: {gen_stats['median']:.2f}s")
        bt.logging.info(f"  P95:    {gen_stats['p95']:.2f}s")
        
        # Inference stats
        inf_stats = stats['inference']
        bt.logging.info(f"")
        bt.logging.info(f"Model Inference (n={inf_stats['count']}):")
        bt.logging.info(f"  Mean:   {inf_stats['mean']:.2f}s")
        bt.logging.info(f"  Median: {inf_stats['median']:.2f}s")
        bt.logging.info(f"  P95:    {inf_stats['p95']:.2f}s")
        
        # Per-GPU stats
        if stats['gpu_inference']:
            bt.logging.info(f"")
            bt.logging.info(f"Per-GPU Inference:")
            for gpu_id in sorted(stats['gpu_inference'].keys()):
                gpu_stats = stats['gpu_inference'][gpu_id]
                bt.logging.info(f"  GPU {gpu_id} (n={gpu_stats['count']}):")
                bt.logging.info(f"    Mean:   {gpu_stats['mean']:.2f}s")
                bt.logging.info(f"    Median: {gpu_stats['median']:.2f}s")
                bt.logging.info(f"    P95:    {gpu_stats['p95']:.2f}s")
        
        # Pool update stats
        pool_stats = stats['pool_update']
        bt.logging.info(f"")
        bt.logging.info(f"Pool Update (n={pool_stats['count']}):")
        bt.logging.info(f"  Mean:   {pool_stats['mean']*1000:.1f}ms")
        bt.logging.info(f"  Median: {pool_stats['median']*1000:.1f}ms")
        bt.logging.info(f"  P95:    {pool_stats['p95']*1000:.1f}ms")
        
        # Throughput
        throughput = stats['throughput']
        bt.logging.info(f"")
        bt.logging.info(f"Throughput:")
        bt.logging.info(f"  Generation: {throughput['generation_mol_per_sec']:.1f} molecules/second")
        bt.logging.info(f"  Inference:  {throughput['inference_mol_per_sec']:.1f} molecules/second")
        bt.logging.info(f"  Avg Generated: {throughput['avg_molecules_generated']:.0f} molecules/iteration")
        bt.logging.info(f"  Avg Scored:    {throughput['avg_molecules_scored']:.0f} molecules/iteration")
        
        # Totals
        totals = stats['totals']
        bt.logging.info(f"")
        bt.logging.info(f"Totals:")
        bt.logging.info(f"  Iterations:          {totals['iterations']:,}")
        bt.logging.info(f"  Molecules Generated: {totals['molecules_generated']:,}")
        bt.logging.info(f"  Molecules Scored:    {totals['molecules_scored']:,}")
        
        # Time breakdown
        if iter_stats['count'] > 0:
            total_time = sum(self.iteration_times)
            gen_time = sum(self.generation_times) if self.generation_times else 0
            inf_time = sum(self.inference_times) if self.inference_times else 0
            pool_time = sum(self.pool_update_times) if self.pool_update_times else 0
            other_time = total_time - gen_time - inf_time - pool_time
            
            bt.logging.info(f"")
            bt.logging.info(f"Time Breakdown:")
            bt.logging.info(f"  Generation:  {gen_time/total_time*100:.1f}% ({gen_time:.1f}s)")
            bt.logging.info(f"  Inference:   {inf_time/total_time*100:.1f}% ({inf_time:.1f}s)")
            bt.logging.info(f"  Pool Update: {pool_time/total_time*100:.1f}% ({pool_time:.1f}s)")
            bt.logging.info(f"  Other:       {other_time/total_time*100:.1f}% ({other_time:.1f}s)")
        
        bt.logging.info("="*70)
    
    def export_to_csv(self, filepath: str):
        """Export timing data to CSV for analysis."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'iteration',
                'iteration_time',
                'generation_time',
                'inference_time',
                'pool_update_time',
                'molecules_generated',
                'molecules_scored',
            ])
            
            # Data
            max_len = max(
                len(self.iteration_times),
                len(self.generation_times),
                len(self.inference_times),
                len(self.pool_update_times),
                len(self.molecules_generated),
                len(self.molecules_scored),
            )
            
            for i in range(max_len):
                row = [
                    i + 1,
                    self.iteration_times[i] if i < len(self.iteration_times) else '',
                    self.generation_times[i] if i < len(self.generation_times) else '',
                    self.inference_times[i] if i < len(self.inference_times) else '',
                    self.pool_update_times[i] if i < len(self.pool_update_times) else '',
                    self.molecules_generated[i] if i < len(self.molecules_generated) else '',
                    self.molecules_scored[i] if i < len(self.molecules_scored) else '',
                ]
                writer.writerow(row)
        
        bt.logging.info(f"Timing data exported to {filepath}")
