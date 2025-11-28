"""
Optimized elite selection and component weighting system.

Solutions:
1. Vectorized weight calculation (pandas-based, 10-100x faster)
2. Adaptive elite fraction (dynamic based on diversity and performance)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import bittensor as bt


# ============================================================================
# SOLUTION 1: Vectorized Weight Calculation
# ============================================================================

class VectorizedComponentWeightCalculator:
    """
    Vectorized component weight calculation using pandas.
    
    Benefits:
    - 10-100x faster than loop-based approach
    - Memory efficient with pandas operations
    - True vectorization with groupby
    """
    
    def __init__(self):
        self.calculation_times = []
        self.total_calculations = 0
        
        bt.logging.info("Initialized VectorizedComponentWeightCalculator")
    
    def calculate_weights_vectorized(
        self,
        top_pool: pd.DataFrame,
        rxn_id: int,
        smoothing: float = 0.1
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate component weights using vectorized operations.
        
        Args:
            top_pool: DataFrame with top molecules
            rxn_id: Reaction ID
            smoothing: Smoothing factor to add to all weights
        
        Returns:
            Dictionary of weights per role: {'A': {id: weight}, 'B': {...}, 'C': {...}}
        """
        import time
        start_time = time.time()
        
        if top_pool.empty:
            return {'A': {}, 'B': {}, 'C': {}}
        
        # Determine score column
        score_column = 'boltz_score' if 'boltz_score' in top_pool.columns else 'combined_score'
        
        # Filter for correct reaction
        rxn_mask = top_pool['product_name'].str.startswith(f'rxn:{rxn_id}:')
        filtered_pool = top_pool[rxn_mask].copy()
        
        if filtered_pool.empty:
            return {'A': {}, 'B': {}, 'C': {}}
        
        # Parse components (vectorized with str.split)
        split_names = filtered_pool['product_name'].str.split(':', expand=True)
        
        # Extract component IDs
        try:
            filtered_pool['A_id'] = split_names[2].astype(int)
            filtered_pool['B_id'] = split_names[3].astype(int)
            
            # C component (if exists)
            if split_names.shape[1] > 4:
                filtered_pool['C_id'] = pd.to_numeric(split_names[4], errors='coerce').fillna(-1).astype(int)
            else:
                filtered_pool['C_id'] = -1
        except (ValueError, KeyError) as e:
            bt.logging.warning(f"Error parsing component IDs: {e}")
            return {'A': {}, 'B': {}, 'C': {}}
        
        # Clip negative scores
        filtered_pool['score_clipped'] = filtered_pool[score_column].clip(lower=0)
        
        # FULLY VECTORIZED: Use groupby + mean (single operation per role)
        weights = {}
        
        # Role A
        A_weights = (
            filtered_pool.groupby('A_id')['score_clipped']
            .mean()
            .add(smoothing)
            .to_dict()
        )
        weights['A'] = A_weights
        
        # Role B
        B_weights = (
            filtered_pool.groupby('B_id')['score_clipped']
            .mean()
            .add(smoothing)
            .to_dict()
        )
        weights['B'] = B_weights
        
        # Role C (if applicable)
        C_mask = filtered_pool['C_id'] >= 0
        if C_mask.any():
            C_weights = (
                filtered_pool[C_mask]
                .groupby('C_id')['score_clipped']
                .mean()
                .add(smoothing)
                .to_dict()
            )
            weights['C'] = C_weights
        else:
            weights['C'] = {}
        
        # Track performance
        elapsed = time.time() - start_time
        self.calculation_times.append(elapsed)
        self.total_calculations += 1
        
        bt.logging.debug(
            f"Vectorized weight calculation: {len(filtered_pool)} molecules, "
            f"{len(weights['A'])} A, {len(weights['B'])} B, {len(weights['C'])} C components "
            f"in {elapsed*1000:.2f}ms"
        )
        
        return weights
    
    def calculate_weights_batch(
        self,
        top_pool: pd.DataFrame,
        rxn_ids: List[int],
        smoothing: float = 0.1
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """
        Calculate weights for multiple reactions in batch.
        
        Args:
            top_pool: DataFrame with top molecules
            rxn_ids: List of reaction IDs
            smoothing: Smoothing factor
        
        Returns:
            Dictionary: {rxn_id: {'A': {id: weight}, 'B': {...}, 'C': {...}}}
        """
        weights_dict = {}
        for rxn_id in rxn_ids:
            weights_dict[rxn_id] = self.calculate_weights_vectorized(
                top_pool, rxn_id, smoothing
            )
        return weights_dict
    
    def get_stats(self) -> Dict[str, float]:
        """Get calculation statistics."""
        if not self.calculation_times:
            return {'avg_time_ms': 0, 'total_calculations': 0}
        
        return {
            'avg_time_ms': np.mean(self.calculation_times) * 1000,
            'min_time_ms': np.min(self.calculation_times) * 1000,
            'max_time_ms': np.max(self.calculation_times) * 1000,
            'total_calculations': self.total_calculations
        }


# ============================================================================
# SOLUTION 2: Adaptive Elite Fraction
# ============================================================================

class AdaptiveEliteSelector:
    """
    Adaptive elite selection with dynamic fraction based on diversity and performance.
    
    Strategy:
    - High diversity + good performance → Keep more elites
    - Low diversity + stagnation → Keep fewer elites (encourage exploration)
    - Adapts based on score improvement rate
    """
    
    # Class constants
    IMPROVEMENT_THRESHOLD_HIGH = 0.01
    IMPROVEMENT_THRESHOLD_LOW = -0.01
    DIVERSITY_MULTIPLIER = 1.5
    
    def __init__(
        self,
        min_elite_frac: float = 0.05,
        max_elite_frac: float = 0.40,
        target_diversity: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        self.min_elite_frac = min_elite_frac
        self.max_elite_frac = max_elite_frac
        self.target_diversity = target_diversity
        self.adaptation_rate = adaptation_rate
        
        # Track history
        self.current_elite_frac = (min_elite_frac + max_elite_frac) / 2
        self.score_history = []
        self.diversity_history = []
        self.elite_frac_history = []
        
        bt.logging.info(
            f"Initialized AdaptiveEliteSelector: "
            f"elite_frac=[{min_elite_frac:.2f}, {max_elite_frac:.2f}], "
            f"target_diversity={target_diversity:.2f}"
        )
    
    def calculate_diversity(self, top_pool: pd.DataFrame) -> float:
        """
        Calculate diversity score based on component usage.
        
        Diversity = average of (unique per role / molecules using that role)
        Higher diversity = more exploration
        """
        if top_pool.empty:
            return 0.0
        
        # Extract unique components per role
        unique_A = set()
        unique_B = set()
        unique_C = set()
        
        count_A = 0
        count_B = 0
        count_C = 0
        
        for name in top_pool['product_name']:
            parts = name.split(':')
            if len(parts) >= 4:
                try:
                    unique_A.add(int(parts[2]))
                    count_A += 1
                    
                    unique_B.add(int(parts[3]))
                    count_B += 1
                    
                    if len(parts) > 4:
                        unique_C.add(int(parts[4]))
                        count_C += 1
                except (ValueError, IndexError):
                    continue
        
        # Calculate diversity per role
        diversities = []
        
        if count_A > 0:
            diversities.append(len(unique_A) / count_A)
        if count_B > 0:
            diversities.append(len(unique_B) / count_B)
        if count_C > 0:
            diversities.append(len(unique_C) / count_C)
        
        # Average diversity across roles
        if diversities:
            diversity = np.mean(diversities)
        else:
            diversity = 0.0
        
        return float(diversity)
    
    def calculate_improvement_rate(self, window: int = 10) -> float:
        """
        Calculate score improvement rate over recent history.
        
        Returns:
            Improvement rate (positive = improving, negative = stagnating)
        """
        if len(self.score_history) < 2:
            return 0.0
        
        recent_scores = self.score_history[-window:]
        
        if len(recent_scores) < 2:
            return 0.0
        
        # Calculate linear regression slope
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # Handle constant scores
        if np.std(y) < 1e-10:
            return 0.0
        
        # Simple slope calculation
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by score range (not average)
        y = np.array(y, dtype=np.float64)
    
        # Replace NaN with -inf (worst score)
        y = np.nan_to_num(y, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        
        # Filter out infinite values for statistics
        valid_scores = y[np.isfinite(y)]
        
        if len(valid_scores) == 0:
            # No valid scores - return zeros
            bt.logging.warning("No valid scores to normalize, returning zeros")
            return np.zeros_like(y)
        
        if len(valid_scores) == 1:
            # Only one valid score - return 1.0 for valid, 0.0 for invalid
            normalized = np.where(np.isfinite(y), 1.0, 0.0)
            return normalized
        
        # Calculate range from valid scores only
        score_min = np.min(valid_scores)
        score_max = np.max(valid_scores)
        score_range = score_max - score_min
        if score_range > 1e-10:
            normalized_slope = slope / score_range
        else:
            # If range is tiny, use absolute slope with cap
            normalized_slope = np.clip(slope, -1.0, 1.0)
        
        return float(normalized_slope)
    
    def adapt_elite_fraction(
        self,
        top_pool: pd.DataFrame,
        current_best_score: float
    ) -> float:
        """
        Adapt elite fraction based on diversity and performance.
        
        Args:
            top_pool: Current top pool
            current_best_score: Current best score
        
        Returns:
            Adapted elite fraction
        """
        # Calculate metrics
        diversity = self.calculate_diversity(top_pool)
        self.score_history.append(current_best_score)
        self.diversity_history.append(diversity)
        
        improvement_rate = self.calculate_improvement_rate(window=10)
        
        # Adaptation logic
        if improvement_rate > self.IMPROVEMENT_THRESHOLD_HIGH:
            # Good improvement → Keep current strategy (slight increase)
            adjustment = self.adaptation_rate * 0.5
        elif improvement_rate < self.IMPROVEMENT_THRESHOLD_LOW:
            # Stagnating → Encourage exploration (decrease elites)
            adjustment = -self.adaptation_rate
        else:
            # Neutral → Adjust based on diversity
            if diversity < self.target_diversity:
                # Low diversity → Decrease elites (encourage exploration)
                adjustment = -self.adaptation_rate
            elif diversity > self.target_diversity * self.DIVERSITY_MULTIPLIER:
                # High diversity → Increase elites (exploit good solutions)
                adjustment = self.adaptation_rate
            else:
                # Good diversity → Maintain
                adjustment = 0
        
        # Update elite fraction
        self.current_elite_frac = np.clip(
            self.current_elite_frac + adjustment,
            self.min_elite_frac,
            self.max_elite_frac
        )
        
        self.elite_frac_history.append(self.current_elite_frac)
        
        bt.logging.debug(
            f"Adaptive elite: diversity={diversity:.3f}, "
            f"improvement={improvement_rate:.4f}, "
            f"elite_frac={self.current_elite_frac:.3f}"
        )
        
        return self.current_elite_frac
    
    def select_diverse_elites_vectorized(
        self,
        top_pool: pd.DataFrame,
        n_elites: int,
        min_score_ratio: float = 0.7
    ) -> pd.DataFrame:
        """
        Select diverse elite molecules using vectorized operations.
        
        Args:
            top_pool: Pool of molecules
            n_elites: Number of elites to select
            min_score_ratio: Minimum score ratio relative to top
        
        Returns:
            DataFrame of selected elites
        """
        if top_pool.empty or n_elites <= 0:
            return pd.DataFrame()
        
        # Take top candidates
        top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
        
        if len(top_candidates) <= n_elites:
            return top_candidates
        
        # Filter by score threshold
        score_column = 'boltz_score' if 'boltz_score' in top_candidates.columns else 'combined_score'
        max_score = top_candidates[score_column].max()
        threshold = max_score * min_score_ratio
        candidates = top_candidates[top_candidates[score_column] >= threshold].copy()
        
        if candidates.empty or len(candidates) <= n_elites:
            return top_candidates.head(n_elites)
        
        # Parse components (vectorized)
        split_names = candidates['product_name'].str.split(':', expand=True)
        
        try:
            candidates['A_id'] = split_names[2].astype(int)
            candidates['B_id'] = split_names[3].astype(int)
            if split_names.shape[1] > 4:
                candidates['C_id'] = pd.to_numeric(split_names[4], errors='coerce').fillna(-1).astype(int)
            else:
                candidates['C_id'] = -1
        except (ValueError, KeyError):
            return top_candidates.head(n_elites)
        
        # VECTORIZED DIVERSITY SCORING
        # Calculate uniqueness score for each molecule
        candidates['uniqueness'] = 0.0
        
        # Count frequency of each component
        A_counts = candidates['A_id'].value_counts()
        B_counts = candidates['B_id'].value_counts()
        C_counts = candidates[candidates['C_id'] >= 0]['C_id'].value_counts()
        
        # Assign uniqueness score (inverse frequency)
        candidates['uniqueness'] += 1.0 / candidates['A_id'].map(A_counts)
        candidates['uniqueness'] += 1.0 / candidates['B_id'].map(B_counts)
        
        C_mask = candidates['C_id'] >= 0
        if C_mask.any():
            candidates.loc[C_mask, 'uniqueness'] += 1.0 / candidates.loc[C_mask, 'C_id'].map(C_counts).fillna(1)
        
        # Combine score and uniqueness
        # Normalize both to [0, 1]
        score_range = candidates[score_column].max() - candidates[score_column].min()
        if score_range > 1e-10:
            score_norm = (candidates[score_column] - candidates[score_column].min()) / score_range
        else:
            score_norm = pd.Series(1.0, index=candidates.index)
        
        uniqueness_range = candidates['uniqueness'].max() - candidates['uniqueness'].min()
        if uniqueness_range > 1e-10:
            uniqueness_norm = (candidates['uniqueness'] - candidates['uniqueness'].min()) / uniqueness_range
        else:
            uniqueness_norm = pd.Series(1.0, index=candidates.index)
        
        # Combined score: 70% performance, 30% diversity
        candidates['combined'] = 0.7 * score_norm + 0.3 * uniqueness_norm
        
        # Select top n_elites by combined score
        selected = candidates.nlargest(n_elites, 'combined')
        
        # Drop temporary columns
        selected = selected.drop(columns=['A_id', 'B_id', 'C_id', 'uniqueness', 'combined'], errors='ignore')
        
        return selected
    
    def get_stats(self) -> Dict[str, any]:
        """Get selection statistics."""
        if not self.score_history:
            return {
                'current_elite_frac': self.current_elite_frac,
                'avg_diversity': 0,
                'improvement_rate': 0,
                'num_adaptations': 0
            }
        
        return {
            'current_elite_frac': self.current_elite_frac,
            'avg_diversity': np.mean(self.diversity_history) if self.diversity_history else 0,
            'current_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'improvement_rate': self.calculate_improvement_rate(),
            'num_adaptations': len(self.elite_frac_history),
            'best_score': max(self.score_history) if self.score_history else float('-inf'),
            'score_trend': 'improving' if self.calculate_improvement_rate() > 0 else 'stagnating'
        }
    
    def plot_history(self, save_path: str = None):
        """
        Plot adaptation history (optional, requires matplotlib).
        
        Args:
            save_path: Path to save plot (if None, display)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Plot 1: Score history
            axes[0].plot(self.score_history, 'b-', linewidth=2)
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Best Score')
            axes[0].set_title('Score Evolution')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Diversity history
            axes[1].plot(self.diversity_history, 'g-', linewidth=2)
            axes[1].axhline(y=self.target_diversity, color='r', linestyle='--', label='Target')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Diversity')
            axes[1].set_title('Diversity Evolution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Elite fraction history
            axes[2].plot(self.elite_frac_history, 'orange', linewidth=2)
            axes[2].axhline(y=self.min_elite_frac, color='r', linestyle='--', alpha=0.5, label='Min')
            axes[2].axhline(y=self.max_elite_frac, color='r', linestyle='--', alpha=0.5, label='Max')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Elite Fraction')
            axes[2].set_title('Adaptive Elite Fraction')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                bt.logging.info(f"Saved adaptation history plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        
        except ImportError:
            bt.logging.warning("matplotlib not available, skipping plot")


# ============================================================================
# INTEGRATED ELITE OPTIMIZER
# ============================================================================

class IntegratedEliteOptimizer:
    """
    Integrated elite optimization combining:
    1. Vectorized weight calculation
    2. Adaptive elite selection
    """
    
    def __init__(
        self,
        min_elite_frac: float = 0.05,
        max_elite_frac: float = 0.40,
        target_diversity: float = 0.5,
        smoothing: float = 0.1
    ):
        self.weight_calculator = VectorizedComponentWeightCalculator()
        self.elite_selector = AdaptiveEliteSelector(
            min_elite_frac=min_elite_frac,
            max_elite_frac=max_elite_frac,
            target_diversity=target_diversity
        )
        self.smoothing = smoothing
        
        bt.logging.info(
            f"Initialized IntegratedEliteOptimizer: "
            f"elite_frac=[{min_elite_frac:.2f}, {max_elite_frac:.2f}], "
            f"target_diversity={target_diversity:.2f}"
        )
    
    def process_top_pool(
        self,
        top_pool: pd.DataFrame,
        rxn_ids: List[int],
        current_best_score: float,
        n_elites: int = 50
    ) -> Tuple[Dict[int, Dict[str, Dict[int, float]]], pd.DataFrame, float]:
        """
        Process top pool: calculate weights and select elites.
        
        Args:
            top_pool: Top pool DataFrame
            rxn_ids: List of reaction IDs
            current_best_score: Current best score
            n_elites: Target number of elites
        
        Returns:
            Tuple of (component_weights, elite_df, elite_fraction)
        """
        # Calculate component weights (vectorized)
        component_weights = self.weight_calculator.calculate_weights_batch(
            top_pool, rxn_ids, self.smoothing
        )
        
        # Adapt elite fraction
        elite_frac = self.elite_selector.adapt_elite_fraction(
            top_pool, current_best_score
        )
        
        # Calculate actual number of elites
        actual_n_elites = max(1, int(n_elites * elite_frac / 0.25))  # Normalize to base of 0.25
        
        # Select diverse elites (vectorized)
        elite_df = self.elite_selector.select_diverse_elites_vectorized(
            top_pool, actual_n_elites
        )
        
        return component_weights, elite_df, elite_frac
    
    def get_comprehensive_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        return {
            'weight_calculator': self.weight_calculator.get_stats(),
            'elite_selector': self.elite_selector.get_stats()
        }
    
    def print_report(self):
        """Print comprehensive report."""
        stats = self.get_comprehensive_stats()
        
        bt.logging.info("="*60)
        bt.logging.info("ELITE OPTIMIZATION REPORT")
        bt.logging.info("="*60)
        
        # Weight calculator stats
        wc_stats = stats['weight_calculator']
        bt.logging.info(f"Weight Calculation:")
        bt.logging.info(f"  Total calculations: {wc_stats['total_calculations']}")
        bt.logging.info(f"  Avg time: {wc_stats['avg_time_ms']:.2f}ms")
        
        # Elite selector stats
        es_stats = stats['elite_selector']
        bt.logging.info(f"")
        bt.logging.info(f"Elite Selection:")
        bt.logging.info(f"  Current elite fraction: {es_stats['current_elite_frac']:.3f}")
        bt.logging.info(f"  Current diversity: {es_stats['current_diversity']:.3f}")
        bt.logging.info(f"  Improvement rate: {es_stats['improvement_rate']:.4f}")
        bt.logging.info(f"  Score trend: {es_stats['score_trend']}")
        bt.logging.info(f"  Best score: {es_stats['best_score']:.4f}")
        bt.logging.info(f"  Num adaptations: {es_stats['num_adaptations']}")
        
        bt.logging.info("="*60)
