"""
Optimized molecular generation using pre-loaded pools and caching.

Performance optimizations:
1. Increased max_workers with adaptive scaling
2. Batch RDKit molecule validations with vectorization
3. Multiprocessing for CPU-bound validation tasks
"""

import os
import random
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import bittensor as bt

# FIXED IMPORTS
from miner.cache_manager import MoleculePoolCache, PersistentSMILESCache
from miner.integrated_cache import IntegratedCacheManager
from miner.descriptor_cache import MolecularDescriptorCache, get_descriptor_cache
from combinatorial_db.reactions import (
    perform_smarts_reaction,
    validate_and_order_reactants
)


# ============================================================================
# WORKER MANAGEMENT (Solution 1: Adaptive Workers)
# ============================================================================

def get_optimal_worker_count() -> int:
    """
    Determine optimal worker count based on system resources.
    """
    cpu_count = os.cpu_count() or 4
    
    # For mixed CPU/I/O workload: use 1.5-2x CPU count
    # Cap at 32 to avoid excessive overhead
    return min(cpu_count * 2, 32)


def adaptive_worker_count() -> int:
    """
    Adjust worker count based on current system load.
    Requires psutil (optional dependency).
    """
    try:
        import psutil
        cpu_count = os.cpu_count() or 4
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # If system is under heavy load, reduce workers
        if cpu_percent > 80 or memory_percent > 85:
            return max(2, cpu_count // 2)
        elif cpu_percent > 60 or memory_percent > 70:
            return cpu_count
        else:
            # System has capacity, use more workers
            return min(cpu_count * 2, 32)
    except ImportError:
        # psutil not available, use static optimal count
        return get_optimal_worker_count()


# ============================================================================
# MOLECULE GENERATION
# ============================================================================

def generate_molecules_from_pools_optimized(
    rxn_id: int,
    n: int,
    pool_cache: MoleculePoolCache,
    component_weights: dict = None
) -> List[str]:
    """
    Generate molecules using pre-loaded pools (no database queries).
    
    Args:
        rxn_id: Reaction ID (4 or 5)
        n: Number of molecules to generate
        pool_cache: Pre-loaded molecule pool cache
        component_weights: Optional weights for biased sampling
    
    Returns:
        List of molecule names in format "rxn:4:A_id:B_id" or "rxn:5:A_id:B_id:C_id"
    """
    roles = pool_cache.reaction_roles[rxn_id]
    
    # Get pre-loaded pools (instant lookup, no DB query)
    molecules_A = pool_cache.get_pool(roles['roleA'])
    molecules_B = pool_cache.get_pool(roles['roleB'])
    molecules_C = pool_cache.get_pool(roles['roleC']) if roles['is_three_component'] else []
    
    if not molecules_A or not molecules_B:
        bt.logging.warning(f"Empty pools for rxn {rxn_id}: A={len(molecules_A)}, B={len(molecules_B)}")
        return []
    
    if roles['is_three_component'] and not molecules_C:
        bt.logging.warning(f"Empty pool C for 3-component rxn {rxn_id}")
        return []
    
    # Extract IDs for sampling
    A_ids = [mol[0] for mol in molecules_A]  # mol[0] is mol_id
    B_ids = [mol[0] for mol in molecules_B]
    C_ids = [mol[0] for mol in molecules_C] if roles['is_three_component'] else None
    
    # Weighted sampling if component weights provided
    if component_weights:
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        
        # Normalize weights
        sum_A = sum(weights_A)
        sum_B = sum(weights_B)
        weights_A = [w / sum_A if sum_A > 0 else 1.0/len(weights_A) for w in weights_A]
        weights_B = [w / sum_B if sum_B > 0 else 1.0/len(weights_B) for w in weights_B]
        
        picks_A = random.choices(A_ids, weights=weights_A, k=n)
        picks_B = random.choices(B_ids, weights=weights_B, k=n)
        
        if roles['is_three_component']:
            weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids]
            sum_C = sum(weights_C)
            weights_C = [w / sum_C if sum_C > 0 else 1.0/len(weights_C) for w in weights_C]
            picks_C = random.choices(C_ids, weights=weights_C, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = random.choices(A_ids, k=n)
        picks_B = random.choices(B_ids, k=n)
        
        if roles['is_three_component']:
            picks_C = random.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    # Remove duplicates while preserving order (optimized)
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def get_smiles_from_reaction_cached(
    product_name: str,
    pool_cache: MoleculePoolCache,
    cache_manager: IntegratedCacheManager
) -> Optional[str]:
    """
    Get SMILES for a reaction product with caching.
    
    This eliminates:
    1. Database queries for molecule SMILES
    2. Repeated reaction computations for same reactants
    """
    # Check cache first - FIXED
    cached_smiles = cache_manager.smiles_cache.get(product_name)
    if cached_smiles:
        return cached_smiles
    
    try:
        parts = product_name.split(":")
        
        if len(parts) == 4:  # Two-component reaction
            _, rxn_id, mol1_id, mol2_id = parts
            rxn_id, mol1_id, mol2_id = int(rxn_id), int(mol1_id), int(mol2_id)
            
            roles = pool_cache.reaction_roles[rxn_id]
            
            # Get molecules from pre-loaded pools
            mol1 = pool_cache.find_molecule(mol1_id, [roles['roleA'], roles['roleB']])
            mol2 = pool_cache.find_molecule(mol2_id, [roles['roleA'], roles['roleB']])
            
            if not mol1 or not mol2:
                return None
            
            smiles1, role_mask1 = mol1[1], mol1[2]
            smiles2, role_mask2 = mol2[1], mol2[2]
            
            # Validate and order reactants
            reactant1, reactant2 = validate_and_order_reactants(
                smiles1, smiles2, role_mask1, role_mask2, roles['roleA'], roles['roleB']
            )
            
            if not reactant1 or not reactant2:
                return None
            
            # Perform reaction
            result_smiles = perform_smarts_reaction(reactant1, reactant2, roles['smarts'])
            
            # Cache result - FIXED
            if result_smiles:
                cache_manager.smiles_cache.set(product_name, result_smiles)
            
            return result_smiles
        
        elif len(parts) == 5:  # Three-component reaction
            _, rxn_id, mol1_id, mol2_id, mol3_id = parts
            rxn_id, mol1_id, mol2_id, mol3_id = int(rxn_id), int(mol1_id), int(mol2_id), int(mol3_id)
            
            roles = pool_cache.reaction_roles[rxn_id]
            
            # Get molecules from pre-loaded pools
            mol1 = pool_cache.find_molecule(mol1_id, [roles['roleA'], roles['roleB']])
            mol2 = pool_cache.find_molecule(mol2_id, [roles['roleA'], roles['roleB']])
            mol3 = pool_cache.find_molecule(mol3_id, [roles['roleC']])
            
            if not mol1 or not mol2 or not mol3:
                return None
            
            smiles1, role_mask1 = mol1[1], mol1[2]
            smiles2, role_mask2 = mol2[1], mol2[2]
            smiles3, role_mask3 = mol3[1], mol3[2]
            
            # Validate and order reactants
            validation_result = validate_and_order_reactants(
                smiles1, smiles2, role_mask1, role_mask2, roles['roleA'], roles['roleB'],
                smiles3, role_mask3, roles['roleC']
            )
            
            if not all(validation_result):
                return None
            
            reactant1, reactant2, reactant3 = validation_result
            
            # Perform cascade reaction (rxn:5 specific)
            if rxn_id == 5:
                suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
                suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
                
                intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
                if not intermediate:
                    return None
                
                result_smiles = perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)
                
                # Cache result - FIXED
                if result_smiles:
                    cache_manager.smiles_cache.set(product_name, result_smiles)
                
                return result_smiles
            
            return None
        
        else:
            return None
    
    except Exception as e:
        bt.logging.error(f"Error in cached reaction {product_name}: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return None


def generate_offspring_from_elites_optimized(
    rxn_id: int,
    n: int,
    elite_names: list[str],
    pool_cache: MoleculePoolCache,
    mutation_prob: float = 0.1,
    avoid_names: set[str] = None
) -> list[str]:
    """
    Generate offspring using pre-loaded pools (no DB queries).
    """
    roles = pool_cache.reaction_roles[rxn_id]
    
    # Extract elite components
    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    for name in elite_names:
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A = int(parts[2])
                B = int(parts[3])
                elite_As.add(A)
                elite_Bs.add(B)
                if len(parts) > 4:
                    C = int(parts[4])
                    elite_Cs.add(C)
            except (ValueError, IndexError):
                continue
    
    # Get pool IDs from pre-loaded cache
    pool_A = pool_cache.get_pool(roles['roleA'])
    pool_B = pool_cache.get_pool(roles['roleB'])
    pool_C = pool_cache.get_pool(roles['roleC']) if roles['is_three_component'] else []
    
    pool_A_ids = [mol[0] for mol in pool_A]
    pool_B_ids = [mol[0] for mol in pool_B]
    pool_C_ids = [mol[0] for mol in pool_C] if roles['is_three_component'] else []
    
    out = []
    local_names = set()
    
    for _ in range(n):
        use_mutA = (not elite_As) or (random.random() < mutation_prob)
        use_mutB = (not elite_Bs) or (random.random() < mutation_prob)
        use_mutC = (not elite_Cs) or (random.random() < mutation_prob)
        
        A = random.choice(pool_A_ids) if use_mutA else random.choice(list(elite_As))
        B = random.choice(pool_B_ids) if use_mutB else random.choice(list(elite_Bs))
        
        if roles['is_three_component']:
            C = random.choice(pool_C_ids) if use_mutC else random.choice(list(elite_Cs))
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        
        if avoid_names and name in avoid_names:
            continue
        if name in local_names:
            continue
        
        out.append(name)
        local_names.add(name)
        if avoid_names is not None:
            avoid_names.add(name)
    
    return out


# ============================================================================
# VALIDATION (Solution 2: Batch RDKit + Solution 3: Multiprocessing)
# ============================================================================

def ultra_light_prefilter(
    smiles: str,
    rot_min: int,
    rot_max: int,
    heavy_min: int,
    mw_max: float,
    tpsa_max: float
) -> Tuple[bool, Optional[str]]:
    """
    Ultra-light prefilter with early exit optimization and descriptor caching.
    
    Strategy:
    1. Check cache first (fastest)
    2. Order checks from cheapest to most expensive
    3. Exit immediately on first failure (early exit)
    4. Cache descriptors for future use
    
    Returns:
        (is_valid, rejection_reason)
    """
    descriptor_cache = get_descriptor_cache()
    
    try:
        # Step 1: Check descriptor cache first
        cached_descriptors = descriptor_cache.get(smiles)
        
        if cached_descriptors:
            # Use cached descriptors (ultra-fast path)
            heavy = cached_descriptors['heavy_atoms']
            if heavy < heavy_min:
                return False, "heavy_atoms_low"
            
            rot = cached_descriptors['rotatable_bonds']
            if not (rot_min <= rot <= rot_max):
                return False, "rotatable_bonds"
            
            mw = cached_descriptors['mol_weight']
            if mw > mw_max:
                return False, "mw_high"
            
            tpsa = cached_descriptors['tpsa']
            if tpsa > tpsa_max:
                return False, "tpsa_high"
            
            return True, None
        
        # Step 2: Compute descriptors if not cached
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, "invalid_smiles"
        
        # Step 3: Check properties in order of computational cost (cheapest first)
        # This allows early exit without computing expensive descriptors
        
        # Cheapest: Heavy atoms (just counting)
        heavy = mol.GetNumHeavyAtoms()
        if heavy < heavy_min:
            # Cache partial result for future reference
            descriptor_cache.set(smiles, {'heavy_atoms': heavy})
            return False, "heavy_atoms_low"
        
        # Medium cost: Rotatable bonds
        rot = Descriptors.NumRotatableBonds(mol)
        if not (rot_min <= rot <= rot_max):
            # Cache partial result
            descriptor_cache.set(smiles, {
                'heavy_atoms': heavy,
                'rotatable_bonds': rot
            })
            return False, "rotatable_bonds"
        
        # More expensive: Molecular weight
        mw = Descriptors.MolWt(mol)
        if mw > mw_max:
            descriptor_cache.set(smiles, {
                'heavy_atoms': heavy,
                'rotatable_bonds': rot,
                'mol_weight': mw
            })
            return False, "mw_high"
        
        # Most expensive: TPSA (surface area calculation)
        tpsa = Descriptors.TPSA(mol)
        if tpsa > tpsa_max:
            descriptor_cache.set(smiles, {
                'heavy_atoms': heavy,
                'rotatable_bonds': rot,
                'mol_weight': mw,
                'tpsa': tpsa
            })
            return False, "tpsa_high"
        
        # All checks passed - cache full descriptors
        full_descriptors = {
            'heavy_atoms': heavy,
            'rotatable_bonds': rot,
            'mol_weight': mw,
            'tpsa': tpsa,
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
        }
        descriptor_cache.set(smiles, full_descriptors)
        
        return True, None
    
    except Exception as e:
        bt.logging.debug(f"Error in prefilter: {e}")
        return False, "rdkit_error"


def validate_molecules_batch_vectorized(
    smiles_list: List[str],
    names_list: List[str],
    rot_min: int,
    rot_max: int,
    heavy_min: int,
    mw_max: float,
    tpsa_max: float,
    seen_keys: set
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """
    Solution 2: Vectorized batch validation of molecules using numpy.
    
    Returns:
        valid_names, valid_smiles, valid_inchikeys, rejection_stats
    """
    rejection_stats = {
        'invalid_smiles': 0,
        'heavy_atoms_low': 0,
        'rotatable_bonds': 0,
        'mw_high': 0,
        'tpsa_high': 0,
        'duplicate': 0
    }
    
    # Step 1: Batch convert SMILES to Mol objects
    mol_objects = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_objects.append(mol)
            valid_indices.append(idx)
        else:
            rejection_stats['invalid_smiles'] += 1
    
    if not mol_objects:
        return [], [], [], rejection_stats
    
    # Step 2: Batch compute all descriptors (vectorized with numpy)
    heavy_atoms = np.array([mol.GetNumHeavyAtoms() for mol in mol_objects])
    rot_bonds = np.array([Descriptors.NumRotatableBonds(mol) for mol in mol_objects])
    mol_weights = np.array([Descriptors.MolWt(mol) for mol in mol_objects])
    tpsa_values = np.array([Descriptors.TPSA(mol) for mol in mol_objects])
    
    # Step 3: Vectorized filtering (numpy boolean indexing)
    valid_mask = (
        (heavy_atoms >= heavy_min) &
        (rot_bonds >= rot_min) &
        (rot_bonds <= rot_max) &
        (mol_weights <= mw_max) &
        (tpsa_values <= tpsa_max)
    )
    
    # Count rejections
    rejection_stats['heavy_atoms_low'] = int(np.sum(heavy_atoms < heavy_min))
    rejection_stats['rotatable_bonds'] = int(np.sum((rot_bonds < rot_min) | (rot_bonds > rot_max)))
    rejection_stats['mw_high'] = int(np.sum(mol_weights > mw_max))
    rejection_stats['tpsa_high'] = int(np.sum(tpsa_values > tpsa_max))
    
    # Step 4: Generate InChIKeys for valid molecules only
    valid_names = []
    valid_smiles = []
    valid_inchikeys = []
    
    for idx, is_valid in enumerate(valid_mask):
        if is_valid:
            original_idx = valid_indices[idx]
            mol = mol_objects[idx]
            
            try:
                inchikey = Chem.MolToInchiKey(mol)
                if inchikey not in seen_keys:
                    valid_names.append(names_list[original_idx])
                    valid_smiles.append(smiles_list[original_idx])
                    valid_inchikeys.append(inchikey)
                else:
                    rejection_stats['duplicate'] += 1
            except:
                rejection_stats['invalid_smiles'] += 1
    
    return valid_names, valid_smiles, valid_inchikeys, rejection_stats


def validate_molecule_chunk_worker(
    chunk_data: Tuple[List[str], List[str]],
    rot_min: int,
    rot_max: int,
    heavy_min: int,
    mw_max: float,
    tpsa_max: float
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """
    Solution 3: Worker function for multiprocessing.
    Validates a chunk of molecules in a separate process.
    
    Args:
        chunk_data: (names_list, smiles_list)
        validation parameters
    
    Returns:
        (valid_molecules, rejection_stats)
        where valid_molecules = [(name, smiles, inchikey), ...]
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    names_list, smiles_list = chunk_data
    
    valid_molecules = []
    rejection_stats = {
        'invalid_smiles': 0,
        'heavy_atoms_low': 0,
        'rotatable_bonds': 0,
        'mw_high': 0,
        'tpsa_high': 0
    }
    
    for name, smiles in zip(names_list, smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                rejection_stats['invalid_smiles'] += 1
                continue
            
            # Fast checks first (fail early)
            heavy = mol.GetNumHeavyAtoms()
            if heavy < heavy_min:
                rejection_stats['heavy_atoms_low'] += 1
                continue
            
            rot = Descriptors.NumRotatableBonds(mol)
            if not (rot_min <= rot <= rot_max):
                rejection_stats['rotatable_bonds'] += 1
                continue
            
            # Slower checks
            mw = Descriptors.MolWt(mol)
            if mw > mw_max:
                rejection_stats['mw_high'] += 1
                continue
            
            tpsa = Descriptors.TPSA(mol)
            if tpsa > tpsa_max:
                rejection_stats['tpsa_high'] += 1
                continue
            
            # Generate InChIKey
            inchikey = Chem.MolToInchiKey(mol)
            valid_molecules.append((name, smiles, inchikey))
        
        except Exception:
            rejection_stats['invalid_smiles'] += 1
    
    return valid_molecules, rejection_stats


def chunk_list(data: List, n_chunks: int) -> List[List]:
    """Split list into n_chunks roughly equal parts."""
    if n_chunks <= 0:
        n_chunks = 1
    chunk_size = max(1, len(data) // n_chunks)
    if len(data) % n_chunks:
        chunk_size += 1
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def validate_molecules_multiprocessing(
    names_list: List[str],
    smiles_list: List[str],
    rot_min: int,
    rot_max: int,
    heavy_min: int,
    mw_max: float,
    tpsa_max: float,
    seen_keys: set,
    n_workers: int = None
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """
    Solution 3: Validate molecules using multiprocessing for true parallelism.
    
    This bypasses Python's GIL for CPU-bound RDKit operations.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    
    # For small batches, don't use multiprocessing (overhead not worth it)
    if len(names_list) < 100:
        return validate_molecules_batch_vectorized(
            smiles_list, names_list,
            rot_min, rot_max, heavy_min, mw_max, tpsa_max,
            seen_keys
        )
    
    # Split data into chunks
    name_chunks = chunk_list(names_list, n_workers)
    smiles_chunks = chunk_list(smiles_list, n_workers)
    chunks = list(zip(name_chunks, smiles_chunks))
    
    # Create worker function with fixed parameters
    worker = partial(
        validate_molecule_chunk_worker,
        rot_min=rot_min,
        rot_max=rot_max,
        heavy_min=heavy_min,
        mw_max=mw_max,
        tpsa_max=tpsa_max
    )
    
    # Process in parallel with error handling
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            # Add timeout to prevent hanging
            results = list(pool.map(worker, chunks, timeout=300))  # 5 min timeout
    
    except Exception as e:
        bt.logging.warning(f"Multiprocessing failed: {e}, falling back to vectorized")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return validate_molecules_batch_vectorized(
            smiles_list, names_list,
            rot_min, rot_max, heavy_min, mw_max, tpsa_max,
            seen_keys
        )
    
    # Merge results
    valid_names = []
    valid_smiles = []
    valid_inchikeys = []
    total_rejection_stats = {
        'invalid_smiles': 0,
        'heavy_atoms_low': 0,
        'rotatable_bonds': 0,
        'mw_high': 0,
        'tpsa_high': 0,
        'duplicate': 0
    }
    
    for valid_molecules, rejection_stats in results:
        # Merge rejection stats
        for key, count in rejection_stats.items():
            total_rejection_stats[key] += count
        
        # Filter duplicates
        for name, smiles, inchikey in valid_molecules:
            if inchikey not in seen_keys:
                valid_names.append(name)
                valid_smiles.append(smiles)
                valid_inchikeys.append(inchikey)
            else:
                total_rejection_stats['duplicate'] += 1
    
    return valid_names, valid_smiles, valid_inchikeys, total_rejection_stats


# Backward compatibility wrapper
def validate_molecules_batch(
    smiles_map: dict,  # {name: smiles}
    rot_min: int,
    rot_max: int,
    heavy_min: int,
    mw_max: float,
    tpsa_max: float,
    seen_keys: set
) -> List[Tuple[str, str, str]]:
    """
    Vectorized validation with descriptor caching and early exit.
    
    Returns: List of (name, smiles, inchikey) tuples for valid molecules
    """
    descriptor_cache = get_descriptor_cache()
    
    valid_results = []
    
    # Process each molecule with caching
    for name, smiles in smiles_map.items():
        try:
            # Check descriptor cache first
            cached_descriptors = descriptor_cache.get(smiles)
            
            if cached_descriptors:
                # Fast path: use cached descriptors
                if cached_descriptors['heavy_atoms'] < heavy_min:
                    continue
                if not (rot_min <= cached_descriptors['rotatable_bonds'] <= rot_max):
                    continue
                if cached_descriptors['mol_weight'] > mw_max:
                    continue
                if cached_descriptors['tpsa'] > tpsa_max:
                    continue
                
                # Valid - generate InChIKey
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchikey = Chem.MolToInchiKey(mol)
                    if inchikey not in seen_keys:
                        valid_results.append((name, smiles, inchikey))
                continue
            
            # Slow path: compute with early exit
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            
            # Early exit checks (ordered by cost)
            heavy = mol.GetNumHeavyAtoms()
            if heavy < heavy_min:
                descriptor_cache.set(smiles, {'heavy_atoms': heavy})
                continue
            
            rot = Descriptors.NumRotatableBonds(mol)
            if not (rot_min <= rot <= rot_max):
                descriptor_cache.set(smiles, {
                    'heavy_atoms': heavy,
                    'rotatable_bonds': rot
                })
                continue
            
            mw = Descriptors.MolWt(mol)
            if mw > mw_max:
                descriptor_cache.set(smiles, {
                    'heavy_atoms': heavy,
                    'rotatable_bonds': rot,
                    'mol_weight': mw
                })
                continue
            
            tpsa = Descriptors.TPSA(mol)
            if tpsa > tpsa_max:
                descriptor_cache.set(smiles, {
                    'heavy_atoms': heavy,
                    'rotatable_bonds': rot,
                    'mol_weight': mw,
                    'tpsa': tpsa
                })
                continue
            
            # All checks passed - cache full descriptors
            full_descriptors = {
                'heavy_atoms': heavy,
                'rotatable_bonds': rot,
                'mol_weight': mw,
                'tpsa': tpsa,
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
            }
            descriptor_cache.set(smiles, full_descriptors)
            
            # InChIKey check (most expensive)
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey in seen_keys:
                continue
            
            valid_results.append((name, smiles, inchikey))
        
        except Exception:
            continue
    
    return valid_results


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_valid_molecules_batch_optimized(
    rxn_ids: List[int],
    n_samples_per_reaction: int,
    pool_cache: MoleculePoolCache,
    cache_manager: IntegratedCacheManager,
    config: Any,
    batch_size: int = 400,
    elite_names: Dict[int, list[str]] = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    seen_inchikeys: set[str] = None,
    component_weights: Dict[int, dict] = None,
    use_multiprocessing: bool = True,
    n_workers: int = None
) -> dict:
    """
    Optimized version with all three performance solutions:
    1. Adaptive worker management
    2. Batch RDKit validations
    3. Multiprocessing for CPU-bound tasks
    """
    start_time = time.time()
    
    all_valid_molecules = []
    all_valid_smiles = []
    seen_keys = seen_inchikeys.copy() if seen_inchikeys else set()
    
    # Get prefilter thresholds
    rot_min = config.min_rotatable_bonds
    rot_max = config.max_rotatable_bonds
    heavy_min = config.min_heavy_atoms
    mw_max = getattr(config, 'prefilter_mw_max', 550.0)
    tpsa_max = getattr(config, 'prefilter_tpsa_max', 140.0)
    
    # Determine optimal worker count
    if n_workers is None:
        n_workers = adaptive_worker_count()
    
    # Statistics
    total_generated = 0
    total_validated = 0
    rejection_stats_total = defaultdict(int)
    
    for rxn_id in rxn_ids:
        roles = pool_cache.reaction_roles.get(rxn_id)
        if not roles:
            bt.logging.error(f"No role info for rxn_id {rxn_id}")
            continue
        
        valid_molecules = []
        valid_smiles = []
        rxn_elite_names = elite_names.get(rxn_id) if elite_names else None
        rxn_component_weights = component_weights.get(rxn_id) if component_weights else None
        
        iteration = 0
        while len(valid_molecules) < n_samples_per_reaction:
            iteration += 1
            needed = n_samples_per_reaction - len(valid_molecules)
            batch_size_actual = min(max(batch_size, 400), needed * 2)
            
            emitted_names = set()
            
            # Generate molecules with elite/random mix
            if rxn_elite_names:
                n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
                n_rand = batch_size_actual - n_elite
                
                elite_batch = generate_offspring_from_elites_optimized(
                    rxn_id=rxn_id,
                    n=n_elite,
                    elite_names=rxn_elite_names,
                    pool_cache=pool_cache,
                    mutation_prob=mutation_prob,
                    avoid_names=emitted_names,
                )
                emitted_names.update(elite_batch)
                
                rand_batch = generate_molecules_from_pools_optimized(
                    rxn_id, n_rand, pool_cache, rxn_component_weights
                )
                rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
                batch_molecules = elite_batch + rand_batch
            else:
                batch_molecules = generate_molecules_from_pools_optimized(
                    rxn_id, batch_size_actual, pool_cache, rxn_component_weights
                )
            
            total_generated += len(batch_molecules)
            
            # Batch SMILES generation with caching - FIXED
            batch_smiles_map = {}  # {name: smiles}
            for name in batch_molecules:
                smiles = get_smiles_from_reaction_cached(name, pool_cache, cache_manager)
                if smiles:
                    batch_smiles_map[name] = smiles
            
            if not batch_smiles_map:
                bt.logging.warning(f"No valid SMILES generated for rxn_id {rxn_id}, breaking")
                break
            
            # Batch validation with multiprocessing or vectorization
            names_list = list(batch_smiles_map.keys())
            smiles_list = list(batch_smiles_map.values())
            
            if use_multiprocessing and len(names_list) >= 100:
                # Use multiprocessing for large batches
                valid_names, valid_smiles_batch, valid_inchikeys, rejection_stats = \
                    validate_molecules_multiprocessing(
                        names_list, smiles_list,
                        rot_min, rot_max, heavy_min, mw_max, tpsa_max,
                        seen_keys, n_workers
                    )
            else:
                # Use vectorized numpy for small batches
                valid_names, valid_smiles_batch, valid_inchikeys, rejection_stats = \
                    validate_molecules_batch_vectorized(
                        smiles_list, names_list,
                        rot_min, rot_max, heavy_min, mw_max, tpsa_max,
                        seen_keys
                    )
            
            # Update statistics
            total_validated += len(valid_names)
            for key, count in rejection_stats.items():
                rejection_stats_total[key] += count
            
            # Add valid molecules
            for name, smiles, inchikey in zip(valid_names, valid_smiles_batch, valid_inchikeys):
                if len(valid_molecules) >= n_samples_per_reaction:
                    break
                valid_molecules.append(name)
                valid_smiles.append(smiles)
                seen_keys.add(inchikey)
            
            # Safety check to avoid infinite loop
            if iteration > 100:
                bt.logging.warning(f"Max iterations reached for rxn_id {rxn_id}")
                break
        
        all_valid_molecules.extend(valid_molecules)
        all_valid_smiles.extend(valid_smiles)
    
    elapsed_time = time.time() - start_time
    
    # Get cache statistics
    descriptor_cache = get_descriptor_cache()
    cache_stats = descriptor_cache.get_stats()
    
    # Log performance statistics
    success_rate = total_validated / total_generated if total_generated > 0 else 0
    bt.logging.info(
        f"Generation complete: {total_validated}/{total_generated} valid "
        f"({success_rate:.1%}) in {elapsed_time:.2f}s "
        f"({total_validated/elapsed_time:.1f} mol/s) "
        f"[workers: {n_workers}, cache hit rate: {cache_stats['hit_rate']:.1%}]"
    )
    
    if rejection_stats_total:
        bt.logging.debug(f"Rejection breakdown: {dict(rejection_stats_total)}")
    
    # Add cache stats to return dict
    return {
        'molecules': all_valid_molecules,
        'smiles': all_valid_smiles,
        'seen_inchikeys': seen_keys,
        'statistics': {
            'total_generated': total_generated,
            'total_validated': total_validated,
            'success_rate': success_rate,
            'elapsed_time': elapsed_time,
            'throughput': total_validated / elapsed_time if elapsed_time > 0 else 0,
            'n_workers': n_workers,
            'rejection_stats': dict(rejection_stats_total),
            'descriptor_cache': cache_stats
        }
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_diversity_penalty(
    candidate_smiles: str,
    existing_smiles: List[str],
    penalty_weight: float = 0.1
) -> float:
    """
    Compute diversity penalty based on Tanimoto similarity to existing molecules.
    Returns a penalty value (0 = very diverse, 1 = very similar).
    """
    if not existing_smiles:
        return 0.0
    
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(candidate_smiles)
        if not mol:
            return 0.0
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        
        max_similarity = 0.0
        for existing_smi in existing_smiles[-100:]:  # Compare with last 100 for efficiency
            existing_mol = Chem.MolFromSmiles(existing_smi)
            if existing_mol:
                existing_fp = AllChem.GetMorganFingerprintAsBitVect(existing_mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity * penalty_weight
    
    except Exception as e:
        bt.logging.warning(f"Error computing diversity penalty: {e}")
        return 0.0


def adaptive_batch_size(
    success_rate: float,
    current_batch_size: int,
    min_batch: int = 200,
    max_batch: int = 2000
) -> int:
    """
    Adaptively adjust batch size based on success rate.
    Low success rate → increase batch size
    High success rate → decrease batch size
    """
    if success_rate < 0.05:
        new_size = int(current_batch_size * 1.5)
    elif success_rate < 0.15:
        new_size = int(current_batch_size * 1.2)
    elif success_rate > 0.4:
        new_size = int(current_batch_size * 0.8)
    else:
        new_size = current_batch_size
    
    return max(min_batch, min(new_size, max_batch))


def compute_chemical_diversity(smiles_list: List[str]) -> float:
    """
    Compute Tanimoto diversity of molecule set.
    Returns value between 0 (identical) and 1 (very diverse).
    """
    if len(smiles_list) < 2:
        return 0.0
    
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        # Generate fingerprints
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
        
        if len(fps) < 2:
            return 0.0
        
        # Compute pairwise Tanimoto similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    except Exception as e:
        bt.logging.warning(f"Error computing diversity: {e}")
        return 0.0
