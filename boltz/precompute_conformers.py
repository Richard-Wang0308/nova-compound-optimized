"""
Precompute conformers for Boltz-2 molecules in parallel.
This avoids expensive RDKit conformer generation during inference.
"""
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from rdkit import Chem
from rdkit.Chem import AllChem
import bittensor as bt


def compute_conformer_fast(smiles: str, max_attempts: int = 5, max_iters: int = 500) -> Optional[Chem.Mol]:
    """
    Compute conformer for a single SMILES string with fast settings.
    
    Args:
        smiles: SMILES string
        max_attempts: Maximum embedding attempts (reduced for speed)
        max_iters: Maximum UFF optimization iterations (reduced for speed)
    
    Returns:
        RDKit Mol object with 3D conformer and atom names, or None if failed
    """
    try:
        # Parse SMILES
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = AllChem.AddHs(mol)
        
        # Set atom names (required by Boltz)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                # Skip molecules with atom names > 4 chars
                return None
            atom.SetProp("name", atom_name)
        
        # Generate conformer with fast settings
        options = AllChem.ETKDGv3()
        options.clearConfs = False
        options.maxAttempts = max_attempts  # Reduced from default
        
        conf_id = AllChem.EmbedMolecule(mol, options)
        
        if conf_id == -1:
            # Try with random coords as fallback
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)
            if conf_id == -1:
                return None
        
        # Fast UFF optimization (reduced iterations)
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
        except (RuntimeError, ValueError):
            # Force field issues - use conformer as-is
            pass
        
        # Store SMILES as property for reference
        mol.SetProp("_SMILES", smiles)
        
        return mol
    
    except Exception as e:
        bt.logging.debug(f"Failed to compute conformer for {smiles}: {e}")
        return None


def precompute_conformers_batch(
    items: List[Tuple[str, str]],  # List of (product_name, smiles)
    output_dir: Path,
    max_workers: int = 32,
    chunk_size: int = 100,
) -> Dict[str, int]:
    """
    Precompute conformers for a batch of molecules in parallel.
    
    Args:
        items: List of (product_name, smiles) tuples
        output_dir: Directory to save precomputed conformers
        max_workers: Number of parallel workers
        chunk_size: Chunk size for ProcessPoolExecutor
    
    Returns:
        Dict with stats: {"success": count, "failed": count}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"success": 0, "failed": 0}
    
    bt.logging.info(f"Precomputing conformers for {len(items)} molecules with {max_workers} workers...")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(compute_conformer_fast, smiles): (name, smiles)
            for name, smiles in items
        }
        
        # Process results as they complete
        for future in as_completed(future_to_item):
            name, smiles = future_to_item[future]
            try:
                mol = future.result()
                
                if mol is not None:
                    # Save as individual .pkl file (Boltz-2 format)
                    output_path = output_dir / f"{name}.pkl"
                    
                    # Ensure all properties are saved
                    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
                    
                    with open(output_path, 'wb') as f:
                        pickle.dump(mol, f)
                    
                    stats["success"] += 1
                    
                    if stats["success"] % 100 == 0:
                        bt.logging.info(f"Processed {stats['success']}/{len(items)} molecules...")
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                bt.logging.warning(f"Error processing {name}: {e}")
                stats["failed"] += 1
    
    bt.logging.info(
        f"Precomputation complete: {stats['success']} success, {stats['failed']} failed"
    )
    
    return stats


def load_precomputed_conformer(product_name: str, precomputed_dir: Path) -> Optional[Chem.Mol]:
    """
    Load a precomputed conformer from .pkl file.
    
    Args:
        product_name: Name of the product/molecule
        precomputed_dir: Directory containing precomputed conformers
    
    Returns:
        RDKit Mol object with conformer or None if not found
    """
    precomputed_dir = Path(precomputed_dir)
    conformer_path = precomputed_dir / f"{product_name}.pkl"
    
    if not conformer_path.exists():
        return None
    
    try:
        with open(conformer_path, 'rb') as f:
            mol = pickle.load(f)
        
        # Verify atom names are present (defensive check)
        for idx, atom in enumerate(mol.GetAtoms()):
            if not atom.HasProp("name"):
                atom_name = f"{atom.GetSymbol()}{idx + 1}"
                atom.SetProp("name", atom_name)
        
        return mol
    
    except Exception as e:
        bt.logging.debug(f"Error loading precomputed conformer for {product_name}: {e}")
        return None


# Utility function for batch checking
def check_precomputed_conformers(
    product_names: List[str],
    precomputed_dir: Path
) -> Tuple[List[str], List[str]]:
    """
    Check which conformers are already precomputed.
    
    Args:
        product_names: List of product names to check
        precomputed_dir: Directory containing precomputed conformers
    
    Returns:
        Tuple of (existing, missing) product names
    """
    precomputed_dir = Path(precomputed_dir)
    existing = []
    missing = []
    
    for name in product_names:
        conformer_path = precomputed_dir / f"{name}.pkl"
        if conformer_path.exists():
            existing.append(name)
        else:
            missing.append(name)
    
    return existing, missing


# Utility function for cleaning up
def cleanup_precomputed_conformers(
    precomputed_dir: Path,
    keep_names: Optional[List[str]] = None,
    dry_run: bool = True
) -> Dict[str, int]:
    """
    Clean up precomputed conformers directory.
    
    Args:
        precomputed_dir: Directory containing precomputed conformers
        keep_names: List of product names to keep (None = keep all)
        dry_run: If True, only report what would be deleted
    
    Returns:
        Dict with stats: {"kept": count, "deleted": count, "total": count}
    """
    precomputed_dir = Path(precomputed_dir)
    
    if not precomputed_dir.exists():
        return {"kept": 0, "deleted": 0, "total": 0}
    
    stats = {"kept": 0, "deleted": 0, "total": 0}
    keep_set = set(keep_names) if keep_names else None
    
    for pkl_file in precomputed_dir.glob("*.pkl"):
        stats["total"] += 1
        name = pkl_file.stem
        
        if keep_set is None or name in keep_set:
            stats["kept"] += 1
        else:
            stats["deleted"] += 1
            if not dry_run:
                pkl_file.unlink()
                bt.logging.debug(f"Deleted: {pkl_file.name}")
    
    if dry_run:
        bt.logging.info(
            f"Dry run: would keep {stats['kept']}/{stats['total']} files, "
            f"delete {stats['deleted']} files"
        )
    else:
        bt.logging.info(
            f"Cleanup complete: kept {stats['kept']}/{stats['total']} files, "
            f"deleted {stats['deleted']} files"
        )
    
    return stats


if __name__ == "__main__":
    """
    Example usage for testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute conformers for Boltz-2")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to test")
    parser.add_argument("--output-dir", type=str, default="./test_conformers", help="Output directory")
    args = parser.parse_args()
    
    if args.smiles:
        # Test single molecule
        print(f"Testing conformer generation for: {args.smiles}")
        mol = compute_conformer_fast(args.smiles)
        
        if mol is not None:
            print(f"✓ Success! Generated conformer with {mol.GetNumAtoms()} atoms")
            
            # Check atom names
            print("\nAtom names:")
            for atom in mol.GetAtoms():
                if atom.HasProp("name"):
                    print(f"  {atom.GetSymbol()} -> {atom.GetProp('name')}")
            
            # Save to file
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "test_molecule.pkl"
            
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
            with open(output_path, 'wb') as f:
                pickle.dump(mol, f)
            
            print(f"\n✓ Saved to: {output_path}")
            
            # Test loading
            loaded_mol = load_precomputed_conformer("test_molecule", output_dir)
            if loaded_mol:
                print(f"✓ Successfully loaded conformer")
            else:
                print(f"✗ Failed to load conformer")
        else:
            print(f"✗ Failed to generate conformer")
    else:
        print("Usage: python precompute_conformers.py --smiles 'CCO' --output-dir ./test")
