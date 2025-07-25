from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import torch
from typing import Optional

class SMILESToGraph:
    """Convert SMILES strings to PyTorch Geometric graph data"""
    
    @staticmethod
    def get_atom_features(atom):
        """Extract atom features for GNN"""
        features = []
        
        # Atomic number (one-hot encoded for common elements)
        atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
        for num in atomic_nums:
            features.append(1 if atom.GetAtomicNum() == num else 0)
        
        # Degree
        degree_options = [0, 1, 2, 3, 4, 5]
        degree = atom.GetDegree()
        for d in degree_options:
            features.append(1 if degree == d else 0)
        
        # Formal charge
        charge_options = [-2, -1, 0, 1, 2]
        charge = atom.GetFormalCharge()
        for c in charge_options:
            features.append(1 if charge == c else 0)
        
        # Hybridization
        hybrid_options = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
        hybrid = atom.GetHybridization()
        for h in hybrid_options:
            features.append(1 if hybrid == h else 0)
        
        # Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # Number of hydrogens
        h_options = [0, 1, 2, 3, 4]
        total_h = atom.GetTotalNumHs()
        for h in h_options:
            features.append(1 if total_h == h else 0)
        
        # In ring
        features.append(1 if atom.IsInRing() else 0)
        
        # Ring size (if in ring)
        ring_sizes = [3, 4, 5, 6, 7, 8]
        atom_ring_info = atom.GetOwningMol().GetRingInfo()
        atom_rings = atom_ring_info.AtomRings()
        in_ring_size = set()
        for ring in atom_rings:
            if atom.GetIdx() in ring:
                in_ring_size.add(len(ring))
        
        for size in ring_sizes:
            features.append(1 if size in in_ring_size else 0)
        
        return features
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(SMILESToGraph.get_atom_features(atom))
            
            # Convert to tensor
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Get edge indices
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([(i, j), (j, i)])  # Undirected graph
            
            if len(edge_indices) == 0:
                # Single atom molecule
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
