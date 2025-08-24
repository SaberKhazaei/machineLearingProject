# file: data/graph_converter.py

import torch
import networkx as nx
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType


class GraphConverter:

    def __init__(self, feature_dim: int = 38):
        # ابعاد ثابت برای ویژگی‌های نودها (اتم‌ها)
        self.feature_dim = feature_dim

    def atom_features(self, atom: Chem.Atom) -> torch.Tensor:
        
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetTotalDegree()
        formal_charge = atom.GetFormalCharge()
        aromatic = 1 if atom.GetIsAromatic() else 0

        feats = [
            atomic_num,
            degree,
            formal_charge,
            aromatic,
        ]

        tensor = torch.tensor(feats, dtype=torch.float)

    
        if tensor.shape[0] < self.feature_dim:
            pad_len = self.feature_dim - tensor.shape[0]
            tensor = torch.cat([tensor, torch.zeros(pad_len)])
        elif tensor.shape[0] > self.feature_dim:
            tensor = tensor[: self.feature_dim]

        return tensor

    def mol_to_graph(self, smiles: str) -> Data:
    
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # نودها
        x = torch.stack([self.atom_features(atom) for atom in mol.GetAtoms()])

        # یال‌ها
        edges = []
        edge_attrs = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # encode bond type
            if bond_type == BondType.SINGLE:
                b = [1, 0, 0, 0]
            elif bond_type == BondType.DOUBLE:
                b = [0, 1, 0, 0]
            elif bond_type == BondType.TRIPLE:
                b = [0, 0, 1, 0]
            elif bond_type == BondType.AROMATIC:
                b = [0, 0, 0, 1]
            else:
                b = [0, 0, 0, 0]

            edges.append([start, end])
            edges.append([end, start])
            edge_attrs.append(b)
            edge_attrs.append(b)

        if not edges:
            # self-loop در صورت نبود پیوند
            edges = [[i, i] for i in range(len(x))]
            edge_attrs = [[1, 0, 0, 0]] * len(x)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def compute_descriptors(self, smiles: str) -> dict:
        """
        محاسبه توصیف‌گرهای استاندارد مولکولی با RDKit
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        desc = {
            "MolWt": Descriptors.MolWt(mol),
            "ExactMolWt": Descriptors.ExactMolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        }
        return desc

    def smiles_to_graph_nx(self, smiles: str) -> nx.Graph:
        """
        تبدیل SMILES به NetworkX Graph (برای visualization یا پردازش اضافه).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        g = nx.Graph()
        for atom in mol.GetAtoms():
            g.add_node(atom.GetIdx(), symbol=atom.GetSymbol())

        for bond in mol.GetBonds():
            g.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                type=str(bond.GetBondType()),
            )

        return g
