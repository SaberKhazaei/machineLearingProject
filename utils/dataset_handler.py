import pandas as pd
from typing import List, Dict
from search.search_engine import MolecularSearchEngine
import ast

class DatasetHandler:
    """Class for handling molecular dataset operations"""
    
    def __init__(self, search_engine: MolecularSearchEngine):
        """Initialize dataset handler"""
        self.search_engine = search_engine
    
    def load_dataset(self, csv_file: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def update_dataset(self, df: pd.DataFrame) -> None:
        """Update search engine with molecules from dataset"""
        for _, row in df.iterrows():
            name = str(row['name'])
            smiles = str(row['smiles'])
            self.search_engine.add_molecule(name, smiles)
    
    def load_and_store_dataset(self, csv_file: str = 'dataset/test_dataset.csv') -> None:
        """Load dataset from CSV, parse, and store in Elasticsearch"""
        try:
            df = pd.read_csv(csv_file)
            # Normalize column names
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            for _, row in df.iterrows():
                name = str(row['compound_name'])
                smiles_repr = row['smiles_representation']
                # Parse SMILES list from string if needed
                if isinstance(smiles_repr, str):
                    try:
                        smiles_list = ast.literal_eval(smiles_repr)
                    except Exception:
                        smiles_list = [smiles_repr]
                else:
                    smiles_list = [smiles_repr]
                for smiles in smiles_list:
                    self.search_engine.add_molecule(name, smiles)
            print(f"Loaded and stored {len(df)} compounds from {csv_file} into Elasticsearch.")
        except Exception as e:
            print(f"Error loading and storing dataset: {e}")
            raise
    
    def test_similarity(self, query_smiles: str, top_k: int = 5) -> List[Dict]:
        """Test similarity search with a query molecule"""
        results = self.search_engine.search_similar_molecules(query_smiles, top_k)
        return [{
            'name': name,
            'similarity_score': score
        } for name, score in results]
