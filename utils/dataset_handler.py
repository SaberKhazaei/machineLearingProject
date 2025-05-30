import pandas as pd
from typing import List, Dict
from search.search_engine import MolecularSearchEngine

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
    
    def test_similarity(self, query_smiles: str, top_k: int = 5) -> List[Dict]:
        """Test similarity search with a query molecule"""
        results = self.search_engine.search_similar_molecules(query_smiles, top_k)
        return [{
            'name': name,
            'similarity_score': score
        } for name, score in results]
