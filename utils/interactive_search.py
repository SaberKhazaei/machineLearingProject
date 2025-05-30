from typing import Optional
from search.search_engine import MolecularSearchEngine

class InteractiveSearch:
    """Class for handling interactive molecular search"""
    
    def __init__(self):
        """Initialize the interactive search system"""
        self.search_engine = MolecularSearchEngine()
        self._load_model()
        
    def _load_model(self) -> None:
        """Load pre-trained model"""
        try:
            self.search_engine.load_model('molecular_gnn_model.pth')
            print("Model loaded successfully!")
        except:
            print("No pre-trained model found. Please train the model first.")
            raise RuntimeError("No model found")
    
    def display_header(self) -> None:
        """Display the interactive search header"""
        print("\n" + "="*60)
        print("INTERACTIVE MOLECULAR SIMILARITY SEARCH")
        print("="*60)
        print("Enter SMILES strings to find similar molecules")
        print("Type 'quit' to exit")
        print("-" * 60)
    
    def search(self, query_smiles: str, top_k: int = 5) -> Optional[list]:
        """Perform a search with the given SMILES string"""
        try:
            results = self.search_engine.search_similar_molecules(query_smiles, top_k)
            return results
        except Exception as e:
            print(f"Error searching molecules: {e}")
            return None
    
    def run(self) -> None:
        """Run the interactive search loop"""
        self.display_header()
        
        while True:
            query = input("\nEnter SMILES string: ").strip()
            
            if query.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            results = self.search(query)
            
            if results:
                print("\nSimilar molecules found:")
                for name, score in results:
                    print(f"- {name}: Similarity score = {score:.4f}")
            else:
                print("No similar molecules found.")
