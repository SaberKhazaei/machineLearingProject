import os
from typing import List, Dict, Optional
from utils.interactive_search import InteractiveSearch
from utils.dataset_handler import DatasetHandler
from search.search_engine import MolecularSearchEngine

def main():
    """Main entry point for the application"""
    # Install required packages first:
    # pip install torch torch-geometric rdkit-py elasticsearch pandas scikit-learn networkx
    
    # Initialize and run interactive search
    try:
        interactive_search = InteractiveSearch()
        interactive_search.run()
    except Exception as e:
        print(f"Error running application: {e}")

def test_with_your_dataset():
    """Test the system with your specific dataset"""
    search_engine = MolecularSearchEngine()
    dataset_handler = DatasetHandler(search_engine)
    
    try:
        # Load dataset
        df = dataset_handler.load_dataset("dataset.csv")
        
        # Update search engine with dataset
        dataset_handler.update_dataset(df)
        
        # Test similarity search
        test_smiles = "CCO"  # Example SMILES string
        results = dataset_handler.test_similarity(test_smiles)
        
        print("\nTest results:")
        for result in results:
            print(f"Molecule: {result['name']}, Similarity: {result['similarity_score']:.4f}")
    except Exception as e:
        print(f"Error testing dataset: {e}")

if __name__ == "__main__":
    main()

# Generate embedding
embedding = self.model(graph.x, graph.edge_index, batch)
embedding_np = embedding.cpu().numpy().flatten()

mol_data['embedding'] = embedding_np
self.embeddings[mol_data['name']] = embedding_np

print(f"Generated embeddings for {len(molecules)} molecules")
    
def index_to_elasticsearch(self, molecules: List[Dict]):
    """Index molecular data and embeddings to Elasticsearch"""
    if not self.use_elasticsearch:
        return
    
    print("Indexing to Elasticsearch...")
    
    for mol_data in molecules:
        # THIS IS WHERE THE EMBEDDING IS INSERTED TO ELASTICSEARCH
        doc = {
            'name': mol_data['name'],
                'smiles': mol_data['smiles'],
                'embedding': mol_data['embedding'].tolist(),  # ← EMBEDDING VECTOR INSERTED HERE
                'molecular_weight': mol_data['molecular_weight'],
                'logp': mol_data['logp']
            }
            
            # Insert document with embedding into Elasticsearch
        self.es.index(index=self.index_name, body=doc)
        
        print(f"Indexed {len(molecules)} molecules to Elasticsearch")
    
    def search_similar_molecules(self, query_smiles: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar molecules given a SMILES string"""
        # Convert query SMILES to graph and generate embedding
        query_graph = SMILESToGraph.smiles_to_graph(query_smiles)
        if query_graph is None:
            return []
        
        self.model.eval()
        with torch.no_grad():
            query_graph = query_graph.to(self.device)
            batch = torch.zeros(query_graph.x.size(0), dtype=torch.long, device=self.device)
            query_embedding = self.model(query_graph.x, query_graph.edge_index, batch)
            query_embedding_np = query_embedding.cpu().numpy().flatten()
        
        if self.use_elasticsearch:
            return self._search_elasticsearch(query_embedding_np, top_k)
        else:
            return self._search_in_memory(query_embedding_np, top_k)
    
    def _search_elasticsearch(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, str, float]]:
        """Search using Elasticsearch cosine similarity"""
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
        
        response = self.es.search(
            index=self.index_name,
            body={"query": script_query, "size": top_k}
        )
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            similarity = (hit['_score'] - 1.0)  # Convert back from ES score
            results.append((source['name'], source['smiles'], similarity))
        
        return results
    
    def _search_in_memory(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, str, float]]:
        """Search using in-memory cosine similarity"""
        if not self.embeddings:
            return []
        
        similarities = []
        for name, embedding in self.embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            mol_data = self.molecules_db[name]
            similarities.append((name, mol_data['smiles'], similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embeddings': self.embeddings,
            'molecules_db': self.molecules_db
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embeddings = checkpoint.get('embeddings', {})
        self.molecules_db = checkpoint.get('molecules_db', {})
        print(f"Model loaded from {path}")

def main():
    """Example usage of the molecular search engine with your specific CSV format"""
    
    # Initialize search engine
    search_engine = MolecularSearchEngine(use_elasticsearch=True)
    
    # Load dataset from your CSV file (checks for duplicates in Elasticsearch)
    csv_file = "dataset.csv"  # Your CSV file name
    molecules = search_engine.load_dataset(csv_path=csv_file, check_duplicates=True)
    
    if not molecules:
        print("No new molecules to process. All molecules already exist in Elasticsearch or invalid data.")
        return
    
    # Train model (only if we have new molecules)
    search_engine.train_model(molecules, epochs=50)
    
    # Generate embeddings
    search_engine.generate_embeddings(molecules)
    
    # Index to Elasticsearch (only new molecules will be added)
    if search_engine.use_elasticsearch:
        search_engine.index_to_elasticsearch(molecules)
    
    # Save model
    search_engine.save_model('molecular_gnn_model.pth')
    
    # Example search with molecules from your dataset
    print("\n" + "="*50)
    print("MOLECULAR SIMILARITY SEARCH")
    print("="*50)
    
    # Test with SMILES from your dataset
    test_queries = [
        ("4-Methoxyphenol", "COC1=CC=C(C=C1)O"),
        ("Allopurinol", "C1=NNC2=C1C(=O)NC=N2"),
        ("Similar to Ascorbic acid", "C(C(C1C(=C(C(=O)O1)O)O)O)O")
    ]
    
    for query_name, query_smiles in test_queries:
        print(f"\nQuery: {query_name}")
        print(f"SMILES: {query_smiles}")
        
        results = search_engine.search_similar_molecules(query_smiles, top_k=5)
        
        print("Top similar molecules:")
        print("-" * 80)
        for i, (name, smiles, similarity) in enumerate(results, 1):
            print(f"{i}. {name}")
            print(f"   SMILES: {smiles}")
            print(f"   Similarity: {similarity:.4f}")
            if similarity > 0.99:
                print("   *** EXACT or VERY CLOSE MATCH ***")
        print("-" * 80)

# Function specifically for your dataset format
def update_dataset_from_csv(csv_file: str = "dataset.csv"):
    """Update the molecular database with new molecules from your CSV file"""
    
    print(f"Updating dataset from {csv_file}...")
    
    # Initialize search engine
    search_engine = MolecularSearchEngine(use_elasticsearch=True)
    
    # Load existing model if available
    try:
        search_engine.load_model('molecular_gnn_model.pth')
        print("Loaded existing model")
    except:
        print("No existing model found, will train new model")
    
    # Load new molecules from CSV (automatically checks for duplicates)
    new_molecules = search_engine.load_dataset(csv_path=csv_file, check_duplicates=True)
    
    if not new_molecules:
        print("No new molecules to add!")
        return
    
    # If we have new molecules, process them
    print(f"Processing {len(new_molecules)} new molecules...")
    
    # Generate embeddings for new molecules
    search_engine.generate_embeddings(new_molecules)
    
    # Index new molecules to Elasticsearch
    if search_engine.use_elasticsearch:
        search_engine.index_to_elasticsearch(new_molecules)
    
    # Save updated model
    search_engine.save_model('molecular_gnn_model.pth')
    
    print("Dataset update completed!")
    return search_engine

# Test function specifically for your dataset
def test_with_your_dataset():
    """Test the system with molecules from your specific dataset"""
    
    print("Testing with your dataset molecules...")
    
    # Initialize search engine
    search_engine = MolecularSearchEngine(use_elasticsearch=True)
    
    # Try to load existing model
    try:
        search_engine.load_model('molecular_gnn_model.pth')
        print("Loaded existing model")
    except:
        print("Please run main() first to train the model")
        return
    
    # Test molecules from your dataset
    test_molecules = {
        "4-Methoxyphenol": "COC1=CC=C(C=C1)O",
        "Allopurinol": "C1=NNC2=C1C(=O)NC=N2", 
        "Aminolevulinic Acid": "C(CC(=O)O)C(=O)CN",
        "Anastrozole": "CC(C)(C#N)C1=CC(=CC(=C1)CN2C=NC=N2)C(C)(C)C#N",
        "Bendamustine": "CN1C2=C(C=C(C=C2)N(CCCl)CCCl)N=C1CCCC(=O)O"
    }
    
    print("\n" + "="*60)
    print("TESTING WITH YOUR DATASET MOLECULES")
    print("="*60)
    
    for name, smiles in test_molecules.items():
        print(f"\nTesting: {name}")
        print(f"SMILES: {smiles}")
        
        results = search_engine.search_similar_molecules(smiles, top_k=3)
        
        if results:
            print("Results:")
            for i, (result_name, result_smiles, similarity) in enumerate(results, 1):
                print(f"  {i}. {result_name} (similarity: {similarity:.4f})")
                if similarity > 0.99:
                    print("     *** EXACT MATCH FOUND! ***")
        else:
            print("  No results found")
        print("-" * 60)

if __name__ == "__main__":
    # Install required packages first:
    # pip install torch torch-geometric rdkit-py elasticsearch pandas scikit-learn networkx
    main()

# Interactive search function for user queries
def interactive_search():
    """Interactive function for user to search molecules"""
    
    # Load pre-trained model
    search_engine = MolecularSearchEngine()
    
    try:
        search_engine.load_model('molecular_gnn_model.pth')
        print("Model loaded successfully!")
    except:
        print("No pre-trained model found. Please train the model first.")
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE MOLECULAR SIMILARITY SEARCH")
    print("="*60)
    print("Enter SMILES strings to find similar molecules")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    while True:
        query = input("\nEnter SMILES string: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        print(f"\nSearching for molecules similar to: {query}")
        results = search_engine.search_similar_molecules(query, top_k=5)
        
        if not results:
            print("No similar molecules found or invalid SMILES string.")
            continue
        
        print("\nResults:")
        print("-" * 80)
        for i, (name, smiles, similarity) in enumerate(results, 1):
            print(f"{i}. {name}")
            print(f"   SMILES: {smiles}")
            print(f"   Similarity: {similarity:.4f}")
            if similarity > 0.99:
                print("   *** EXACT or VERY CLOSE MATCH ***")
            print("-" * 80)

# Uncomment to run interactive search
# interactive_search()