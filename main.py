import os
from utils.interactive_search import InteractiveSearch
from utils.dataset_handler import DatasetHandler
from search.search_engine import MoleculeSearchEngine

"""
USAGE:
- To update the dataset from a CSV file, call update_dataset_from_csv('dataset.csv')
- To load and store dataset in Elasticsearch, call load_and_store_dataset_to_elastic()
- To test similarity search with your dataset, call test_with_your_dataset()
- To run interactive search in the terminal, call run_interactive_search()

For API usage, use api_server.py instead.
"""

def update_dataset_from_csv(csv_file: str = "dataset.csv"):
    """Update the molecular database with new molecules from your CSV file"""
    print(f"Updating dataset from {csv_file}...")
    search_engine = MoleculeSearchEngine(use_elasticsearch=True)
    try:
        search_engine.load_model('molecular_gnn_model.pth')
        print("Loaded existing model")
    except Exception:
        print("No existing model found, will train new model")
    # Load new molecules from CSV (automatically checks for duplicates)
    dataset_handler = DatasetHandler(search_engine)
    try:
        df = dataset_handler.load_dataset(csv_file)
        dataset_handler.update_dataset(df)
        search_engine.save_model('molecular_gnn_model.pth')
        print(f"Processed and saved {len(df)} molecules.")
    except Exception as e:
        print(f"Error updating dataset: {e}")
    print("Dataset update completed!")
    return search_engine

def load_and_store_dataset_to_elastic(csv_file: str = "dataset/test_dataset.csv"):
    """Load dataset from CSV and store in Elasticsearch"""
    print(f"Loading and storing dataset from {csv_file} to Elasticsearch...")
    search_engine = MoleculeSearchEngine(use_elasticsearch=True)
    dataset_handler = DatasetHandler(search_engine)
    try:
        dataset_handler.load_and_store_dataset(csv_file)
        print("Dataset successfully loaded and stored in Elasticsearch!")
    except Exception as e:
        print(f"Error loading and storing dataset: {e}")
    return search_engine

def test_with_your_dataset():
    """Test the system with molecules from your specific dataset"""
    print("Testing with your dataset molecules...")
    search_engine = MoleculeSearchEngine(use_elasticsearch=True)
    try:
        search_engine.load_model('molecular_gnn_model.pth')
        print("Loaded existing model")
    except Exception:
        print("Please run update_dataset_from_csv() first to train the model")
        return
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
            for i, (result_name, score) in enumerate(results, 1):
                print(f"  {i}. {result_name} (similarity: {score:.4f})")
                if score > 0.99:
                    print("     *** EXACT MATCH FOUND! ***")
        else:
            print("  No results found")
        print("-" * 60)

def run_interactive_search():
    """Run interactive search in the terminal."""
    try:
        interactive_search = InteractiveSearch()
        interactive_search.run()
    except Exception as e:
        print(f"Error running interactive search: {e}")

if __name__ == "__main__":
    # Example usage: Uncomment the function you want to run
    # update_dataset_from_csv('dataset.csv')
    load_and_store_dataset_to_elastic('dataset/test_dataset.csv')
    # test_with_your_dataset()
    # run_interactive_search()
    print("This script provides utility functions for dataset update, testing, and interactive search.\nFor API usage, run api_server.py.")