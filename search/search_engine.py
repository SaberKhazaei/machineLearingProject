from typing import List, Tuple, Dict, Optional
from search.elasticsearch.client import ElasticsearchClient
import torch
import os
from models.gnn_model import MolecularGNN
from data.graph_converter import SMILESToGraph

class MolecularSearchEngine:
    """Main class for molecular similarity search using GNN embeddings"""
    
    def __init__(self, model_path: str = None, use_elasticsearch: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MolecularGNN().to(self.device)
        self.molecules_db = {}  # Store molecule data
        self.embeddings = {}   # Store precomputed embeddings
        self.use_elasticsearch = use_elasticsearch
        
        if use_elasticsearch:
            try:
                self.es_client = ElasticsearchClient()
                self.index_name = 'molecular_embeddings'
                self._create_elasticsearch_index()
            except Exception as e:
                print(f"Elasticsearch connection failed: {e}")
                print("Falling back to in-memory search")
                self.use_elasticsearch = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_elasticsearch_index(self):
        """Create Elasticsearch index for molecular embeddings"""
        mapping = {
            "properties": {
                "name": {"type": "keyword"},
                "smiles": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 64  # Should match embedding_dim in model
                },
                "molecular_weight": {"type": "float"},
                "logp": {"type": "float"}
            }
        }
        self.es_client.create_index(self.index_name, mapping)
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def add_molecule(self, name: str, smiles: str):
        """Add a molecule to the database"""
        try:
            graph = SMILESToGraph.smiles_to_graph(smiles)
            if graph is None:
                print(f"Invalid SMILES string: {smiles}")
                return False
            
            # Generate embedding
            graph = graph.to(self.device)
            embedding = self.model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long).to(self.device))
            
            # Store molecule data
            self.molecules_db[name] = {
                'smiles': smiles,
                'molecular_weight': Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
                'logp': Descriptors.MolLogP(Chem.MolFromSmiles(smiles))
            }
            
            # Store embedding
            self.embeddings[name] = embedding.cpu().detach().numpy()
            
            # Store in Elasticsearch if enabled
            if self.use_elasticsearch:
                doc = {
                    'name': name,
                    'smiles': smiles,
                    'embedding': embedding.cpu().detach().numpy().tolist(),
                    'molecular_weight': self.molecules_db[name]['molecular_weight'],
                    'logp': self.molecules_db[name]['logp']
                }
                self.es_client.index_document(self.index_name, doc, name)
            
            return True
        except Exception as e:
            print(f"Error adding molecule {name}: {e}")
            return False
    
    def search_similar_molecules(self, query_smiles: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar molecules based on GNN embeddings"""
        try:
            query_graph = SMILESToGraph.smiles_to_graph(query_smiles)
            if query_graph is None:
                return []
            
            query_graph = query_graph.to(self.device)
            query_embedding = self.model(query_graph.x, query_graph.edge_index, 
                                      torch.zeros(query_graph.num_nodes, dtype=torch.long).to(self.device))
            
            if self.use_elasticsearch:
                # Use Elasticsearch for similarity search
                query = {
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding.cpu().detach().numpy().tolist()}
                            }
                        }
                    }
                }
                
                results = self.es_client.search(self.index_name, query, top_k)
                return [(hit['_id'], hit['_score']) for hit in results]
            
            # In-memory search using cosine similarity
            similarities = []
            for name, embedding in self.embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_embedding.cpu().detach().numpy()),
                    torch.tensor(embedding)
                ).item()
                similarities.append((name, similarity))
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
