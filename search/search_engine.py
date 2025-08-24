# file: engine/search_engine.py

import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from rdkit import Chem
from rdkit.Chem import Descriptors
from data.graph_converter import GraphConverter


class MoleculeSearchEngine:
    

    def __init__(self, index_name: str = "molecules", es_host: str = "http://localhost:9200" , use_elasticsearch: bool = True):
        self.index_name = index_name
        self.es = Elasticsearch(es_host)
        self.converter = GraphConverter()

        if use_elasticsearch:
            self.es = Elasticsearch(es_host)
        else:
            self.es = None
        
            
    def create_index(self):
        
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "name": {"type": "text"},
                        "smiles": {"type": "keyword"},
                        "descriptors": {"type": "object", "enabled": True},
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"Created index '{self.index_name}'.")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def add_molecule(self, name: str, smiles: str):
     
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")

            descriptors = self.converter.compute_descriptors(smiles)

            doc = {
                "name": name,
                "smiles": smiles,
                "descriptors": descriptors,
            }
            self.es.index(index=self.index_name, body=doc)
        except Exception as e:
            print(f"Error adding molecule {name}: {e}")

    def load_dataset(self, dataset_path: str):
        """
        بارگذاری دیتاست CSV و اضافه کردن همه مولکول‌ها
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        required_cols = {"name", "smiles"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        actions = []
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    continue

                descriptors = self.converter.compute_descriptors(row["smiles"])

                doc = {
                    "_index": self.index_name,
                    "_source": {
                        "name": row["name"],
                        "smiles": row["smiles"],
                        "descriptors": descriptors,
                    },
                }
                actions.append(doc)
            except Exception as e:
                print(f"Error adding molecule {row['name']}: {e}")

        if actions:
            helpers.bulk(self.es, actions)
            print(f"Loaded and stored {len(actions)} compounds from {dataset_path} into Elasticsearch.")
        else:
            print("No valid molecules found in dataset.")

    def search_by_name(self, query: str, top_k: int = 5):
        """
        جستجو بر اساس نام مولکول
        """
        body = {
            "query": {
                "match": {"name": {"query": query, "fuzziness": "AUTO"}}
            }
        }
        res = self.es.search(index=self.index_name, body=body, size=top_k)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def search_by_descriptor(self, descriptor: str, value: float, tolerance: float = 0.1, top_k: int = 5):
        """
        جستجو بر اساس یک توصیفگر (مثلا MolWt)
        """
        body = {
            "query": {
                "range": {
                    f"descriptors.{descriptor}": {
                        "gte": value * (1 - tolerance),
                        "lte": value * (1 + tolerance),
                    }
                }
            }
        }
        res = self.es.search(index=self.index_name, body=body, size=top_k)
        return [hit["_source"] for hit in res["hits"]["hits"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Molecule Search Engine")
    parser.add_argument("--dataset", type=str, default="dataset/test_dataset.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    engine = MoleculeSearchEngine()
    engine.create_index()
    engine.load_dataset(args.dataset)

    print("Dataset successfully loaded and stored in Elasticsearch!")
    print("This script provides utility functions for dataset update, testing, and interactive search.")
    print("For API usage, run api_server.py.")
