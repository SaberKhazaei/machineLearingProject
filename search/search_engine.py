# file: search/search_engine.py
import os
import ast
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from rdkit import Chem
from data.graph_converter import GraphConverter
from rdkit.Chem import AllChem
from rdkit import DataStructs


class MoleculeSearchEngine:
    def __init__(self, index_name: str = "molecules", es_host: str = "http://localhost:9200", use_elasticsearch: bool = True):
        self.index_name = index_name
        self.converter = GraphConverter()
        self.molecules_db = {}  # نگهداری موقت مولکول‌ها داخل حافظه

        if use_elasticsearch:
            self.es = Elasticsearch(es_host)
        else:
            self.es = None

    def create_index(self):
        if self.es is None:
            raise RuntimeError("Elasticsearch is not enabled.")

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

    def _normalize_smiles_field(self, smiles_field):
        """
        ورودی می‌تواند رشته ساده یا رشته‌ی نمایانگر لیست باشد.
        این تابع لیستی از SMILES واقعی برمی‌گرداند.
        """
        if pd.isna(smiles_field):
            return []
        if isinstance(smiles_field, str):
            s = smiles_field.strip()
            # اگر به صورت "['A','B']" است، سعی کن literal_eval کنی
            if (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            # اگر جداکننده‌ای مثل ; یا | دارد، آن را هم هندل کن
            if ";" in s:
                return [p.strip() for p in s.split(";") if p.strip()]
            if "|" in s:
                return [p.strip() for p in s.split("|") if p.strip()]
            # در غیر این صورت یک SMILES ساده است
            return [s]
        elif isinstance(smiles_field, (list, tuple)):
            return [str(x) for x in smiles_field]
        else:
            return [str(smiles_field)]

    def add_molecule(self, name: str, smiles: str):
        """
        اضافه کردن یک مولکول (ممکن است چند SMILES باشد)؛
        همچنین مولکول را در molecules_db نیز ثبت می‌کند.
        """
        if self.es is None:
            raise RuntimeError("Elasticsearch is not enabled.")
        try:
            smiles_list = self._normalize_smiles_field(smiles)
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    print(f"Invalid SMILES for {name}: {s}")
                    continue

                descriptors = self.converter.compute_descriptors(s)

                doc = {
                    "name": name,
                    "smiles": s,
                    "descriptors": descriptors,
                }
                self.es.index(index=self.index_name, body=doc)
                # update in-memory DB (store last seen smiles for that name)
                if name not in self.molecules_db:
                    self.molecules_db[name] = {"smiles": [], "descriptors": []}
                self.molecules_db[name]["smiles"].append(s)
                self.molecules_db[name]["descriptors"].append(descriptors)
        except Exception as e:
            print(f"Error adding molecule {name}: {e}")

    def load_dataset(self, dataset_path: str):
        if self.es is None:
            raise RuntimeError("Elasticsearch is not enabled.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Normalize common column names to lower-case and underscores for flexibility
        cols = {c.lower().strip(): c for c in df.columns}
        # try to map typical names
        name_col = cols.get("name") or cols.get("compound_name") or cols.get("compound name") or cols.get("compound")
        smiles_col = cols.get("smiles") or cols.get("smiles_representation") or cols.get("smiles representation") or cols.get("smiles_rep")

        if name_col is None or smiles_col is None:
            raise ValueError(f"Dataset must contain columns for name and smiles. Found: {list(df.columns)}")

        actions = []
        count = 0
        for _, row in df.iterrows():
            try:
                name = str(row[name_col])
                smiles_field = row[smiles_col]
                smiles_list = self._normalize_smiles_field(smiles_field)
                for s in smiles_list:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None:
                        continue

                    descriptors = self.converter.compute_descriptors(s)

                    doc = {
                        "_index": self.index_name,
                        "_source": {
                            "name": name,
                            "smiles": s,
                            "descriptors": descriptors,
                        },
                    }
                    actions.append(doc)
                    # update in-memory DB
                    if name not in self.molecules_db:
                        self.molecules_db[name] = {"smiles": [], "descriptors": []}
                    self.molecules_db[name]["smiles"].append(s)
                    self.molecules_db[name]["descriptors"].append(descriptors)
                    count += 1
            except Exception as e:
                print(f"Error adding molecule {row.get(name_col, 'UNKNOWN')}: {e}")

        if actions:
            helpers.bulk(self.es, actions)
            print(f"Loaded and stored {count} compounds from {dataset_path} into Elasticsearch.")
        else:
            print("No valid molecules found in dataset.")

    def search_by_name(self, query: str, top_k: int = 5):
        if self.es is None:
            raise RuntimeError("Elasticsearch is not enabled.")

        body = {
            "query": {
                "match": {"name": {"query": query, "fuzziness": "AUTO"}}
            }
        }
        res = self.es.search(index=self.index_name, body=body, size=top_k)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def search_by_descriptor(self, descriptor: str, value: float, tolerance: float = 0.1, top_k: int = 5):
        if self.es is None:
            raise RuntimeError("Elasticsearch is not enabled.")

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

    def compute_fingerprint(self, smiles: str):
        """
        Compute Morgan fingerprint (ECFP4) for a molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    def search_similar_molecules(self, query_smiles: str, top_k: int = 5):
        """
        Search for molecules most similar to the given SMILES.
        Uses Tanimoto similarity on Morgan fingerprints.
        """
        query_fp = self.compute_fingerprint(query_smiles)
        if query_fp is None:
            raise ValueError(f"Invalid SMILES: {query_smiles}")

        # از ایندکس ES همه مولکول‌ها رو بیاریم (ممکنه بزرگ باشه)
        res = self.es.search(index=self.index_name, body={"query": {"match_all": {}}}, size=10000)
        hits = res["hits"]["hits"]

        similarities = []
        for hit in hits:
            mol_data = hit["_source"]
            smiles = mol_data.get("smiles")
            mol_fp = self.compute_fingerprint(smiles)
            if mol_fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(query_fp, mol_fp)
            similarities.append((mol_data["name"], smiles, sim))

        # مرتب‌سازی بر اساس similarity
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]

        return similarities

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
