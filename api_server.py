import os
import io
import logging
from typing import List, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

from utils.dataset_handler import DatasetHandler
from search.search_engine import MoleculeSearchEngine

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

# ---------- App & singletons ----------
app = FastAPI(title="Molecular Search API")

# Initialize search engine and dataset handler (singleton for the app)
# use_elasticsearch=True assumed; اگر می‌خواهی بدون ES اجرا کنی این را False کن
search_engine = MoleculeSearchEngine(use_elasticsearch=True)
dataset_handler = DatasetHandler(search_engine)

MODEL_PATH = "molecular_gnn_model.pth"

# Try to load existing model on startup (اگر متد موجود باشد)
try:
    search_engine.load_model(MODEL_PATH)
    logger.info("Loaded pre-trained model from %s", MODEL_PATH)
except Exception as e:
    # ممکن است پیاده‌سازی save/load مدل را نداشته باشید — این مشکلی نیست
    logger.info("No pre-trained model loaded (or load_model not implemented). %s", str(e))


# ---------- Pydantic models ----------
class SearchRequest(BaseModel):
    smiles: str
    top_k: int = 5


# ---------- Endpoints ----------

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns: name, smiles (or common variants).
    The molecules will be added to the database (Elasticsearch) and embeddings/descriptors generated.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Normalize column names (support common variants)
        cols = [c.lower().strip() for c in df.columns]
        if "name" not in cols or "smiles" not in cols:
            rename_map = {}
            for c in df.columns:
                lc = c.lower().strip()
                if lc in ["compound name", "compound_name", "compoundname"]:
                    rename_map[c] = "name"
                if lc in ["smiles representation", "smiles_representation", "smiles representation"]:
                    rename_map[c] = "smiles"
            if rename_map:
                df = df.rename(columns=rename_map)

        if "name" not in df.columns or "smiles" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'name' and 'smiles' columns (or common variants such as 'Compound Name' and 'SMILES Representation').",
            )

        # Use dataset handler to update (this will call search_engine.add_molecule)
        dataset_handler.update_dataset(df)

        # Attempt to save model if implemented
        try:
            search_engine.save_model(MODEL_PATH)
        except Exception:
            logger.debug("save_model not implemented or failed; continuing.")

        return {"message": f"Uploaded and processed {len(df)} molecules."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process uploaded CSV")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {e}")


@app.post("/search-similar")
def search_similar(request: SearchRequest):
    """
    Search molecules similar to the given SMILES.
    Returns: list of {name, smiles, similarity_score}
    """
    try:
        results = search_engine.search_similar_molecules(request.smiles, top_k=request.top_k)
        output = [
            {
                "name": name,
                "smiles": smiles,
                "similarity_score": float(score),
            }
            for name, smiles, score in results
        ]
        return {"query": request.smiles, "results": output}
    except ValueError as ve:
        # e.g. invalid SMILES
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Similarity search failed")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


@app.get("/search-by-name")
def search_by_name(
    name: str,
    top_k: int = 5,
    include_docs: bool = False
):
    """
    Search by molecule name (fuzzy match). Modified behavior:
      - find matching documents for 'name' (Elasticsearch fuzzy match),
      - take the first matched document's SMILES,
      - compute similarity (Morgan fingerprints / Tanimoto) against indexed molecules,
      - return ONLY the NAMES of similar molecules + similarity scores.

    Parameters:
      - name: query molecule name (string)
      - top_k: how many similar names to return (default 5)
      - include_docs: if True, also include matched documents (name, smiles, descriptors) in the response

    Returns JSON:
    {
      "query": "<name>",
      "primary_smiles": "<smiles used for similarity search>",
      "similar_molecules": [ {"name": "<name>", "similarity_score": 0.95}, ... ],
      "matched_documents": [...]   # optional if include_docs=True
    }
    """
    try:
        # 1) fetch matching documents from ES (fuzzy match by name)
        docs = search_engine.search_by_name(name, top_k=top_k)
        if not docs:
            raise HTTPException(status_code=404, detail=f"No molecule found matching name: {name}")

        matched_documents = []
        for d in docs:
            matched_documents.append({
                "name": d.get("name"),
                "smiles": d.get("smiles"),
                "descriptors": d.get("descriptors")
            })

        # 2) determine primary SMILES
        primary_smiles = None
        for d in matched_documents:
            s = d.get("smiles")
            if s and isinstance(s, str) and s.strip():
                primary_smiles = s
                break

        if not primary_smiles:
            db_entry = search_engine.molecules_db.get(name)
            if db_entry:
                smiles_list = db_entry.get("smiles", [])
                if smiles_list:
                    primary_smiles = smiles_list[0]

        if not primary_smiles:
            resp = {"query": name, "primary_smiles": None, "similar_molecules": []}
            if include_docs:
                resp["matched_documents"] = matched_documents
            return resp

        # 3) perform similarity search
        similar_results = search_engine.search_similar_molecules(primary_smiles, top_k=top_k)

        seen = set()
        similar_names = []
        for mol_name, mol_smiles, score in similar_results:
            if mol_name == name:
                continue
            if mol_name in seen:
                continue
            seen.add(mol_name)
            # فقط اسم + نمره شباهت
            similar_names.append({
                "name": mol_name,
                "similarity_score": float(score * 100)  # درصد شباهت
            })

        # 4) assemble response
        resp = {
            "query": name,
            "primary_smiles": primary_smiles,
            "similar_molecules": similar_names
        }
        if include_docs:
            resp["matched_documents"] = matched_documents

        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("search_by_name failed")
        raise HTTPException(status_code=500, detail=f"Search by name failed: {e}")


# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
