import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from utils.dataset_handler import DatasetHandler
from search.search_engine import MoleculeSearchEngine
import uvicorn
import io

app = FastAPI(title="Molecular Similarity API")

# Initialize search engine and dataset handler (singleton for the app)
search_engine = MoleculeSearchEngine(use_elasticsearch=True)
dataset_handler = DatasetHandler(search_engine)

MODEL_PATH = 'molecular_gnn_model.pth'

# Try to load existing model on startup (if implemented)
try:
    search_engine.load_model(MODEL_PATH)
except Exception:
    pass


class SearchRequest(BaseModel):
    smiles: str
    top_k: int = 5


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns: name, smiles (or Compound Name, SMILES Representation).
    The molecules will be added to the database and embeddings generated.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        # Read CSV into pandas (use BytesIO to support UploadFile.file)
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        # try to normalize column names if not standard
        cols = [c.lower().strip() for c in df.columns]
        if 'name' not in cols or 'smiles' not in cols:
            # try to rename common variants
            rename_map = {}
            for c in df.columns:
                lc = c.lower().strip()
                if lc in ['compound name', 'compound_name', 'compoundname']:
                    rename_map[c] = 'name'
                if lc in ['smiles representation', 'smiles_representation', 'smiles representation']:
                    rename_map[c] = 'smiles'
            if rename_map:
                df = df.rename(columns=rename_map)
        # final check
        if 'name' not in df.columns or 'smiles' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'name' and 'smiles' columns (or equivalent variants).")

        # Use dataset handler to update (this will call search_engine.add_molecule)
        dataset_handler.update_dataset(df)
        # Save model if available
        try:
            search_engine.save_model(MODEL_PATH)
        except Exception:
            pass
        return {"message": f"Uploaded and processed {len(df)} molecules."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {e}")


@app.post("/search-similar")
def search_similar(request: SearchRequest):
    try:
        results = search_engine.search_similar_molecules(request.smiles, top_k=request.top_k)
        output = []
        for name, smiles, score in results:
            output.append({
                'name': name,
                'smiles': smiles,
                'similarity_score': float(score)
            })
        return {"results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.get("/search-by-name")
def search_by_name(name: str, top_k: int = 5):
    """
    Search by molecule name (fuzzy match). Returns ES documents.
    """
    try:
        docs = search_engine.search_by_name(name, top_k=top_k)
        output = []
        for d in docs:
            output.append({
                "name": d.get("name"),
                "smiles": d.get("smiles"),
                "descriptors": d.get("descriptors")
            })
        return {"query": name, "results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search by name failed: {e}")


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
