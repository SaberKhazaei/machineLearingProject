import os
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from utils.dataset_handler import DatasetHandler
from search.search_engine import MoleculeSearchEngine

app = FastAPI(title="Molecular Search API")

# Initialize search engine and dataset handler (singleton for the app)
search_engine = MoleculeSearchEngine(use_elasticsearch=True)
dataset_handler = DatasetHandler(search_engine)

MODEL_PATH = "molecular_gnn_model.pth"

# Try to load existing model on startup
try:
    search_engine.load_model(MODEL_PATH)
except Exception:
    pass


# ---------- Models ----------
class SearchRequest(BaseModel):
    smiles: str
    top_k: int = 5


# ---------- Endpoints ----------
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns: name, smiles (or equivalent variants).
    The molecules will be added to the database and embeddings generated.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Read CSV into pandas
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Normalize column names
        cols = [c.lower().strip() for c in df.columns]
        if "name" not in cols or "smiles" not in cols:
            rename_map = {}
            for c in df.columns:
                lc = c.lower().strip()
                if lc in ["compound name", "compound_name", "compoundname"]:
                    rename_map[c] = "name"
                if lc in ["smiles representation", "smiles_representation"]:
                    rename_map[c] = "smiles"
            if rename_map:
                df = df.rename(columns=rename_map)

        if "name" not in df.columns or "smiles" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'name' and 'smiles' columns (or variants).",
            )

        # Update dataset
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
    """
    Search molecules similar to the given SMILES.
    """
    try:
        results = search_engine.search_similar_molecules(
            request.smiles, top_k=request.top_k
        )
        output = [
            {
                "name": name,
                "smiles": smiles,
                "similarity_score": float(score),
            }
            for name, smiles, score in results
        ]
        return {"query": request.smiles, "results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


@app.get("/search-by-name")
def search_by_name(name: str, top_k: int = 5):
    """
    Search by molecule name (fuzzy match). Returns ES documents.
    """
    try:
        docs = search_engine.search_by_name(name, top_k=top_k)
        output = [
            {
                "name": d.get("name"),
                "smiles": d.get("smiles"),
                "descriptors": d.get("descriptors"),
            }
            for d in docs
        ]
        return {"query": name, "results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search by name failed: {e}")


# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
