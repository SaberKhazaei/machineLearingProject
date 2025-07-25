import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from utils.dataset_handler import DatasetHandler
from search.search_engine import MolecularSearchEngine
import uvicorn

app = FastAPI(title="Molecular Similarity API")

# Initialize search engine and dataset handler (singleton for the app)
search_engine = MolecularSearchEngine(use_elasticsearch=True)
dataset_handler = DatasetHandler(search_engine)

MODEL_PATH = 'molecular_gnn_model.pth'

# Try to load existing model on startup
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
    Upload a CSV file with columns: name, smiles.
    The molecules will be added to the database and embeddings generated.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        df = pd.read_csv(file.file)
        if 'name' not in df.columns or 'smiles' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'name' and 'smiles' columns.")
        dataset_handler.update_dataset(df)
        # Save model after update
        search_engine.save_model(MODEL_PATH)
        return {"message": f"Uploaded and processed {len(df)} molecules."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {e}")

@app.post("/search-similar")
def search_similar(request: SearchRequest):
    """
    Given a SMILES string, return the most similar molecules from the database.
    """
    try:
        results = search_engine.search_similar_molecules(request.smiles, top_k=request.top_k)
        # Format results for output
        output = []
        for name, score in results:
            # Get SMILES from molecules_db if available
            smiles = search_engine.molecules_db.get(name, {}).get('smiles', None)
            output.append({
                'name': name,
                'smiles': smiles,
                'similarity_score': score
            })
        return {"results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 