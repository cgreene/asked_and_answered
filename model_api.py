from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import uvicorn

# Create the FastAPI app.
app = FastAPI()

# Define the request schema.
class QueryRequest(BaseModel):
    text: str
    normalize: bool = True
    # precision can be: "float32", "int8", "uint8", "binary", "ubinary" or None
    # Default to None for standard float32 embeddings
    precision: str | None = None

# Global variable to hold the model.

# On startup, load the model once.
model = None
model_ready = False


def _load_model():
    """Internal helper to load the SentenceTransformer model once.

    This is intentionally synchronous and called on first request so the
    FastAPI process can start immediately and report a health state.
    """
    global model, model_ready
    if model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    try:
        model.to(device)
    except Exception:
        # Some SentenceTransformer model objects may not implement .to(); ignore.
        pass
    model_ready = True
    print("Model loaded and moved to device:", device)

# Define the endpoint to encode a query.
@app.post("/encode")
async def encode(request: QueryRequest):
    global model
    # Load the model lazily on first request to avoid blocking FastAPI startup.
    if model is None:
        _load_model()

    with torch.no_grad():
        embedding = model.encode(
            [request.text],
            normalize_embeddings=request.normalize,
            precision=request.precision,
        )
    # Convert the NumPy array to a list so it can be serialized as JSON.
    return {"embedding": embedding.tolist()}


@app.get("/health")
async def health():
    """Health endpoint to report model readiness for smoke tests."""
    return {"ready": bool(model_ready)}

# Run the app with Uvicorn if this file is executed directly.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
