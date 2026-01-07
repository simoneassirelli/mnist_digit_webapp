from __future__ import annotations

import base64

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.preprocessing.canvas_to_mnist import canvas_png_bytes_to_mnist_tensor
from src.serving.model_loader import LoadedModel, load_mnist_model
from src.api.schemas import PredictResponse


class PredictRequest(BaseModel):
    # data URL like: "data:image/png;base64,AAAA..."
    image_data_url: str


def create_app() -> FastAPI:
    app = FastAPI(title="MNIST Digit Recognizer")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://simoneassirelli.github.io"],  # dev only
        allow_credentials=False,
        allow_methods=["*"],  # includes OPTIONS
        allow_headers=["*"],
    )
    # model is loaded once when the server starts, not on every request.
    loaded: LoadedModel = load_mnist_model()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if "base64," not in req.image_data_url:
            raise HTTPException(status_code=400, detail="Invalid image_data_url")

        # we strip off "data:image/png;base64," prefix
        b64 = req.image_data_url.split("base64,", 1)[1]
        try:
            # decode base64 into raw PNG bytes (binary)
            png_bytes = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Base64 decode failed")

        # Convert PNG → MNIST tensor (preprocessing)
        x = canvas_png_bytes_to_mnist_tensor(
            png_bytes,
            normalize_mean=loaded.normalize_mean,
            normalize_std=loaded.normalize_std,
        ).to(loaded.device)

        with torch.no_grad():
            # logits are raw scores
            logits = loaded.model(x)
            # softmax turns scores into probabilities
            probs = torch.softmax(logits, dim=1)[0]
            # argmax returns the index of the max value
            pred = int(torch.argmax(probs).item())
            # max returns the max value
            conf = float(torch.max(probs).item())

        return PredictResponse(prediction=pred, confidence=conf)

    return app


app = create_app()
