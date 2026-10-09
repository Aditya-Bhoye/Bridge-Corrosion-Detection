"""Local inference API. Run from repository root: python -m uvicorn backend.app:app"""
import base64
import io
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .model import UNet, enhance

app = FastAPI(title="BridgeGuard inference")
weights = Path(os.environ.get("MODEL_PATH", "backend/corrosion_unet_enhanced.pth"))
model = None
lock = Lock()
MAX_BYTES = 10 * 1024 * 1024


@app.get("/api/health")
def health():
    return {"model_ready": weights.is_file(), "model": "U-Net", "input_size": 256}


def predict(payload):
    global model
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > 25_000_000:
                raise HTTPException(413, "Image exceeds 25 megapixels.")
            original = source.convert("RGB")
            enhanced = Image.fromarray(enhance(np.array(original)))
            resized = enhanced.resize((256, 256), Image.Resampling.BILINEAR)
            tensor = torch.tensor(np.array(resized).transpose(2, 0, 1) / 255., dtype=torch.float32).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not decode image.")
    with lock:
        if model is None:
            try:
                candidate = UNet()
                candidate.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
                candidate.eval()
                model = candidate
            except Exception:
                raise HTTPException(503, "Model weights could not be loaded. Export the notebook's state_dict and check MODEL_PATH.")
        with torch.inference_mode():
            mask = model(tensor).squeeze().numpy() > 0.5
    coverage = round(float(mask.mean() * 100), 2)
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[mask] = [235, 121, 58, 210]
    overlay = Image.fromarray(rgba).resize(original.size, Image.Resampling.NEAREST)
    stream = io.BytesIO()
    overlay.save(stream, format="PNG")
    return {"coverage": coverage, "mask": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(), "model": "U-Net"}


@app.post("/api/detect")
async def detect(file: UploadFile):
    try:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(415, "Use a JPEG, PNG or WebP image.")
        payload = await file.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise HTTPException(413, "Image must be smaller than 10 MB.")
        if not payload:
            raise HTTPException(400, "Image is empty.")
        if not weights.is_file():
            raise HTTPException(503, "Trained weights are missing. Add backend/corrosion_unet_enhanced.pth or set MODEL_PATH.")
        return await run_in_threadpool(predict, payload)
    finally:
        await file.close()
