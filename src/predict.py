import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import VAL_TRANSFORMS
from src.model import load_model

MODEL_DIR = Path("model")
_model = None
_classes = None


def _load():
    global _model, _classes
    if _model is None:
        _classes = json.loads((MODEL_DIR / "classes.json").read_text())
        _model = load_model(str(MODEL_DIR / "model.pth"), num_classes=len(_classes))


def predict_image(image: Image.Image) -> dict:
    _load()
    tensor = VAL_TRANSFORMS(image).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5 = probs.topk(min(5, len(_classes)))
    results = [
        {"label": _classes[i], "confidence": round(float(p), 4)}
        for i, p in zip(top5.indices.tolist(), top5.values.tolist())
    ]
    return {"prediction": results[0]["label"], "confidence": results[0]["confidence"], "top5": results}
