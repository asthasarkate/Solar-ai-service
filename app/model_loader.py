import json
from pathlib import Path

from tensorflow import keras

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "production_model.keras"
CLASS_INDICES_PATH = MODELS_DIR / "production_class_indices.json"

_model = None
_class_indices = None
_class_names = None


def load_model():
    global _model, _class_indices, _class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    _model = keras.models.load_model(str(MODEL_PATH))

    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_PATH}")

    with open(CLASS_INDICES_PATH) as f:
        _class_indices = json.load(f)

    _class_names = [
        name for name, idx in sorted(_class_indices.items(), key=lambda x: x[1])
    ]

    return _model


def get_model():
    return _model


def get_class_names():
    return _class_names


def get_class_indices():
    return _class_indices
