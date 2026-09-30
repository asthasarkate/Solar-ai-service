import numpy as np

from app import severity_engine

def mock_predict(class_names: list[str]) -> dict:
    probs = np.random.dirichlet(np.ones(len(class_names)))
    pred_idx = int(np.argmax(probs))
    raw_class = class_names[pred_idx]
    raw_confidence = float(probs[pred_idx])
    return severity_engine.analyze_prediction(raw_class, raw_confidence)
