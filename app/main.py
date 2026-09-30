import os
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import model_loader, preprocessing, severity_engine

load_dotenv()

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5000")
PORT = int(os.getenv("PORT", "8000"))

_model_loaded = False
_mock_mode = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded, _mock_mode

    try:
        model_loader.load_model()
        _model_loaded = True
    except Exception as e:
        print(f"[WARNING] Failed to load production model: {e}")
        traceback.print_exc()
        print("[INFO] Falling back to mock predictor")
        _mock_mode = True

    model = model_loader.get_model()
    class_names = model_loader.get_class_names()
    mode = "MOCK" if _mock_mode else "PRODUCTION"

    if model is not None:
        arch = model.__class__.__name__
        params = model.count_params()
    else:
        arch = "N/A"
        params = 0

    print("=" * 60)
    print("  Solar Panel Fault Detection API")
    print("=" * 60)
    print(f"  Mode:       {mode}")
    print(f"  Model:      {arch}")
    print(f"  Parameters: {params:,}")
    print(f"  Classes:    {', '.join(class_names) if class_names else 'N/A'}")
    print(f"  Listening:  http://127.0.0.1:{PORT}")
    print("[INFO] /predict expects multipart field name: 'image'")
    print("=" * 60)

    yield

    print("Shutting down...")


app = FastAPI(title="Solar Panel Fault Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    faultType: str
    severity: str
    confidence: float
    recommendation: str
    recommendedProfessional: str
    diyGuidance: str
    diySafe: bool


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model_loaded,
        "mock_mode": _mock_mode,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()

    if _mock_mode:
        from app.mock_predictor import mock_predict

        class_names = model_loader.get_class_names() or [
            "cracks",
            "dust",
            "physical_damage",
            "shading",
        ]
        result = mock_predict(class_names)
        return PredictionResponse(**result)

    try:
        input_tensor = preprocessing.preprocess_image(image_bytes)
        model = model_loader.get_model()
        predictions = model.predict(input_tensor, verbose=0)
        probs = predictions[0]
        class_names = model_loader.get_class_names()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        class_name = class_names[pred_idx]
        result_dict = severity_engine.analyze_prediction(class_name, confidence)

        return PredictionResponse(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
