# Solar AI Fault Detection Service

The **Solar AI Fault Detection Service** is a Python-based microservice built using FastAPI and Keras/TensorFlow. It acts as an image classification API for detecting and evaluating faults on solar panels (such as dust, cracks, physical damage, and shading). It receives images via a `/predict` endpoint and returns a detailed analysis including the fault type, severity, confidence level, professional recommendations, and DIY (Do-It-Yourself) safety guidance.

---

## 🏗️ Comprehensive Codebase & Architecture Analysis

The codebase is highly organized into two distinct sections: the **API Service** running the backend, and the **Model Training Lifecycle** used for research and development. 

### 1. The FastAPI Application (`/app/`)
This directory holds the core logic for the web server and prediction pipeline.

*   **`main.py`** 
    *   **Purpose:** The entry point for the FastAPI server.
    *   **Highlights:** Uses a `lifespan` context manager to load the production model into memory at startup. If the model fails to load, it falls back to a **mock predictor mode** to ensure the API stays online. It exposes the `/health` and `/predict` endpoints.
*   **`severity_engine.py`** 
    *   **Purpose:** Interprets the model's raw predictions and applies business logic to provide actionable insights.
    *   **Highlights:** Uses predefined dictionaries (`SEVERITY_MAP`, `RECOMMENDATION_MAP`, `PROFESSIONAL_MAP`, and `DIY_GUIDANCE`) to interpret faults. The `analyze_prediction()` function ties these together to return structured data like `diySafe` (Boolean indicating if the user can fix the issue themselves) and `recommendedProfessional` (e.g., "Certified Solar Panel Technician").
*   **`model_loader.py`**: Fetches and loads the `.keras` model and class names.
*   **`preprocessing.py`**: Prepares incoming image bytes into the tensor format required for the Keras model.
*   **`mock_predictor.py`**: Randomly generates predictions when the service is running in mock mode.
*   **`test_severity_engine.py`**: Unit tests for the business logic in `severity_engine.py`.

### 2. Production Models (`/models/`)
Contains the finalized artifacts that are actively used by the FastAPI service.
*   **`production_model.keras`**: The trained AI model (weights ~28MB).
*   **`production_class_indices.json`**: Maps the model's numerical outputs to human-readable string labels (e.g., `0 -> cracks`, `1 -> dust`).

### 3. Model Training Pipeline (`/model-training/`)
This directory is isolated from the main web server and contains the research and development pipeline for creating the solar fault detection model.
*   **`README.md`**: Documents instructions for placing raw datasets from Kaggle and Roboflow into `/dataset_raw/`. It explains that the pipeline consolidates these datasets to recognize four specific classes: `cracks`, `dust`, `physical_damage`, and `shading`.
*   **`config.py`**: Central configuration file containing model hyperparameters such as `IMG_SIZE = (224, 224)` and `BATCH_SIZE = 32`.
*   **`/notebooks/`**: Contains the Jupyter notebooks and scripts for the ML workflow (e.g., `01_data_preprocessing.ipynb` for data cleaning, `02_synthesize_shading.py` for data augmentation, and training notebooks for VGG16 and MobileNetV2).

### 4. Root Level Utility Files
*   **`update_backend.py`**: A helper script written to patch external NodeJS backend files outside of this repository. It automatically finds and replaces text inside Express controllers and Mongoose schemas to inject the `diySafe` and `diyGuidance` fields.
*   **`requirements.txt`**: Lists Python dependencies, including `fastapi`, `uvicorn`, `h5py`, and `absl-py`.
*   **`/test-images/`**: Contains sample solar panel images used for manually testing the API endpoints.

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.9+
- pip (Python package installer)
- Git

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
cd Solar-ai-service
```

### 2. Create a Virtual Environment (Recommended)
Isolating dependencies ensures they do not interfere with other Python projects on your system.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Copy the example environment file and adjust the ports or CORS origins if necessary:
```bash
cp .env.example .env
```

---

## 🚀 Running the Application

Once dependencies are installed, you can start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: The `--reload` flag is useful for development as it restarts the server upon code changes.*

The API will now be listening on `http://localhost:8000`.

---

## 📡 API Endpoints

### 1. Health Check
- **Endpoint:** `GET /health`
- **Description:** Verifies that the service is running and indicates if the production model or mock predictor is currently loaded in memory.
- **Example Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "mock_mode": false
  }
  ```

### 2. Predict Fault
- **Endpoint:** `POST /predict`
- **Description:** Accepts an image upload to predict panel faults. The image is run through the CNN model, and the confidence score is routed through the severity engine.
- **Payload:** `multipart/form-data` with the key name `image`.
- **Response Example:**
  ```json
  {
    "faultType": "Dust",
    "severity": "Medium",
    "confidence": 92.5,
    "recommendation": "Clean the panels.",
    "recommendedProfessional": "Professional Panel Cleaning Service",
    "diyGuidance": "Safe to clean yourself. Turn off the system if possible, then gently rinse the panel with plain water in the early morning or evening...",
    "diySafe": true
  }
  ```

---

## 🤖 Model Retraining
If you wish to retrain the model with new data or compare different model architectures (like VGG16 vs MobileNetV2):
1. Navigate to the `model-training/` directory.
2. Read the local `model-training/README.md` file for strict instructions on placing raw datasets.
3. Run the Jupyter Notebooks sequentially in the `model-training/notebooks/` directory. 
4. Once a new model is generated and validated, copy it to `models/production_model.keras` to serve it in production.
