# Solar AI Fault Detection Service

The Solar AI Fault Detection Service is a Python-based microservice built with FastAPI and Keras/TensorFlow. It provides an API to analyze images of solar panels and detect faults such as **Dust, Cracks, Physical Damage, and Shading**. It evaluates the severity of the fault, suggests recommendations, and determines if it is safe for users to perform Do-It-Yourself (DIY) repairs or if a professional is needed.

## Features
- **FastAPI Backend**: High-performance asynchronous API for image inference.
- **Deep Learning Model**: Uses a Keras/TensorFlow model trained to classify multiple fault types.
- **Severity Engine**: Business logic to translate AI confidence scores into human-readable advice, severity levels, and DIY safety guidelines.
- **Graceful Fallback**: Reverts to a mock prediction mode if the AI model fails to load, ensuring the API stays online.

## Project Structure
```text
Solar-ai-service/
├── app/                        # FastAPI application code
│   ├── main.py                 # FastAPI application and endpoints
│   ├── severity_engine.py      # Business logic for fault severity and DIY safety
│   ├── model_loader.py         # Utility to load Keras model
│   └── preprocessing.py        # Image tensor preprocessing
├── models/                     # Production model files
│   └── production_model.keras  # Trained Keras model
├── model-training/             # Model training pipelines (Jupyter Notebooks & datasets)
├── test-images/                # Sample images for testing the API
├── .env.example                # Example environment variables
└── requirements.txt            # Python dependencies
```

## Prerequisites
- Python 3.9+
- pip (Python package installer)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd Solar-ai-service
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Copy the example environment file and adjust if necessary:
   ```bash
   cp .env.example .env
   ```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will now be listening on `http://localhost:8000`.

## API Endpoints

### 1. Health Check
- **Endpoint:** `GET /health`
- **Description:** Verifies that the service is running and indicates if the production model or mock predictor is loaded.

### 2. Predict Fault
- **Endpoint:** `POST /predict`
- **Description:** Accepts an image upload to predict panel faults.
- **Payload:** `multipart/form-data` with key `image`.
- **Response Example:**
  ```json
  {
    "faultType": "Dust",
    "severity": "Medium",
    "confidence": 92.5,
    "recommendation": "Clean the panels.",
    "recommendedProfessional": "Professional Panel Cleaning Service",
    "diyGuidance": "Safe to clean yourself...",
    "diySafe": true
  }
  ```

## Model Training
If you wish to retrain the model, please refer to the `model-training/README.md` file for instructions on placing raw datasets and running the Jupyter Notebooks.
