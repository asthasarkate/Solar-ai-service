# Solar AI Fault Detection Service

The **Solar AI Fault Detection Service** is a Python-based microservice built using FastAPI and Keras/TensorFlow. It acts as an image classification API for detecting and evaluating faults on solar panels (such as dust, cracks, physical damage, and shading). It receives images via a `/predict` endpoint and returns a detailed analysis including the fault type, severity, confidence level, professional recommendations, and DIY (Do-It-Yourself) safety guidance.

---

## 📂 Detailed Project Structure

```text
Solar-ai-service/
├── app/                             # FastAPI core application logic
│   ├── __init__.py                  
│   ├── main.py                      # FastAPI application, lifecycle events, and endpoints
│   ├── mock_predictor.py            # Fallback mock logic when model fails to load
│   ├── model_loader.py              # Utility to load the production Keras model
│   ├── preprocessing.py             # Image bytes to tensor preprocessing logic
│   ├── severity_engine.py           # Business logic mapping predictions to DIY safety
│   └── test_severity_engine.py      # Unit tests for the severity engine
├── models/                          # Live Production Models
│   ├── production_class_indices.json# JSON mapping output classes to names (e.g. 0: cracks)
│   └── production_model.keras       # The trained Deep Learning model (~28MB)
├── model-training/                  # Data Science & Model Training Pipeline
│   ├── dataset_raw/                 # (Ignored in Git) Raw datasets from Kaggle
│   ├── dataset/                     # (Ignored in Git) Preprocessed training images
│   ├── models/                      # Intermediate and candidate models
│   ├── notebooks/                   # Jupyter Notebooks for the ML workflow
│   │   ├── 01_data_preprocessing.ipynb
│   │   ├── 02_synthesize_shading.ipynb
│   │   ├── 03_train_vgg16.ipynb
│   │   └── 04_train_mobilenetv2.ipynb
│   ├── results/                     # Training results (confusion matrices, graphs)
│   ├── config.py                    # Training hyperparameters (BATCH_SIZE, IMG_SIZE)
│   └── README.md                    # Instructions specific to training models
├── test-images/                     # Sample solar images used for testing the API
├── .env.example                     # Example environment variables
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies (FastAPI, TensorFlow, etc.)
├── update_backend.py                # Helper script to sync schemas to a Node.js backend
└── README.md                        # Project documentation (This File)
```

---

## 🛠️ Complete Installation Guide (Step-by-Step)

Follow these steps exactly to install and run the project on your local machine.

### Step 1: Prerequisites
Ensure you have the following installed on your computer:
*   **Python 3.9 or higher**: Download from [python.org](https://www.python.org/downloads/)
*   **Git**: Download from [git-scm.com](https://git-scm.com/downloads)

### Step 2: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone <your-github-repo-url>
cd Solar-ai-service
```

### Step 3: Create a Virtual Environment
It is highly recommended to create a virtual environment to install dependencies without affecting your system-wide Python.
```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment
Activate the environment you just created. The command depends on your Operating System:

**On Windows:**
```cmd
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
source venv/bin/activate
```
*(You should now see `(venv)` at the beginning of your terminal prompt).*

### Step 5: Install Project Dependencies
Install all required Python packages (including FastAPI, Uvicorn, and TensorFlow/Keras libraries) from the requirements file:
```bash
pip install -r requirements.txt
```

### Step 6: Configure Environment Variables
Copy the example `.env` file to set up your local environment:
**On Windows (Command Prompt):**
```cmd
copy .env.example .env
```
**On Mac/Linux / Windows (PowerShell):**
```bash
cp .env.example .env
```

---

## 🚀 Running the Application

Once everything is installed and your virtual environment is active, start the server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: The `--reload` flag automatically restarts the server if you make changes to the Python files.*

You will see output indicating that the model has loaded and the server is listening. 

### Verify it's working:
Open your web browser and navigate to the Health Check endpoint:
[http://localhost:8000/health](http://localhost:8000/health)

You should see a JSON response confirming the server is healthy:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mock_mode": false
}
```

---

## 📡 Testing the API Prediction

You can test the actual AI prediction endpoint using tools like **Postman**, **cURL**, or Python.

**Endpoint:** `POST /predict`

**Using cURL (from another terminal):**
```bash
curl -X POST -F "image=@test-images/test.jpg" http://localhost:8000/predict
```

**Expected JSON Response:**
```json
{
  "faultType": "Dust",
  "severity": "Medium",
  "confidence": 92.5,
  "recommendation": "Clean the panels.",
  "recommendedProfessional": "Professional Panel Cleaning Service",
  "diyGuidance": "Safe to clean yourself. Turn off the system if possible, then gently rinse the panel with plain water...",
  "diySafe": true
}
```

---

## 🤖 Model Retraining Details
If you wish to retrain the model with new data or compare different model architectures (like VGG16 vs MobileNetV2):
1. Navigate to the `model-training/` directory.
2. Read the local `model-training/README.md` file for strict instructions on placing raw datasets.
3. Run the Jupyter Notebooks sequentially in the `model-training/notebooks/` directory. 
4. Once a new model is generated and validated, copy it to `models/production_model.keras` to serve it in production.
