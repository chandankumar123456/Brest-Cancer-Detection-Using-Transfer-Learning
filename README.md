# OnchoScan AI: Breast Cancer Detection (Transfer Learning + Grad-CAM)

OnchoScan AI is an end-to-end web application for breast histopathology image classification.
It predicts whether a patch is likely Benign (IDC-) or Malignant (IDC+) using transfer learning,
returns confidence probabilities, and provides Grad-CAM explainability to highlight influential
regions of the image.

The system also includes an LLM-assisted hospital recommendation module that suggests nearby
healthcare options based on user location and predicted result.

## 1) What This Project Delivers

1. End-to-end pipeline from model training to web inference.
2. Transfer learning-based binary classification for IDC histopathology.
3. Grad-CAM heatmap generation for visual explainability.
4. Frontend for image upload, prediction display, confidence bars, and heatmap rendering.
5. FastAPI backend with documented endpoints.
6. LLM-assisted hospital recommendation workflow with fallback behavior.

## 2) Tech Stack

### Backend
- Python
- FastAPI + Uvicorn
- TensorFlow / Keras
- Pillow, NumPy, scikit-learn
- Matplotlib, Seaborn (evaluation visualizations)

### Frontend
- HTML
- CSS
- Vanilla JavaScript

### Data and Integrations
- Kaggle IDC histopathology dataset
- OpenStreetMap Nominatim (geocoding)
- OpenStreetMap Overpass (nearby hospitals)
- OpenAI-compatible LLM endpoint or local Ollama

## 3) Project Structure

```text
Breast Cancer Detection/
|- backend/
|  |- main.py
|  |- model.py
|  |- train.py
|  |- hospital_recommender.py
|  |- requirements.txt
|  `- saved_model/
|- frontend/
|  |- index.html
|  |- styles.css
|  `- app.js
|- run.sh
`- README.md
```

## 4) System Architecture

### Inference Flow
1. User uploads image from browser.
2. Frontend sends multipart request to `/predict`.
3. Backend validates file type and size.
4. Image is preprocessed to 224x224 RGB and normalized.
5. Transfer-learning model returns malignant probability.
6. Backend returns label, confidence, and class probabilities.
7. If requested, Grad-CAM overlay is generated and returned as base64 PNG.
8. Frontend renders prediction card, bars, and heatmap.

### Hospital Recommendation Flow
1. User enters location in frontend.
2. Frontend calls `/recommend-hospitals` with location + diagnosis.
3. Backend geocodes location to latitude/longitude.
4. Backend fetches nearby hospitals from OpenStreetMap.
5. Backend requests LLM summary (OpenAI-compatible, else Ollama, else fallback text).
6. Frontend renders summary and ranked hospitals list.

## 5) Transfer Learning Design

Training is transfer-learning based and supports EfficientNet backbones:

- EfficientNetB0 (default, faster)
- EfficientNetB3 (balanced)
- EfficientNetB7 (heavier, slower)

Model head:
1. GlobalAveragePooling
2. BatchNormalization
3. Dense + Dropout blocks
4. Sigmoid output for binary classification

Training strategy is phased:
1. Phase 1: freeze backbone, train head.
2. Phase 2: unfreeze top layers and fine-tune.
3. Phase 3: deeper fine-tuning with lower learning rate.

Additional training features:
- Class weighting for imbalance
- Optional focal loss
- Data augmentation
- Optional mixup
- Validation-threshold selection for better F1

## 6) How Grad-CAM Works (Conceptual)

Grad-CAM provides model explainability by highlighting image regions that most affected
the final prediction.

Process used in this project:
1. Forward pass gets feature maps from the last convolutional layer.
2. Gradients of target class score are computed with respect to feature maps.
3. Spatial importance weights are produced from pooled gradients.
4. Weighted feature maps are combined into a heatmap.
5. Heatmap is normalized, resized, and overlaid on input image.
6. Result is returned to frontend and shown as explainability image.

Interpretation note:
Grad-CAM is an aid for model interpretation, not a clinical ground truth mask.

## 7) Prerequisites

1. Python 3.11 recommended.
2. macOS/Linux/Windows supported (GPU optional).
3. Kaggle credentials for automated dataset download.

## 8) Clone and Setup

```bash
git clone <your-repository-url>
cd "Breast Cancer Detection"
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

## 9) Dataset Setup

### Option A: Kaggle API (recommended)

```bash
mkdir -p ~/.kaggle
# place kaggle.json here and ensure permissions:
chmod 600 ~/.kaggle/kaggle.json
```

Then run training; dataset will auto-download.

### Option B: Manual download

1. Download dataset: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
2. Extract into backend dataset directory.
3. Run training command.

## 10) Training Commands

### Fast transfer-learning profile (recommended for quick completion)

```bash
cd backend
source ../venv/bin/activate
python train.py --backbone EfficientNetB0 --data-fraction 0.35 --batch-size 64 --epochs-head 2 --epochs-finetune-1 3 --epochs-finetune-2 4
```

### Higher-capacity profile

```bash
cd backend
source ../venv/bin/activate
python train.py --backbone EfficientNetB3 --data-fraction 0.6 --batch-size 32 --epochs-head 3 --epochs-finetune-1 5 --epochs-finetune-2 6
```

### Full heavy profile (slowest)

```bash
cd backend
source ../venv/bin/activate
python train.py --backbone EfficientNetB7 --data-fraction 1.0 --batch-size 16 --epochs-head 4 --epochs-finetune-1 8 --epochs-finetune-2 10 --enable-mixup
```

Training outputs:
- backend/saved_model/breast_cancer_model.keras
- backend/saved_model/metrics_report.json
- backend/saved_model/confusion_matrix.png

## 11) Run the Application

### Option A: direct server start

```bash
cd backend
source ../venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option B: helper script

```bash
./run.sh
```

Open in browser:
- http://localhost:8000
- API docs: http://localhost:8000/docs

## 12) API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Serve frontend |
| GET | /health | Backend health + model loaded status |
| GET | /model-info | Model metadata |
| POST | /predict | Prediction from uploaded image |
| POST | /recommend-hospitals | LLM-assisted nearby hospital recommendations |

### Predict Request
- Content-Type: multipart/form-data
- Field: file
- Supported types: jpeg, png, bmp, tiff, webp
- Max size: 20 MB

Optional query:
- include_heatmap=true

### Hospital Recommendation Request Body

```json
{
  "location": "Hyderabad, Telangana",
  "diagnosis": "Malignant",
  "radius_km": 35
}
```

## 13) Environment Variables

### Model/Inference
- PREDICTION_THRESHOLD (default: 0.5)

### LLM (OpenAI-compatible)
- LLM_API_KEY
- LLM_API_BASE (default: https://api.openai.com/v1)
- LLM_MODEL (default: gpt-4o-mini)

### LLM (Ollama fallback)
- OLLAMA_MODEL (default: llama3.1:8b)

LLM fallback order:
1. OpenAI-compatible endpoint
2. Local Ollama endpoint
3. Built-in deterministic text fallback

## 14) Model Section (Placeholders)

This section is intentionally left as placeholders for your final training run.

- Backbone used: 
- Dataset fraction: 
- Batch size: 
- Best validation AUC: 
- Best validation F1: 
- Test accuracy: 
- Test precision: 
- Test recall: 
- Test F1: 
- Selected threshold: 
- Notes: 

## 15) Professional Notes and Limitations

1. This is a research-support tool, not a diagnostic medical device.
2. Model performance depends on dataset quality, stain variability, and domain shift.
3. Grad-CAM highlights importance regions but does not prove pathology boundaries.
4. Hospital recommendations are logistics guidance, not treatment advice.

## 16) Troubleshooting

1. TensorFlow install fails:
Use Python 3.11 environment.

2. Training too slow:
Use EfficientNetB0, higher batch size, lower data fraction.

3. No model loaded at inference:
Confirm backend/saved_model/breast_cancer_model.keras exists.

4. Hospital recommendation LLM unavailable:
Set LLM credentials or run Ollama locally; fallback summary still works.

## 17) Disclaimer

This project is for research and educational use only.
It is not FDA approved and must not be used as a standalone clinical diagnosis system.
Always consult qualified medical professionals for clinical decisions.
