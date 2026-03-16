# OnchoScan AI — Breast Cancer Detection Using Transfer Learning

> **End-to-end deep-learning web application** for classifying breast histopathology image patches as **Benign (IDC−)** or **Malignant (IDC+)** using EfficientNet transfer learning, Grad-CAM explainability, and LLM-assisted hospital recommendations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture and Workflow](#3-system-architecture-and-workflow)
4. [Dataset Explanation](#4-dataset-explanation)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation and Metrics](#8-evaluation-and-metrics)
9. [Project Directory Structure](#9-project-directory-structure)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Expected Outputs](#11-expected-outputs)
12. [How Predictions Work](#12-how-predictions-work)
13. [Design Decisions](#13-design-decisions)
14. [Limitations and Future Improvements](#14-limitations-and-future-improvements)

---

## 1. Project Overview

OnchoScan AI is a production-ready, full-stack application that detects Invasive Ductal Carcinoma (IDC) in breast tissue histopathology images. The system combines three core capabilities:

1. **AI-powered classification** — An ensemble of up to three EfficientNet models (B0, B3, B7) fine-tuned on the IDC Histopathology dataset classifies 50×50 pixel tissue patches as benign or malignant and returns calibrated probability scores.
2. **Visual explainability** — Grad-CAM heatmaps highlight the tissue regions that most influenced the model's prediction, giving clinicians and researchers an interpretable view of the AI's reasoning.
3. **Hospital recommendation engine** — After a prediction, users can enter their location to receive a ranked list of nearby hospitals fetched from OpenStreetMap, accompanied by an LLM-generated follow-up recommendation.

The system is delivered as a **FastAPI backend** serving a **vanilla JavaScript frontend** with a modern glassmorphism UI. It is designed for research and educational use and is **not a medical device**.

### What This Project Delivers

| Capability | Description |
|---|---|
| End-to-end pipeline | From raw dataset download through model training to web-based inference |
| Transfer learning | EfficientNet backbones (B0/B3/B7) pre-trained on ImageNet, fine-tuned for IDC classification |
| Ensemble inference | Averages probabilities from all available models for more robust predictions |
| Grad-CAM explainability | Generates heatmap overlays showing which tissue regions drive the prediction |
| Modern web interface | Drag-and-drop upload, animated confidence bars, responsive layout |
| Hospital recommendations | Location-based hospital search with LLM summarization (OpenAI, Ollama, or deterministic fallback) |
| Documented API | FastAPI auto-generated interactive docs at `/docs` |

### Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn, TensorFlow/Keras |
| ML Libraries | NumPy, scikit-learn, Matplotlib, Seaborn, Pillow |
| Data | Kaggle IDC Histopathology dataset, kagglehub, kaggle CLI |
| Frontend | HTML5, CSS3 (custom design system), Vanilla JavaScript (ES6+) |
| External APIs | OpenStreetMap Nominatim (geocoding), Overpass API (hospital search) |
| LLM Integration | OpenAI-compatible API, Ollama (local), deterministic fallback |

---

## 2. Problem Statement

### The Medical Problem

Breast cancer is the most commonly diagnosed cancer worldwide. Invasive Ductal Carcinoma (IDC) is the most common subtype, accounting for approximately 80% of all breast cancers. Pathologists diagnose IDC by examining hematoxylin and eosin (H&E) stained tissue sections under a microscope — a process that is time-consuming, subjective, and prone to inter-observer variability.

### The Technical Challenge

Automating IDC detection requires a classifier that can:

- Distinguish subtle morphological differences between benign and malignant tissue in small 50×50 pixel patches extracted from whole-mount slide images.
- Handle significant class imbalance (approximately 57% benign, 43% malignant in the dataset).
- Generalize across stain variations, scanner differences, and tissue preparation artifacts.
- Provide interpretable outputs so that predictions can be reviewed by domain experts.

### Why Transfer Learning

Training a deep convolutional neural network from scratch on medical images is impractical for most teams because:

1. **Limited labeled data** — While 277K patches is large by medical standards, it is small relative to ImageNet-scale datasets needed to train deep architectures from scratch.
2. **Feature reuse** — Low-level features learned on ImageNet (edges, textures, color distributions) transfer effectively to histopathology images.
3. **Training efficiency** — Pre-trained backbones converge faster and require fewer epochs, reducing compute costs.

EfficientNet was chosen specifically because it achieves state-of-the-art accuracy with fewer parameters than alternatives like ResNet or VGG by using compound scaling of depth, width, and resolution.

---

## 3. System Architecture and Workflow

### High-Level Architecture

```
┌─────────────────────┐     HTTP      ┌────────────────────────────────┐
│   Frontend (HTML/    │ ◄──────────► │   FastAPI Backend (main.py)    │
│   CSS/JS)            │              │                                │
│                      │              │  ┌─── /predict ──────────────┐ │
│  - Drag-drop upload  │              │  │  model.py                 │ │
│  - Results display   │              │  │  ├─ preprocess_image()    │ │
│  - Grad-CAM view     │              │  │  ├─ predict() [ensemble]  │ │
│  - Hospital search   │              │  │  └─ _make_gradcam_overlay │ │
│                      │              │  └────────────────────────────┘ │
│                      │              │  ┌─── /recommend-hospitals ──┐ │
│                      │              │  │  hospital_recommender.py  │ │
│                      │              │  │  ├─ geocode_location()    │ │
│                      │              │  │  ├─ fetch_nearby_hospitals│ │
│                      │              │  │  └─ LLM summarization    │ │
│                      │              │  └────────────────────────────┘ │
└─────────────────────┘              └────────────────────────────────┘
                                              │
                                     ┌────────┴─────────┐
                                     │  saved_model/     │
                                     │  ├─ model_B0.keras│
                                     │  ├─ model_B3.keras│
                                     │  └─ model_B7.keras│
                                     └──────────────────┘
```

### Inference Flow (Step-by-Step)

1. **User uploads image** — The frontend provides drag-and-drop or file-browser upload. Files are validated client-side for type (JPEG, PNG, BMP, TIFF, WebP) and size (≤20 MB).
2. **Frontend sends request** — A `POST /predict?include_heatmap=true` multipart form-data request is sent to the backend.
3. **Backend validates** — `main.py` checks content type against an allowlist (`ALLOWED_TYPES`) and enforces the 20 MB limit. Empty files are rejected.
4. **Image preprocessing** — `model.py → preprocess_image()` converts raw bytes to a PIL Image, converts to RGB, resizes to 224×224 using LANCZOS interpolation, normalizes pixel values to [0, 1], and expands to a batch tensor of shape `(1, 224, 224, 3)`.
5. **Ensemble prediction** — `model.py → predict()` runs the preprocessed tensor through every loaded model (B0, B3, B7). Each model returns a malignant probability. The probabilities are averaged to produce the final ensemble score.
6. **Threshold application** — The averaged probability is compared against a configurable threshold (default 0.5, overridable via `PREDICTION_THRESHOLD` env var). Values at or above the threshold are classified as Malignant.
7. **Grad-CAM generation** (optional) — If `include_heatmap=true` and the B7 model is loaded, `_make_gradcam_overlay()` computes gradient-weighted class activation maps from the last convolutional layer, producing a red-channel heatmap overlaid on the original image. The result is returned as a base64-encoded PNG string.
8. **Response** — The backend returns a JSON response containing the prediction label, confidence score, per-class probabilities, individual model predictions, ensemble method, and optionally the Grad-CAM overlay.
9. **Frontend renders results** — `app.js → displayResults()` updates the prediction badge, animates confidence and probability bars, displays metadata, renders the Grad-CAM image, and enables the hospital recommendation input.

### Hospital Recommendation Flow

1. **User enters location** — After receiving a prediction, the user types a city or area name.
2. **Geocoding** — `hospital_recommender.py → geocode_location()` queries the Nominatim API to resolve the text to latitude, longitude, and a display name.
3. **Hospital search** — `fetch_nearby_hospitals()` queries the Overpass API for hospitals within a configurable radius (default 35 km, range 5–200 km). Results are sorted by Haversine distance and limited to 8 entries.
4. **LLM summarization** — The system builds a structured prompt with patient context and hospital details, then attempts summarization in this order:
   - OpenAI-compatible endpoint (requires `LLM_API_KEY` env var)
   - Local Ollama instance (`llama3.1:8b` by default)
   - Deterministic text template (always works, no external dependency)
5. **Response** — Returns resolved location, hospital list (name, distance, address, phone, website, emergency status), LLM model used, and summary text.

---

## 4. Dataset Explanation

### IDC Histopathology Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle: Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) |
| Total images | 277,524 image patches |
| Image size | 50×50 pixels (RGB PNG) |
| Magnification | 40× |
| Origin | Extracted from 162 whole-mount slide images of breast cancer specimens |
| Classes | **Class 0 (Benign)**: approximately 157,000 patches — **Class 1 (Malignant/IDC+)**: approximately 120,000 patches |
| Class ratio | Approximately 57% benign / 43% malignant |

### Folder Structure on Disk

The dataset is organized by patient ID, with each patient folder containing subfolders `0/` (benign) and `1/` (malignant):

```
dataset/
├── <patient_id_1>/
│   ├── 0/   ← benign patches
│   │   ├── <patient_id>_idx5_x51_y1001_class0.png
│   │   └── ...
│   └── 1/   ← malignant patches (IDC+)
│       ├── <patient_id>_idx5_x51_y1001_class1.png
│       └── ...
├── <patient_id_2>/
│   ├── 0/
│   └── 1/
└── ...
```

### How the Code Processes It

The `collect_image_paths()` function in `train.py` recursively walks the dataset directory using `Path.rglob("*.png")`. For each PNG file, it checks the parent folder name:
- Parent folder named `"0"` → appended to the benign list with label 0
- Parent folder named `"1"` → appended to the malignant list with label 1

This approach is agnostic to the patient-level folder structure and works regardless of how deeply nested the class folders are.

### Dataset Download

The `download_dataset()` function in `train.py` uses a cascading strategy:
1. Check if a user-specified path exists (via `--dataset` argument)
2. Check `backend/dataset/` and the default kagglehub cache location
3. Attempt automatic download via `kagglehub.dataset_download()`
4. Fall back to the `kaggle` CLI tool with `--unzip`
5. Exit with an error if all methods fail

---

## 5. Data Preprocessing Pipeline

### During Training (`train.py`)

#### Splitting

Data is split using scikit-learn's `train_test_split` with stratified sampling to maintain class proportions:
- **Training set**: 80% of total data
- **Validation set**: 10% of total data
- **Test set**: 10% of total data

When `--data-fraction` is less than 1.0, a stratified subsample is taken first, then split. This enables faster experimentation without skewing class distributions.

#### Image Loading and Normalization (`load_and_preprocess`)

Each image goes through this pipeline:
1. `tf.io.read_file()` — Read raw file bytes
2. `tf.image.decode_png(channels=3)` — Decode PNG to RGB tensor
3. `tf.image.resize(IMG_SIZE)` — Resize from 50×50 to 224×224 (bilinear interpolation)
4. `tf.cast(tf.float32) / 255.0` — Normalize pixel values from [0, 255] to [0.0, 1.0]

The 224×224 size matches the input resolution expected by EfficientNet architectures.

#### Data Augmentation (`augment`)

Applied **only to the training set** to improve generalization:

| Augmentation | Parameters | Purpose |
|---|---|---|
| Random horizontal flip | 50% probability | Tissue orientation is arbitrary |
| Random vertical flip | 50% probability | Same rationale |
| Random brightness | max_delta=0.25 | Simulate stain intensity variation |
| Random contrast | range [0.75, 1.30] | Simulate scanner differences |
| Random saturation | range [0.75, 1.30] | Simulate stain concentration variation |
| Random hue | max_delta=0.04 | Minor color shift tolerance |
| Random 90° rotation | 0°, 90°, 180°, or 270° | Rotational invariance |
| Clip to [0, 1] | — | Ensure valid pixel range after transforms |

#### MixUp Regularization (Optional)

When `--enable-mixup` is set, the `mixup_batch()` function interpolates between pairs of training images and their labels using a mixing coefficient sampled from a Beta(0.2, 0.2) distribution. This is implemented without the `tensorflow-probability` dependency by sampling from two Gamma distributions and dividing. MixUp reduces overconfident predictions and acts as a regularizer.

#### tf.data Pipeline

The complete data pipeline is built using `tf.data.Dataset`:
1. `from_tensor_slices()` — Create dataset from path and label arrays
2. `shuffle()` — Shuffle training data each epoch (seed=42 for reproducibility)
3. `map(load_and_preprocess)` — Parallel image loading with `AUTOTUNE` workers
4. `map(augment)` — Parallel augmentation (training only)
5. `batch()` — Batch images (configurable, default 64)
6. `map(mixup_batch)` — MixUp on batched tensors (training only, optional)
7. `prefetch(AUTOTUNE)` — Overlap preprocessing with training

### During Inference (`model.py`)

The `preprocess_image()` function handles single-image preprocessing for the web API:
1. `Image.open(io.BytesIO(image_bytes))` — Load from raw bytes using PIL
2. `.convert("RGB")` — Ensure 3-channel RGB (handles RGBA, grayscale, etc.)
3. `.resize((224, 224), Image.LANCZOS)` — High-quality downsampling
4. `np.array(dtype=np.float32) / 255.0` — Normalize to [0, 1]
5. `np.expand_dims(axis=0)` — Add batch dimension → shape `(1, 224, 224, 3)`

---

## 6. Model Architecture

### Backbone: EfficientNet

The project supports three EfficientNet variants as interchangeable backbones:

| Variant | Parameters | Characteristics | Default Batch Size |
|---|---|---|---|
| EfficientNetB0 | ~5.3M | Fastest training and inference, good for rapid iteration | 64 |
| EfficientNetB3 | ~12.2M | Balanced accuracy and speed | 32 |
| EfficientNetB7 | ~66.3M | Highest capacity, best accuracy, slowest | 16 |

All variants are loaded with ImageNet pre-trained weights (`weights="imagenet"`) and `include_top=False` to remove the original 1000-class classification head.

### Custom Classification Head

On top of the EfficientNet backbone, the code in `build_model()` adds:

```
EfficientNet Backbone (variable depth)
    ↓
GlobalAveragePooling2D()          — Reduce spatial dims to a single vector
    ↓
BatchNormalization()              — Stabilize activations
    ↓
Dense(hidden_dim, activation='swish')  — hidden_dim=512 for B7, 320 for B0/B3
    ↓
Dropout(0.35)                     — Regularization
    ↓
Dense(192, activation='swish')    — Second hidden layer
    ↓
Dropout(0.3)                      — Additional regularization
    ↓
Dense(1, activation='sigmoid', dtype='float32')  — Binary output probability
```

**Key details from the code:**
- The `swish` activation (x × sigmoid(x)) is used instead of ReLU because it is the native activation in EfficientNet and provides smoother gradients.
- The final Dense layer is explicitly set to `dtype='float32'` to ensure the output probability is full precision even when mixed-precision training (`mixed_float16`) is enabled.
- The `hidden_dim` is 512 for EfficientNetB7 (larger backbone produces richer features) and 320 for B0/B3.

### Ensemble Architecture

During inference (`model.py → predict()`), all available trained models are loaded into a dictionary keyed by variant name (`"B0"`, `"B3"`, `"B7"`). Each model independently produces a malignant probability, and the final prediction uses the **average** of all model probabilities. This ensemble approach reduces variance and improves robustness compared to any single model.

Grad-CAM is generated specifically from the B7 model (if available) because it has the deepest feature hierarchy and produces the most detailed activation maps.

---

## 7. Training Pipeline

### Overview

Training is implemented in `train.py` and uses a **3-phase progressive fine-tuning strategy** that gradually unfreezes deeper backbone layers while reducing the learning rate. This prevents catastrophic forgetting of pre-trained features while allowing the model to adapt to the medical imaging domain.

### Phase 1: Head Training (Frozen Backbone)

| Parameter | Value |
|---|---|
| Epochs | Configurable (default: 2) |
| Learning rate | 1e-3 |
| Loss | BinaryCrossentropy with label_smoothing=0.02 |
| Backbone layers | All frozen (trainable=False) |
| EarlyStopping patience | 4 epochs |

**Purpose**: Only the custom classification head is trained. The high learning rate quickly fits the new layers to the frozen backbone's feature representations. Label smoothing prevents the model from becoming overconfident on easy samples.

### Phase 2: Top-Layer Fine-Tuning

| Parameter | Value |
|---|---|
| Epochs | Configurable (default: 3) |
| Learning rate | 2e-4 |
| Loss | BinaryFocalLoss(gamma=2.0, alpha=0.35) |
| Backbone layers | Top N unfrozen (default: 40) |
| EarlyStopping patience | 5 epochs |

**Purpose**: The top 40 layers of the backbone are unfrozen and fine-tuned at a lower learning rate. Switching to focal loss at this stage helps the model focus on difficult-to-classify samples while down-weighting easy examples. The alpha parameter of 0.35 slightly favors the malignant class.

### Phase 3: Deep Fine-Tuning

| Parameter | Value |
|---|---|
| Epochs | Configurable (default: 4) |
| Learning rate | 7e-5 |
| Loss | BinaryFocalLoss(gamma=2.0, alpha=0.35) |
| Backbone layers | Top N unfrozen (default: 80) |
| Dropout rate | Reduced to 0.30 |
| EarlyStopping patience | 6 epochs |

**Purpose**: More backbone layers are unfrozen for maximum domain adaptation. The very low learning rate ensures fine-grained weight updates that do not destroy useful pre-trained features. Dropout is slightly reduced because the model benefits from more capacity at this stage.

### Focal Loss Implementation

The `BinaryFocalLoss` class in `train.py` implements:

```
Focal Loss = -alpha_t * (1 - p_t)^gamma * BCE(y, y_hat)
```

Where:
- `gamma = 2.0` — Focusing parameter. Higher values increase focus on hard samples.
- `alpha = 0.35` — Class balance factor. Values < 0.5 slightly favor the malignant class.
- `p_t` — The model's estimated probability for the true class.

When a sample is correctly classified with high confidence, `(1 - p_t)^gamma` becomes very small, effectively down-weighting the loss for easy examples. This is critical for the IDC dataset where many patches are trivially classifiable.

### Class Weighting

The `compute_class_weights()` function calculates inverse-frequency weights:

```python
w_class = total_samples / (2 * count_class)
```

For the typical IDC dataset distribution (~57% benign, ~43% malignant), this produces weights of approximately:
- Benign: ~0.88
- Malignant: ~1.16

These weights are passed to `model.fit(class_weight=...)` to ensure the loss function penalizes misclassification of the underrepresented malignant class more heavily.

### Callbacks

Each training phase uses the same set of callbacks configured by `build_callbacks()`:

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | monitor='val_auc', mode='max', restore_best_weights=True | Stop training when validation AUC stops improving |
| `ReduceLROnPlateau` | monitor='val_loss', factor=0.3, min_lr=1e-7 | Reduce learning rate by 70% when loss plateaus |
| `ModelCheckpoint` | monitor='val_auc', save_best_only=True, mode='max' | Save only the best model by validation AUC |
| `TensorBoard` | log_dir='logs/{phase_name}' | Training visualization in TensorBoard |
| `CSVLogger` | file='logs/{phase_name}_history.csv' | Training history as CSV for post-analysis |

### Runtime Configuration

The `configure_runtime()` function at training startup:
1. Detects available GPUs via `tf.config.list_physical_devices("GPU")`
2. Enables memory growth to prevent TensorFlow from allocating all GPU memory
3. Enables mixed-precision training (`mixed_float16`) when a GPU is available, which uses 16-bit floats for most operations while keeping 32-bit for numerically sensitive operations

### Training Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | (auto-detect) | Path to extracted IDC dataset root |
| `--backbone` | EfficientNetB0 | Backbone architecture (B0, B3, or B7) |
| `--data-fraction` | 0.35 | Fraction of dataset to use (0.1–1.0) |
| `--batch-size` | 64 | Training batch size |
| `--epochs-head` | 2 | Phase 1 epochs |
| `--epochs-finetune-1` | 3 | Phase 2 epochs |
| `--epochs-finetune-2` | 4 | Phase 3 epochs |
| `--unfreeze-phase2` | 40 | Layers to unfreeze in Phase 2 |
| `--unfreeze-phase3` | 80 | Layers to unfreeze in Phase 3 |
| `--enable-mixup` | False | Enable MixUp augmentation |

### Reproducibility

The training script sets seeds for reproducibility:
- `tf.random.set_seed(42)`
- `np.random.seed(42)`
- `random.seed(42)`

Data shuffling and splitting also use `random_state=42` / `seed=42`.

---

## 8. Evaluation and Metrics

### Threshold Optimization

After training, `evaluate_best_threshold()` finds the optimal classification threshold by:

1. Running all validation images through the best saved model
2. Computing the precision-recall curve using scikit-learn
3. Calculating F1 score at each threshold: `F1 = 2 × precision × recall / (precision + recall)`
4. Selecting the threshold that maximizes F1

This threshold (often ≠ 0.5) is then used for test set evaluation and can be set as the `PREDICTION_THRESHOLD` environment variable for inference.

### Test Set Evaluation

The `evaluate_test_set()` function produces:

| Metric | Description | Why It Matters |
|---|---|---|
| **Accuracy** | Fraction of correctly classified patches | Overall performance indicator |
| **Precision (per class)** | True positives / (True positives + False positives) | For malignant class: measures false alarm rate |
| **Recall (per class)** | True positives / (True positives + False negatives) | For malignant class: measures missed cancer rate |
| **F1 Score** | Harmonic mean of precision and recall | Balanced metric that penalizes both false positives and false negatives |
| **AUC** | Area under the ROC curve (computed during training) | Measures discrimination ability across all thresholds |
| **Confusion Matrix** | 2×2 table of actual vs predicted classes | Visual summary of error patterns |

### Training-Time Metrics

The model is compiled with these real-time metrics tracked during each epoch:
- `accuracy` — Standard classification accuracy
- `auc` — Area Under the ROC Curve
- `precision` — Precision for the positive (malignant) class
- `recall` — Recall for the positive (malignant) class

### Interpreting Results

- **High precision, low recall for malignant** → Model is conservative, misses some cancers but rarely gives false alarms
- **Low precision, high recall for malignant** → Model flags most cancers but also has more false positives
- **F1 score** balances these two concerns; the threshold optimization step finds the best trade-off

### Generated Evaluation Artifacts

- `saved_model/metrics_report.json` — JSON file with all computed metrics and the selected threshold
- `saved_model/confusion_matrix.png` — Seaborn heatmap visualization (160 DPI) with actual vs predicted labels

---

## 9. Project Directory Structure

```
Breast Cancer Detection/
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── model.py                   # Model loading, preprocessing, inference, Grad-CAM
│   ├── train.py                   # Complete training pipeline with 3-phase fine-tuning
│   ├── hospital_recommender.py    # Geocoding, hospital search, LLM summarization
│   ├── requirements.txt           # Python dependencies with version constraints
│   └── saved_model/               # Generated after training (gitignored)
│       ├── model_B0.keras         # Trained EfficientNetB0 weights
│       ├── model_B3.keras         # Trained EfficientNetB3 weights
│       ├── model_B7.keras         # Trained EfficientNetB7 weights
│       ├── breast_cancer_model.keras  # Best checkpoint from latest training run
│       ├── metrics_report.json    # Test set evaluation metrics
│       └── confusion_matrix.png   # Confusion matrix visualization
├── frontend/
│   ├── index.html                 # Main UI with upload zone, results panel, info sections
│   ├── styles.css                 # Complete design system (glassmorphism, animations, responsive)
│   └── app.js                     # Client-side logic (file handling, API calls, rendering)
├── run.sh                         # Shell script to activate venv and start the server
├── .gitignore                     # Excludes venv, __pycache__, saved_model, logs, .env
└── README.md                      # This documentation
```

### File Responsibilities

#### `backend/main.py`
The FastAPI application that ties everything together. It:
- Loads models on startup via the `lifespan` context manager
- Defines five routes: `/` (frontend), `/health`, `/model-info`, `/predict`, `/recommend-hospitals`
- Validates uploaded files (type allowlist, size limit, empty file check)
- Configures CORS (all origins allowed for development)
- Mounts the `frontend/` directory as static files at `/static`
- Loads environment variables from a `.env` file via `python-dotenv`

#### `backend/model.py`
Handles all model-related operations:
- `load_model()` — Loads B0, B3, B7 models from `saved_model/` into a global cache
- `preprocess_image()` — Converts raw image bytes to a normalized tensor
- `predict()` — Runs ensemble inference and optionally generates Grad-CAM
- `_make_gradcam_overlay()` — Computes Grad-CAM using TensorFlow GradientTape
- `_find_last_conv_layer_name()` — Locates the deepest Conv2D layer, searching nested models
- `get_model_info()` — Returns metadata about loaded models (architecture, parameters, input shape)

#### `backend/train.py`
The complete training script (448 lines), containing:
- `BinaryFocalLoss` — Custom focal loss implementation
- `configure_runtime()` — GPU detection and mixed-precision setup
- `download_dataset()` — Dataset acquisition with multiple fallback methods
- `collect_image_paths()` — Recursive image discovery with label extraction
- `load_and_preprocess()`, `augment()`, `mixup_batch()` — Data pipeline functions
- `build_model()` — Model construction with configurable backbone and unfreezing
- `compute_class_weights()` — Inverse-frequency class weighting
- `compile_model()` — Optimizer and loss configuration
- `build_callbacks()` — Training callback suite
- `evaluate_best_threshold()` — F1-optimal threshold search
- `evaluate_test_set()` — Comprehensive test evaluation with visualization
- `train()` — Main orchestrator for the 3-phase training pipeline
- `parse_args()` — CLI argument parsing

#### `backend/hospital_recommender.py`
Self-contained module for location-based hospital recommendations:
- `geocode_location()` — Free-text to coordinates via Nominatim
- `fetch_nearby_hospitals()` — Overpass API query with Haversine distance calculation
- `summarize_with_openai_compatible()` — OpenAI API call with configurable endpoint
- `summarize_with_ollama()` — Local Ollama model call
- `fallback_summary()` — Deterministic template-based summary
- `build_recommendation_prompt()` — Structured prompt construction
- `get_hospital_recommendations()` — Main entry point with cascading LLM fallback

#### `frontend/index.html`
The single-page application structure:
- Animated background orbs for visual depth
- Navigation bar with brand logo (SVG) and model status badge (live health check)
- Hero section with model statistics cards
- Upload panel with drag-and-drop zone and image preview
- Results panel with four states: empty, loading (animated), results, error
- "How It Works" section (4-step visual guide)
- "About the Model" section (architecture, dataset, training, performance)
- Footer with clinical disclaimer

#### `frontend/app.js`
Client-side application logic (377 lines):
- File validation (type and size) before upload
- FileReader-based image preview
- Fetch API calls to `/predict` and `/recommend-hospitals` with timeout handling
- Animated progress bars with staggered delays
- Toast notification system (info, ok, error types with auto-dismiss)
- Model health status polling (every 30 seconds when offline)
- Keyboard accessibility for drag-and-drop zone

#### `frontend/styles.css`
Complete design system including:
- CSS custom properties (design tokens) for colors, spacing, typography
- Glassmorphism effect (semi-transparent backgrounds with backdrop blur)
- Responsive layout with CSS Grid (single column below 860px)
- Animated background orbs using CSS keyframes
- Button variants (primary, outline, ghost)
- Progress bar animations with CSS transitions
- Prediction badge styles (green for benign, red for malignant)
- Hospital card styles
- Loading animation with scanning bar and pulse rings
- Google Fonts: Inter (body) and JetBrains Mono (code/values)

#### `run.sh`
Startup helper script that:
1. Detects and activates the virtual environment
2. Warns if no trained model is found (but still starts the server)
3. Launches Uvicorn with hot-reload on port 8000

---

## 10. How to Run the Project

### Prerequisites

- **Python 3.11** (recommended; TensorFlow compatibility)
- **pip** (Python package manager)
- **Kaggle credentials** (for automated dataset download) or manual dataset download
- macOS, Linux, or Windows (GPU optional but recommended for training)

### Step 1: Clone and Set Up Environment

```bash
git clone <your-repository-url>
cd "Breast Cancer Detection"
python3.11 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Step 2: Prepare the Dataset

**Option A — Automatic download (Kaggle API):**

```bash
mkdir -p ~/.kaggle
# Copy your kaggle.json API token to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The training script will automatically download the dataset on first run.

**Option B — Manual download:**

1. Download from: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
2. Extract into `backend/dataset/`
3. Verify that the extracted folder contains patient subfolders with `0/` and `1/` subfolders

### Step 3: Train the Model

Choose a training profile based on available compute:

**Fast profile** (~30 minutes on GPU, suitable for testing):
```bash
cd backend
python train.py --backbone EfficientNetB0 --data-fraction 0.35 --batch-size 64 --epochs-head 2 --epochs-finetune-1 3 --epochs-finetune-2 4
```

**Balanced profile** (~2 hours on GPU):
```bash
cd backend
python train.py --backbone EfficientNetB3 --data-fraction 0.6 --batch-size 32 --epochs-head 3 --epochs-finetune-1 5 --epochs-finetune-2 6
```

**Full profile** (~8+ hours on GPU, maximum accuracy):
```bash
cd backend
python train.py --backbone EfficientNetB7 --data-fraction 1.0 --batch-size 16 --epochs-head 4 --epochs-finetune-1 8 --epochs-finetune-2 10 --enable-mixup
```

### Step 4: Start the Server

**Option A — Direct start:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option B — Helper script:**
```bash
./run.sh
```

### Step 5: Open the Application

- **Web UI**: http://localhost:8000
- **API docs**: http://localhost:8000/docs (interactive Swagger UI)
- **Health check**: http://localhost:8000/health

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PREDICTION_THRESHOLD` | 0.5 | Classification threshold for malignant prediction |
| `LLM_API_KEY` | (none) | API key for OpenAI-compatible LLM endpoint |
| `LLM_API_BASE` | https://api.openai.com/v1 | Base URL for the LLM API |
| `LLM_MODEL` | gpt-4o-mini | Model name for the LLM API |
| `OLLAMA_MODEL` | llama3.1:8b | Local Ollama model name (fallback) |

Create a `.env` file in the `backend/` directory to set these values.

---

## 11. Expected Outputs

### Training Outputs

| File | Location | Description |
|---|---|---|
| `breast_cancer_model.keras` | `backend/saved_model/` | Best model checkpoint (by validation AUC) from the latest training phase |
| `metrics_report.json` | `backend/saved_model/` | JSON with threshold, F1, accuracy, per-class precision/recall, and support count |
| `confusion_matrix.png` | `backend/saved_model/` | Seaborn heatmap of the test set confusion matrix |
| Phase logs | `backend/logs/` | TensorBoard event files and CSV training histories for each phase |

**Example `metrics_report.json`:**
```json
{
  "threshold": 0.4823,
  "f1": 0.9512,
  "accuracy": 0.9567,
  "benign_precision": 0.9621,
  "benign_recall": 0.9587,
  "malignant_precision": 0.9483,
  "malignant_recall": 0.9542,
  "support": 27760
}
```

### Inference Outputs

**`POST /predict` response:**
```json
{
  "prediction": "Malignant",
  "confidence": 0.9234,
  "idc_positive_prob": 0.9234,
  "idc_negative_prob": 0.0766,
  "is_malignant": true,
  "model_predictions": {
    "B0": 0.9156,
    "B3": 0.9312,
    "B7": 0.9234
  },
  "ensemble_method": "average_probability",
  "img_size_used": "224x224",
  "threshold_used": 0.5,
  "filename": "sample_patch.png",
  "gradcam_overlay_base64": "iVBORw0KGgo..."
}
```

**`POST /recommend-hospitals` response:**
```json
{
  "location": {
    "query": "Hyderabad, Telangana",
    "resolved_name": "Hyderabad, Telangana, India",
    "lat": 17.385,
    "lon": 78.4867
  },
  "diagnosis": "Malignant",
  "llm_model_used": "gpt-4o-mini",
  "summary": "Based on your location, prioritize contacting Apollo Hospitals...",
  "hospitals": [
    {
      "name": "Apollo Hospitals",
      "distance_km": 3.45,
      "address": "Jubilee Hills, Hyderabad",
      "phone": "+91-40-2320-8888",
      "website": "https://apollohospitals.com",
      "emergency": "yes"
    }
  ]
}
```

### Expected Performance Range

| Metric | Typical Range |
|---|---|
| Test Accuracy | 95–97% |
| Test AUC | ~0.98 |
| Test F1 | 0.95–0.97 |
| Inference Time | <1 second per image |

Actual results depend on the backbone chosen, data fraction, and training duration.

---

## 12. How Predictions Work

### Inference Pipeline (Detailed)

When a user uploads an image through the web interface, the following sequence executes:

**1. Client-Side Validation (`app.js`)**
- `isValidImageType()` checks MIME type against `['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']`
- File size is checked against the 20 MB limit
- A `FileReader` generates a preview thumbnail

**2. API Request**
- `analyze()` builds a `FormData` object with the file
- Sends `POST /predict?include_heatmap=true` with a 30-second timeout

**3. Server-Side Validation (`main.py`)**
- Content type is checked against `ALLOWED_TYPES`
- File bytes are read and size is verified
- Empty files are rejected

**4. Preprocessing (`model.py → preprocess_image()`)**
- Raw bytes → PIL Image → RGB conversion → 224×224 resize (LANCZOS) → float32 normalization to [0,1] → batch dimension added

**5. Model Ensemble (`model.py → predict()`)**
- Each loaded model (B0, B3, B7) runs `model.predict()` on the preprocessed tensor
- Individual probabilities are stored in a dictionary
- The average probability is computed: `avg_prob = sum(probs) / len(probs)`

**6. Classification**
- If `avg_prob >= threshold`: label = "Malignant", confidence = avg_prob
- If `avg_prob < threshold`: label = "Benign", confidence = 1 - avg_prob

**7. Grad-CAM Generation (`model.py → _make_gradcam_overlay()`)**

This is the most computationally interesting step:

1. `_find_last_conv_layer_name()` traverses model layers in reverse order to find the deepest `Conv2D` layer. It also searches inside nested models (the EfficientNet backbone is a nested `tf.keras.Model`).
2. A `tf.keras.models.Model` is created with the original input and two outputs: the last conv layer's activations and the final prediction.
3. Inside a `tf.GradientTape` context, a forward pass produces both the convolutional feature maps and the prediction.
4. Gradients of the prediction with respect to the feature maps are computed.
5. Gradients are globally average-pooled across spatial dimensions to get importance weights per channel.
6. Feature maps are weighted by these importance values and summed, then ReLU'd (only positive contributions).
7. The heatmap is normalized to [0, 1] and resized to 224×224.
8. The heatmap is overlaid onto the original image as a red-channel heat overlay with alpha blending (α=0.35).
9. The result is encoded as a PNG and base64-encoded for JSON transport.

**8. Response Assembly**
- The JSON response includes: prediction label, confidence, per-class probabilities, individual model predictions, ensemble method, image size, threshold, filename, and optionally the Grad-CAM base64 string.

**9. Frontend Rendering (`app.js → displayResults()`)**
- The prediction badge is colored (green for benign, red for malignant)
- Confidence bar width is animated from 0% to the confidence percentage
- Probability bars are animated with staggered delays (400ms for benign, 550ms for malignant)
- Grad-CAM image is rendered if available
- Hospital recommendation input is enabled

### Interpreting Predictions

| Output Field | Interpretation |
|---|---|
| `prediction` | "Benign" or "Malignant" — the final classification |
| `confidence` | How confident the model is in its prediction (0.0–1.0) |
| `idc_positive_prob` | Probability that the tissue contains IDC (malignant) |
| `idc_negative_prob` | Probability that the tissue is benign (1 - idc_positive_prob) |
| `model_predictions` | Individual model probabilities — useful for assessing agreement |
| `gradcam_overlay_base64` | Red regions = high importance for the prediction |

**Grad-CAM interpretation**: Red-highlighted tissue regions are where the model "looked" when making its prediction. These regions may correspond to morphological features associated with IDC such as dense nuclear clusters, irregular gland structures, or stromal invasion patterns. However, Grad-CAM is a model interpretation aid, not a pathological boundary annotation.

---

## 13. Design Decisions

### Why EfficientNet over Other Architectures

EfficientNet uses **compound scaling** to jointly optimize network depth, width, and resolution. Compared to alternatives:
- **vs. ResNet**: EfficientNet achieves higher accuracy with 8.4× fewer parameters (B0 vs ResNet-50). For medical imaging where training data is limited, parameter efficiency reduces overfitting risk.
- **vs. VGG**: VGG-16 has 138M parameters versus EfficientNetB0's 5.3M — a 26× difference. The smaller model is faster to train and deploy.
- **vs. InceptionV3**: EfficientNet provides a family of scaled models (B0–B7), allowing users to trade accuracy for speed without changing the architecture.

### Why 3-Phase Progressive Fine-Tuning

Fine-tuning all layers simultaneously with a single learning rate often leads to:
1. **Catastrophic forgetting** — Pre-trained features in early layers get destroyed
2. **Unstable training** — Deep layers need smaller updates than the new classification head

The 3-phase approach addresses both issues:
- Phase 1 fits the head to frozen features (safe, fast convergence)
- Phase 2 adapts upper backbone layers (domain-specific features)
- Phase 3 reaches deeper layers (fine-grained adaptation without destroying low-level features)

### Why Focal Loss Instead of Standard BCE

The IDC dataset has a ~57/43 benign/malignant split. While not extreme, standard BCE treats all samples equally. Focal loss with gamma=2.0 down-weights easy samples exponentially, forcing the model to focus on ambiguous patches near the decision boundary. Combined with class weighting, this produces better calibrated predictions.

### Why Ensemble Averaging

Individual EfficientNet variants have different inductive biases due to their varying depth and width. Averaging their probabilities:
- Reduces prediction variance
- Provides more calibrated confidence scores
- Makes the system more robust to edge cases where one model might fail

### Why PIL + NumPy for Inference (Not tf.data)

During inference, only a single image needs to be processed. Using PIL and NumPy avoids the overhead of constructing a `tf.data` pipeline for one sample. PIL's LANCZOS resampling also provides higher-quality downscaling than TensorFlow's default bilinear interpolation.

### Why Deterministic LLM Fallback

The hospital recommendation system is designed to work in all environments:
- **With API key**: Uses OpenAI-compatible endpoint for natural, context-aware summaries
- **With Ollama**: Uses a locally running model for privacy-sensitive deployments
- **Without either**: Falls back to a deterministic template that still provides actionable hospital guidance

This cascading design ensures the feature never fails completely.

### Why Vanilla JavaScript (No Framework)

The frontend uses no frameworks (React, Vue, etc.) because:
1. The UI has a single page with limited state management needs
2. Eliminates build tooling requirements (no Webpack, Vite, etc.)
3. Reduces the attack surface and dependency chain
4. The application loads faster with no JavaScript bundle to parse

---

## 14. Limitations and Future Improvements

### Current Limitations

1. **Not a medical device** — This tool is not FDA approved and must not be used for clinical diagnosis without pathologist validation.
2. **Domain shift** — The model is trained on a single dataset. Performance may degrade on tissue samples prepared with different staining protocols, scanners, or magnification levels.
3. **Patch-level only** — The system classifies individual 50×50 patches. It does not provide whole-slide-level diagnosis or tumor localization.
4. **Binary classification** — The model only distinguishes IDC vs. non-IDC. It does not identify other breast cancer subtypes (DCIS, lobular carcinoma, etc.).
5. **Grad-CAM limitations** — Heatmaps show correlation, not causation. They highlight where the model looks, not necessarily the true pathological features.
6. **LLM dependency** — The hospital recommendation quality depends on the LLM used. The deterministic fallback provides basic but less personalized guidance.
7. **No authentication** — The API has no authentication or rate limiting, and CORS allows all origins. This is suitable for development/research but not production deployment.

### Potential Future Improvements

1. **Multi-class classification** — Extend the model to detect additional cancer subtypes beyond IDC.
2. **Whole-slide image support** — Implement a sliding-window approach to classify entire slide images and generate spatial heatmaps of predicted malignancy.
3. **Model calibration** — Apply temperature scaling or Platt scaling to improve probability calibration beyond the F1-optimized threshold.
4. **Uncertainty estimation** — Add Monte Carlo dropout or deep ensemble uncertainty quantification to flag low-confidence predictions for human review.
5. **DICOM support** — Accept medical imaging standard DICOM files in addition to standard image formats.
6. **Federated learning** — Enable training across institutions without sharing patient data.
7. **User authentication** — Add API key authentication and rate limiting for production use.
8. **Docker deployment** — Containerize the application for consistent, reproducible deployment.
9. **Test suite** — Add automated unit tests and integration tests for the training pipeline, inference API, and frontend logic.
10. **Stain normalization** — Implement preprocessing-stage stain normalization (e.g., Macenko or Reinhard methods) to improve generalization across different laboratories.

---

## API Reference

### `GET /health`

Returns server health status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0"
}
```

### `GET /model-info`

Returns metadata about loaded models including architecture names, input shapes, and parameter counts.

**Response:**
```json
{
  "status": "loaded",
  "models_loaded": ["B0", "B3", "B7"],
  "model_details": {
    "B0": {
      "architecture": "BreastCancerDetector_EfficientNetB0",
      "input_shape": [null, 224, 224, 3],
      "total_params": 5521345
    }
  },
  "dataset": "IDC Histopathology (Kaggle)",
  "classes": ["Benign", "Malignant"]
}
```

### `POST /predict`

Classify a histopathology image as benign or malignant.

**Request:** `multipart/form-data` with `file` field. Optional query parameter `include_heatmap=true`.

**Constraints:** JPEG/PNG/BMP/TIFF/WebP only, ≤20 MB.

**Response:** See [Expected Outputs](#11-expected-outputs) section for response format.

### `POST /recommend-hospitals`

Get nearby hospital recommendations based on location and diagnosis.

**Request body:**
```json
{
  "location": "City or area name (2–120 characters)",
  "diagnosis": "Benign or Malignant (optional, default: Unknown)",
  "radius_km": 35
}
```

**Constraints:** `radius_km` must be between 5 and 200.

**Response:** See [Expected Outputs](#11-expected-outputs) section for response format.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| TensorFlow installation fails | Use Python 3.11 environment. On macOS Apple Silicon, ensure `tensorflow-metal` is installed. |
| Training is too slow | Use `--backbone EfficientNetB0`, increase `--batch-size`, reduce `--data-fraction` |
| "No trained models found" at inference | Run `train.py` first, or verify that `.keras` files exist in `backend/saved_model/` |
| Hospital recommendations fail | LLM features require `LLM_API_KEY` env var or local Ollama. The deterministic fallback always works. |
| Frontend not loading | Ensure the server is started from the `backend/` directory so the relative path to `frontend/` resolves correctly |
| GPU not detected | Verify CUDA/cuDNN installation. Check `tf.config.list_physical_devices("GPU")` in a Python shell. |

---

## Disclaimer

This project is for **research and educational use only**. It is not FDA approved and must not be used as a standalone clinical diagnosis system. Always consult qualified medical professionals for clinical decisions. AI predictions should supplement, never replace, expert pathologist review.
