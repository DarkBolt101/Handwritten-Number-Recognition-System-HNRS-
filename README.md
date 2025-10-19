## Digit Recognition System
Simple, modular digit recognition with a Tkinter GUI, OpenCV image processing, MNIST-style chip creation, and three model backends (CNN, SVM, Random Forest).


### Quick Start

Prerequisites
- Python 3.9+
- pip


### Install and Setup (from scratch)

Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

Create and activate a virtual environment (recommended)
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies
```bash
pip install --upgrade pip
pip install numpy opencv-python pillow tensorflow scikit-learn joblib matplotlib scikit-image pandas
```

Prepare MNIST CSVs
```bash
python task1_preprocessing.py normalize --train_out mnist_train_normalized.csv --test_out mnist_test_normalized.csv
```

Train models
```bash
# All models
python task3_training.py all

# Or train individually
python task3_training.py cnn --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_cnn_model.keras
python task3_training.py rf  --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_rf_model.joblib
python task3_training.py svm --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_svm_model.joblib
```

Test models (optional)
```bash
python task3_testing.py all
```

Run the GUI
```bash
python main_gui.py
```


### Repository Map

- `main_gui.py`: Tkinter app. Drawing, file open, pipeline run, display.
- `segmentation.py`: Segmentation methods. Produces binary mask.
- `splitting.py`: Splitting methods for touching digits.
- `digit_extraction.py`: Contours, optional splitting, MNIST chip creation.
- `models.py`: Loads CNN, SVM, RF. Runs predictions.
- `task1_preprocessing.py`: Build CSVs from MNIST.
- `task3_training.py`: Train CNN, SVM, RF on CSVs.
- `task3_testing.py`: Test trained models on CSVs or simple folders.

Saved models (expected at repo root)
- `mnist_cnn_model.keras`
- `mnist_svm_model.joblib`, `mnist_svm_scaler.joblib`
- `mnist_rf_model.joblib`


### GUI Flow

1) Input
- Draw on canvas or open an image. The app converts to grayscale.

2) Segmentation (`segmentation.py`)
- Create a binary mask with white foreground on black background.
- Methods: `auto` (Otsu), `otsu`, `adaptive`, `kmeans`, `local_threshold`, `canny_edges`.
- The last method used is tracked for display.

3) Contours and optional splitting (`digit_extraction.py`, `splitting.py`)
- Find external contours from the mask.
- If a contour is wide (aspect ratio ≥ 1.25) try splitting using the selected method: `simple`, `projection`, `kmeans1d`, `skeleton`, or `auto`.
- Track natural separation, successful splitting, or none.

4) MNIST chip creation (`digit_extraction.py`)
- For each region, build a 28×28 chip:
  - Optional grayscale base, else binary mask
  - Pad to square, resize to 20×20, pad to 28×28
  - Center using moments
  - Ensure white digit on black background

5) Prediction (`models.py`)
- CNN, SVM, RF each predict digits for the chips.
- `auto` picks the model with the best average confidence among those that returned predictions.
- The GUI shows all model outputs and which model was selected.


### Using the GUI (`main_gui.py`)

Controls
- Open Image: choose an image file
- Process: run the pipeline
- Clear: reset canvas and state

Settings
- Model: `auto`, `cnn`, `svm`, `rf`
- Segmentation: `auto`, `otsu`, `adaptive`, `kmeans`, `local_threshold`, `canny_edges`
- Split Method: `auto`, `simple`, `projection`, `kmeans1d`, `skeleton`
- Confidence Threshold: informational
- Foreground Threshold: informational
- Advanced: Use Grayscale Chips, Pre-split Erosion

Tabs
- Processing Pipeline: original, grayscale, segmentation, detected boxes, chips grid
- Results & Analysis: predictions and confidences for each model
- Statistics: counters, current settings, system status

Notes
- The drawing buffer is 600×500 to match layout, so the bottom of the canvas is captured.
- Drawn input is inverted as needed for display and processing.


### Segmentation (`segmentation.py`)

API
- `segment_image_combined(gray_image, method="auto") -> mask`

Methods
- Otsu: Gaussian blur, Otsu threshold
- Adaptive: Gaussian blur, adaptive Gaussian threshold
- K-means: 2-cluster k-means on intensities
- Local Threshold: adaptive mean with a larger window
- Canny Edges: edges, close, dilate

Tracking
- The module records the last method used. The GUI reads it.


### Splitting (`splitting.py`)

API
- `split_touching_digits(mask, contour, split_method="simple") -> List[contour]`
- `get_last_splitting_method() -> str`

Policy
- Attempt splitting only if aspect ratio ≥ 1.25.

Methods
- `simple`: split at middle, validate both parts have enough pixels
- `projection`: find valleys in smoothed vertical projection
- `kmeans1d`: 1D k-means on x coordinates of foreground pixels
- `skeleton`: skeletonize, split at a valley in the skeleton projection
- `auto`: try simple first, then advanced methods

Tracking
- Records successful methods or failure reasons. Natural separation is tracked in `digit_extraction.py`.


### Digit Extraction (`digit_extraction.py`)

API
- `extract_digits_combined(mask, gray_image, segmentation_method, split_method, use_grayscale_chips, enable_pre_erosion, one_digit_only) -> (chips, vis)`
- `get_last_segmentation_method() -> str`
- `get_last_splitting_method() -> str`

Steps
- Optional erosion
- Find external contours
- Split only wide contours (aspect ratio ≥ 1.25)
- Track natural separation, splitting method, or none
- Create 28×28 chips and a visualization image with green boxes


### Models (`models.py`)

API
- `ModelManager`
  - Loads `mnist_cnn_model.keras`, `mnist_svm_model.joblib` + `mnist_svm_scaler.joblib`, `mnist_rf_model.joblib`
  - `predict_all_models(chips) -> (all_predictions, all_confidences)`
  - `predict_with_model(chips, model_name="auto") -> (predictions, all_predictions, all_confidences)`

Confidence
- CNN: max softmax per chip, averaged
- SVM: `predict_proba` if available, else sigmoid of decision scores
- RF: `predict_proba`, max per chip averaged


### Preprocessing (`task1_preprocessing.py`)

Build CSVs from MNIST.
```bash
# Normalized MNIST to CSV
python task1_preprocessing.py normalize --train_out mnist_train_normalized.csv --test_out mnist_test_normalized.csv --show_n 0

# Binarized MNIST to CSV
python task1_preprocessing.py binarize --train_out mnist_train_binarized.csv --test_out mnist_test_binarized.csv --threshold 128 --show_n 0
```


### Training (`task3_training.py`)

Prepare CSVs first with Task 1. Then train models.
```bash
# All
python task3_training.py all

# CNN
python task3_training.py cnn --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_cnn_model.keras

# Random Forest
python task3_training.py rf --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_rf_model.joblib

# SVM
python task3_training.py svm --train_csv mnist_train_normalized.csv --test_csv mnist_test_normalized.csv --out mnist_svm_model.joblib
```

### Testing (`task3_testing.py`)

Evaluate on test CSVs.
```bash
# All
python task3_testing.py all

# CNN
python task3_testing.py cnn --test_csv mnist_test_normalized.csv

# RF
python task3_testing.py rf --test_csv mnist_test_normalized.csv

# SVM
python task3_testing.py svm --test_csv mnist_test_normalized.csv
```

The script prints accuracy and a confusion matrix. Folder-based helpers are included for quick checks on simple image folders.


### Design Choices

- Separate modules for GUI, segmentation, splitting, extraction, and models.
- Direct tracking of last segmentation and splitting methods for display.
- Split only when aspect ratio ≥ 1.25 to avoid false splits.


