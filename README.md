# 🌸 IrisAI • Botanical Species Classification & Machine Learning Dashboard

An end-to-end, production-grade Machine Learning system and modern web application that classifies iris flowers into three distinct species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) based on morphometric dimensions.

Featuring the bioluminescent glassmorphic UI design system from **[Stitch Project 14704649354205307970](https://stitch.withgoogle.com/projects/14704649354205307970)** ("IrisAI Classification Dashboard").

---

## 📌 Table of Contents
1. [Overview & Objectives](#-overview--objectives)
2. [Stitch UI Design System](#-stitch-ui-design-system)
3. [Dataset Overview](#-dataset-overview)
4. [Architecture & Featurization Pipeline](#-architecture--featurization-pipeline)
5. [Model Comparison & Benchmark Results](#-model-comparison--benchmark-results)
6. [Project Structure](#-project-structure)
7. [Installation & Getting Started](#-installation--getting-started)
8. [Usage & Interfaces](#-usage--interfaces)
   - [1. Modern Web Application (`server.py`)](#1-modern-web-application-serverpy)
   - [2. Running the Training Pipeline](#2-train-all-models--generate-reports)
   - [3. CLI Real-Time Prediction](#3-cli-inference)
   - [4. Optional Streamlit Dashboard](#4-optional-streamlit-dashboard)
9. [REST API Documentation](#-rest-api-documentation)
10. [Visual Artifacts Gallery](#-visual-artifacts-gallery)

---

## 🎯 Overview & Objectives
- **Target Goal:** Accurate multi-class classification of Iris specimens from 4 numerical morphological features.
- **Classes:** 
  - `0`: *Iris Setosa* (linearly separable, small petals)
  - `1`: *Iris Versicolor* (intermediate size, slight boundary overlap)
  - `2`: *Iris Virginica* (largest petals, slight boundary overlap)
- **Engineering Principles:**
  - **Zero Data Leakage:** Preprocessing pipelines (`StandardScaler`) fit strictly on training splits.
  - **Stratified Splitting:** Preserves exact 1:1:1 class proportions across train and holdout test sets.
  - **Systematic Model Zoo:** Compares Linear, Instance-Based, Tree-Based, Bagging Ensemble, and Maximum-Margin Kernel models.
  - **Diagnostic Rigor:** Evaluated with Accuracy, Macro/Weighted F1, multi-class Confusion Matrices, One-vs-Rest ROC curves, and 2D Decision Boundaries.

---

## 📊 Dataset Overview
- **Source:** Ronald Fisher's classic Iris biological dataset (1936) / UCI Machine Learning Repository.
- **Total Samples:** 150 (50 per species - perfectly balanced).
- **Features:**
  1. `sepal_length`: Sepal length in cm ($4.3 - 7.9$)
  2. `sepal_width`: Sepal width in cm ($2.0 - 4.4$)
  3. `petal_length`: Petal length in cm ($1.0 - 6.9$)
  4. `petal_width`: Petal width in cm ($0.1 - 2.5$)
- **Data Quality:** Zero missing values, zero physical measurement anomalies.

---

## ⚙️ Architecture & Featurization Pipeline

```
┌────────────────────────┐
│   Raw Iris Dataset     │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Schema & Data Checks  │ (Zero nulls, positive values, 150 rows)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ Stratified Split (80%) │ ────► Holdout Test Split (20%, 30 samples)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  StandardScaler (Fit)  │ (Strict featurization on train fold only)
└───────────┬────────────┘
            ▼
┌──────────────────────────────────────────────────────────┐
│             5-Fold Stratified Cross-Validation           │
│  • Logistic Regression      • K-Nearest Neighbors (k=11) │
│  • Decision Tree (depth=3)  • Random Forest (100 trees)  │
│  • Support Vector Machine (RBF Kernel)                   │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│              Production Artifacts & Serving              │
│  • Serialized Pipeline (.joblib)                         │
│  • Interactive Streamlit Dashboard (app.py)              │
│  • CLI Predictor (src/predict.py)                        │
└──────────────────────────────────────────────────────────┘
```

---

## 🏆 Model Comparison & Benchmark Results

Evaluated across identical 5-fold stratified cross-validation folds and a 30-sample holdout test set:

| Model | CV Accuracy | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **96.7% ± 1.7%** | **96.7%** | **0.970** | **0.967** | **0.967** | **0.997** |
| **Decision Tree ($max\_depth=3$)** | 95.8% ± 2.6% | 96.7% | 0.970 | 0.967 | 0.967 | 0.972 |
| **Random Forest (100 trees)** | 95.0% ± 3.1% | 96.7% | 0.970 | 0.967 | 0.967 | 0.993 |
| **K-Nearest Neighbors ($k=3$)** | 96.7% ± 1.7% | 93.3% | 0.944 | 0.933 | 0.933 | 0.993 |
| **Logistic Regression** | 95.8% ± 2.6% | 93.3% | 0.933 | 0.933 | 0.933 | 0.997 |

---

## 🔍 Key Machine Learning Findings

1. **Petals are the Primary Discriminators:** Feature importance analysis reveals that `petal_length` (~55%) and `petal_width` (~42%) provide over **97%** of the total information gain. Sepal measurements contribute minimal marginal gain.
2. **Linear Separability of Setosa:** *Iris Setosa* is completely separable in 2D petal space with a large margin. No model makes a single error on Setosa.
3. **Smooth Kernel Boundaries Excel:** Because *Versicolor* and *Virginica* have a mild overlapping boundary, SVM (RBF) and KNN produce smooth boundaries that adapt naturally to specimen density without the rigid orthogonal staircase cuts of single decision trees.

---

## 🎨 Stitch UI Design System

This project features a complete UI redesign built to match the state-of-the-art **[Stitch Project 14704649354205307970](https://stitch.withgoogle.com/projects/14704649354205307970)** ("IrisAI Classification Dashboard"):

- **Theme & Elevation:** Bioluminescent deep charcoal (`#0F131C` canvas) with layered glassmorphic elevation surfaces (`#111827`, `#181C24`, `#1E293B`, `#262A33`, `#31353E`) and glowing radial mesh backgrounds.
- **Taxonomic Color Tokens:**
  - **Iris Setosa**: Cyan (`#06B6D4` / `#4CD7F6`) • Cluster A
  - **Iris Versicolor**: Electric Indigo (`#8B5CF6` / `#D0BCFF`) • Cluster B
  - **Iris Virginica**: Emerald (`#10B981` / `#4EDEA3`) • Cluster C
  - **Boundary Drift / Outliers**: Rose (`#F43F5E` / `#FFB4AB`)
- **Typography:** `Geist` for body and structural headlines; `JetBrains Mono` for tabular metrics, feature readouts, and softmax logits.
- **Brand Identity:** High-precision stylized neural/botanical IrisAI SVG logo with glowing petal nodes.
- **5 Screen Views:**
  1. **Overview Dashboard**: Fisher's 1936 hero, 4-bento metrics, quick inference slider box, species cards, recent predictions feed, and deep dive links.
  2. **Predict Studio**: Step controls, benchmark presets, morphometric diagrams, real-time softmax probability bars, and botanical reference cards.
  3. **Dataset Analytics**: Interactive SVG scatter plots with decision cluster hulls, correlation metrics ($r = +0.96$), and cluster filtering.
  4. **Model Performance**: 3x3 normalized confusion matrix with misclassification spotlight, species diagnostics table, and $k$-NN hyperparameter tuning curve.
  5. **Dataset Explorer**: Searchable botanical registry, species filter pills, sorting options, and CSV export.

---

## 📁 Project Structure

```
Iris-Flower-Classification/
├── public/                              # Modern Stitch-designed Web Frontend
│   ├── index.html                       # Responsive SPA with all 5 Stitch views & modals
│   └── app.js                           # Frontend controller, live inference & charts
├── server.py                            # Production-grade Python HTTP server & REST API
├── test_server.py                       # Automated test suite for server & API endpoints
├── data/
│   └── iris.csv                         # Saved dataset (150 botanical records)
├── models/
│   ├── best_model.joblib                # Serialized top production model
│   ├── model_metadata.joblib            # Metadata, features & target labels
│   ├── metrics.json                     # Ground-truth evaluation metrics for API/UI
│   └── *_pipeline.joblib                # All 5 fitted candidate pipelines
├── notebooks/
│   └── iris_classification.ipynb        # Storytelling exploratory notebook
├── reports/
│   └── figures/                         # Generated diagnostic figures & charts
├── src/
│   ├── __init__.py
│   ├── data.py                          # Data loading, validation, splitting
│   ├── eda.py                           # Automated plotting routines
│   ├── models.py                        # Model zoo, CV, tuning, serialization
│   ├── evaluate.py                      # Metrics, confusion matrices, ROC, boundaries
│   └── predict.py                       # Inference engine and CLI
├── app.py                               # Styled Streamlit dashboard
├── train.py                             # Master training and evaluation orchestrator
├── pyproject.toml                       # Project configuration
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## 🚀 Installation & Getting Started

### 1. Prerequisites
- Python 3.10+ installed
- Standard virtual environment (`.venv`)

### 2. Setup Environment
```powershell
# Activate existing virtual environment:
.\.venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

---

## 💻 Usage & Interfaces

### 1. Modern Web Application (`server.py`)
Launch the production REST API server and Stitch web dashboard:
```powershell
.\.venv\Scripts\python.exe server.py
```
Open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

*(Note: The web client in `public/index.html` also includes built-in offline calculation heuristics, allowing it to be opened directly in any web browser!)*

### 2. Run Automated API & Server Test Suite
```powershell
.\.venv\Scripts\python.exe test_server.py
```

### 3. Train All Models & Generate Reports
Executes ingestion, EDA, 5-fold cross-validation, hyperparameter tuning, holdout testing, and exports `models/metrics.json`:
```powershell
.\.venv\Scripts\python.exe train.py
```

### 4. CLI Inference
Classify any specimen directly from terminal:
```powershell
.\.venv\Scripts\python.exe -m src.predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

### 5. Optional Streamlit Dashboard
Launch the interactive Streamlit laboratory (updated with Stitch dark theme):
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 🔌 REST API Documentation

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check, service status, and model readiness |
| `/api/predict` | `POST` | Real-time prediction with probabilities and morphological ratios |
| `/api/metrics` | `GET` | Model leaderboard, confusion matrix, and hyperparameter tuning |
| `/api/dataset` | `GET` | Filterable, searchable, and sortable botanical dataset records |

#### Example Predict Request (`POST /api/predict`):
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
#### Example Predict Response:
```json
{
  "predicted_species": "setosa",
  "confidence": 0.998,
  "probabilities": {
    "setosa": 0.998,
    "versicolor": 0.0016,
    "virginica": 0.0004
  },
  "morphology": {
    "sepal_ratio": 1.46,
    "petal_ratio": 7.0,
    "boundary_sigma": 3.24,
    "cluster": "Cluster A"
  },
  "model_name": "Support Vector Machine",
  "latency_ms": 0.8
}
```

---

## 🎨 Visual Artifacts Gallery

All diagnostic plots are saved at 200+ DPI in `reports/figures/`:
- `reports/figures/feature_distributions.png`: Multi-species KDE and histogram distributions.
- `reports/figures/pairplot.png`: Pairwise relationship matrix.
- `reports/figures/pca_projection.png`: 2D PCA cluster visualization.
- `reports/figures/confusion_matrices.png`: Side-by-side model confusion matrices.
- `reports/figures/roc_curves.png`: One-vs-Rest ROC curves.
- `reports/figures/decision_boundaries.png`: 2D classifier boundary contours.
- `reports/figures/knn_elbow_curve.png`: KNN $k$ parameter optimization.
- `reports/figures/feature_importance.png`: Feature importance breakdown.
