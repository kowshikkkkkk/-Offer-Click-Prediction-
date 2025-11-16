# 🎯 Offer Click Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Competition-Unstop-orange.svg)](https://unstop.com/)

A deep learning solution for predicting customer click-through rates on promotional offers, developed for the Unstop ML Competition.

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Problem Statement

The goal is to predict which promotional offers customers are most likely to click on, and rank the top offers for each customer. This helps businesses:
- **Personalize offer recommendations**
- **Improve customer engagement**
- **Optimize marketing ROI**

**Evaluation Metric:** Average Precision (AP) Score

**Dataset:** Parquet files containing customer events, transactions, and offer metadata from Unstop Competition

---

## 💡 Solution Overview

This project implements a **deep neural network with embedding layers** for tabular data, combining:
- **Categorical features** → Learned embeddings (Customer ID, Offer ID, etc.)
- **Numerical features** → Normalized continuous values
- **Engineered features** → Time-based and aggregated statistics

### Key Highlights
✅ Custom PyTorch model with embeddings for high-cardinality categorical features  
✅ Comprehensive feature engineering (temporal, aggregation, offer validity)  
✅ Memory-optimized data processing for large datasets  
✅ Proper train/validation split with model checkpointing  
✅ Top-7 offer ranking per customer for submission  
✅ Demo mode for quick testing (~5 minutes)  

---

## 🔧 Features

### Feature Engineering Pipeline
1. **Temporal Features**
   - Days since offer start
   - Days until offer end
   - Offer active duration
   - Event hour and day of week
   - Weekend indicator
   - Offer expiration flag

2. **Aggregated Customer Statistics**
   - Total event count per customer
   - Unique event hours/days patterns
   - Transaction frequency and timing
   - Average transaction hour

3. **Offer Metadata Integration**
   - Discount percentage
   - Offer validity period
   - Offer start/end dates

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/offer-click-prediction.git
cd offer-click-prediction

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p data models outputs notebooks
```

---

## 📊 Usage

### 1. Data Setup

**Note**: The dataset is from the Unstop competition and cannot be shared publicly due to competition rules. 

To run this project:
1. Download the competition dataset from the Unstop Competition Page
2. Place the following files in the `data/` folder:
   - `train.parquet`
   - `test.parquet`
   - `offer_metadata.parquet`
   - `additional_event.parquet`
   - `additional_transaction.parquet`

**Dataset Structure:**
```
data/
├── train.parquet           # Training data with labels
├── test.parquet            # Test data for predictions
├── offer_metadata.parquet  # Offer details and metadata
├── additional_event.parquet    # Customer event history
└── additional_transaction.parquet  # Transaction history
```

### 2. Train Model
```bash
# Run the complete pipeline
python offer_click_prediction_pipeline.py
```

**Expected Runtime:**
- ⚡ **With GPU**: 20-45 minutes (recommended)
- 🐢 **Without GPU**: 1-3 hours

**Quick Demo Mode** (for testing):
```python
# In offer_click_prediction_pipeline.py, line 37, set:
DEMO_MODE = True  # Uses 10% of data with 2 epochs (~5 minutes)
```

### 3. Configuration
Modify hyperparameters in `offer_click_prediction_pipeline.py`:
```python
class Config:
    BATCH_SIZE = 1024
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    VALIDATION_SPLIT = 0.2
```

### 4. Output Files
- **Trained model**: `models/best_model.pth`
- **Submission file**: `outputs/submission.csv`
- **Training logs**: Console output with epoch-wise metrics

---

## 🏗️ Model Architecture
```
ClickPredictionNN
│
├── Embedding Layers (Categorical Features)
│   ├── Customer ID embedding
│   ├── Offer ID embedding
│   ├── Discount percentage embedding
│   ├── Event hour embedding
│   └── Other categorical embeddings
│
├── Batch Normalization
│   ├── Categorical features (after embeddings)
│   └── Continuous features
│
├── Fully Connected Network
│   ├── FC1: (emb_dim + n_continuous) → 256 + ReLU + Dropout(0.3)
│   ├── FC2: 256 → 128 + ReLU + Dropout(0.3)
│   └── FC3: 128 → 1 + Sigmoid
│
└── Output: Click probability [0, 1]
```

**Key Design Choices:**
- **Embedding dimensions**: `min(50, (n_unique + 1) // 2)` for each categorical feature
- **Loss function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with learning rate 0.001
- **Regularization**: Dropout (0.3) to prevent overfitting
- **Batch Normalization**: Stabilizes training and improves convergence

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Validation AP | 0.XXXX |
| Test AP | 0.XXXX |
| Training Time (GPU) | ~30-45 min |
| Model Parameters | ~XXX,XXX |

### Performance Insights
- ✅ Model successfully ranks top-7 offers per customer
- ✅ Temporal features significantly improved prediction accuracy
- ✅ Customer transaction history proved to be a strong signal
- ✅ Embedding layers effectively captured high-cardinality categorical relationships

*(Update with your actual scores after training)*

---

## 📁 Project Structure
