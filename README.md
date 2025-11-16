# -Offer-Click-Prediction-
Predicting which offers customers will click using deep learning and feature engineering | Unstop Competition Project
# 🎯 Offer Click Prediction

A deep learning solution for predicting customer click-through rates on American Express promotional offers, developed for the Unstop ML Competition.

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

The goal is to predict which promotional offers customers are most likely to click on, and rank the top offers for each customer. This helps American Express:
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
