# 🚰 Pump It Up: Data Mining the Water Table

![DrivenData Logo](https://www.drivendata.org/static/images/drivendata-logo.svg)

This project is based on the [DrivenData competition: Pump It Up - Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/). The goal is to predict the functionality status of water pumps across Tanzania based on various environmental, geospatial, and operational features.

---

## 📊 Dataset

The dataset used in this project is provided by **Tanzania's Ministry of Water** and made available through DrivenData. It contains detailed information on over 59,000 waterpoints, including features such as construction year, pump type, water quality, installer, GPS coordinates, and more.

- **Target variable**: `status_group`  
  - `functional`
  - `functional needs repair`
  - `non functional`

---

## ⚙️ Project Overview

### 🔍 1. Exploratory Data Analysis (EDA)

- Analyzed feature distributions, missing values, and correlations.
- Investigated the relationship between key features (e.g., water quality, installer, population) and pump status.
- Identified significant class imbalance across target classes.

### 🧼 2. Feature Engineering and Preprocessing

- Created custom **data pipelines** for systematic preprocessing.
- Used `GeoContextImputer`:  
  A custom imputation strategy that leverages geographic grouping (e.g., by region/district) to fill in missing values more contextually and accurately.
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance during model training.

### 🤖 3. Model Building and Evaluation

Trained and compared multiple machine learning models:
- Random Forest Classifier 🌲
- Decision Tree
- Extra Trees
- XGBoost
- Support Vector Classifier (SVC)

Used **cross-validation** with `RandomForestClassifier` to establish a robust baseline.

### 🔎 4. Hyperparameter Optimization

Performed parameter tuning using:
- **GridSearchCV** for small, controlled search spaces.
- **RandomizedSearchCV** for broader, more efficient search.

### 📈 5. Evaluation Metrics

Evaluated models using:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Macro and Weighted averages
- Confusion Matrix
- ROC AUC (One-vs-Rest)
- Classification Reports

### 📊 6. Visualization

Visualized decision boundaries and data separability using:
- **PCA (Principal Component Analysis)**
- **t-SNE (t-distributed Stochastic Neighbor Embedding)**

For each class (1-vs-rest), plotted:
- 2D projection of data points
- Logistic regression decision boundaries
- Color-coded probability contours

---

## 📁 Project Structure

```bash
├── data/                  # Dataset (if small enough or scripts to download it)
├── notebooks/             # Jupyter notebooks for EDA, training, tuning, and visualization
├── src/                   # Custom transformers, pipeline utilities
├── models/                # Trained models (optional)
├── visuals/               # Saved plots: PCA, t-SNE, ROC curves, etc.
├── README.md              # This file
├── requirements.txt       # Project dependencies
