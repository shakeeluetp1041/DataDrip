# 💧 Waterpumps Functionality Predictions Across Tanzania

This project focuses on predicting the **functional status of waterpumps in Tanzania** using machine learning techniques. The aim is to assist government and aid organizations in efficiently identifying which waterpoints need maintenance or replacement, ultimately improving access to clean water across the country.

---

## 📊 Dataset Overview

- **Total Records**: 59,400  
- **Total Features**: 41 (including the target)  
- **Target Variable**: `status_group`  
  - `functional`  
  - `functional needs repair`  
  - `non functional`

The dataset was provided in CSV format. Upon loading, the shape of the dataset was found to be:
# 💧 Waterpumps Functionality Predictions Across Tanzania

This project focuses on predicting the **functional status of waterpumps in Tanzania** using machine learning techniques. The aim is to assist government and aid organizations in efficiently identifying which waterpoints need maintenance or replacement, ultimately improving access to clean water across the country.

---

## 📊 Dataset Overview

- **Total Records**: 59,400  
- **Total Features**: 41 (including the target)  
- **Target Variable**: `status_group`  
  - `functional`  
  - `functional needs repair`  
  - `non functional`

The dataset was provided in CSV format. Upon loading, the shape of the dataset was found to be:(59400, 41)

## 🔧 Data Processing

As a first step, **each column was individually analyzed and processed** based on its data type and relevance to the target variable.  

> 🛠 **Note**: Detailed column-wise preprocessing decisions are documented in the Presentation Sprint1.pdf file in the repo.

### Common preprocessing techniques used:
- Handling missing values
- Encoding high-cardinality categorical features
- Ordinal, label and one-hot encoding
- Dropping uninformative or duplicate features


---

## 🚀 Model Training

We employed the **XGBoost Classifier** for this multi-class classification problem due to its robustness and ability to handle missing data and mixed feature types effectively.

### ✅ Results:
- **Training Accuracy**: `87.09%`  
- **Testing Accuracy**: `81.77%`

---

## 📌 Project Highlights

- End-to-end data cleaning and preprocessing tailored per feature  
- Efficient handling of high-cardinality and mixed-type features  
- Use of **gradient boosting** for powerful and scalable classification  
- Practical insights for deployment in rural infrastructure monitoring

