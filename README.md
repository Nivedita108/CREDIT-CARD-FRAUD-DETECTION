# CREDIT-CARD-FRAUD-DETECTION
Dataset Link: https://www.kaggle.com/datasets/zalando-research/fashionmnist

A machine learning-based solution to detect fraudulent credit card transactions using a Random Forest Classifier and SMOTE for class imbalance handling. This project demonstrates data preprocessing, model training, evaluation, and real-time user input prediction for fraud detection.

## Project Overview

Credit card fraud has become a significant concern with the rise of digital payments. The goal of this project is to classify transactions as **fraudulent** or **non-fraudulent**, focusing on reducing false negatives and false positives using machine learning techniques.

## Problem Statement

Build a model that can accurately classify credit card transactions and minimize:
- **False Negatives** (missed frauds)
- **False Positives** (false alarms)

## Objectives

- Preprocess the dataset and handle class imbalance
- Build a Random Forest classification model
- Evaluate model using metrics like accuracy, confusion matrix, and ROC-AUC
- Visualize results using heatmaps and plots
- Allow user input for real-time fraud detection

  ---
  
## Methodology

### 1. **Data Collection**
- Dataset: `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. **Data Preprocessing**
- Dropped irrelevant features (`Time`)
- Normalized `Amount` using `StandardScaler`
- Applied **SMOTE** to handle class imbalance

### 3. **Model Training**
- Split data (80:20) into train/test sets
- Trained a **Random Forest Classifier**

### 4. **Prediction & Evaluation**
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC Score
- Visualized using seaborn/matplotlib

### 5. **User Input Module**
- Accepts transaction feature values
- Predicts likelihood of fraud in real time
  
---

## Evaluation Metrics

- **Confusion Matrix**  
- **Accuracy**  
- **AUC-ROC Score**  
- **Visualization Tools**:
  - Correlation Matrix
  - Class Distribution (Bar Chart & Pie Chart)
  - Confusion Matrix Heatmap

## Results

- High accuracy (typically above 90%)
- Low false positives
- SMOTE significantly improved model sensitivity
- Real-time prediction functionality enabled

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib
- imbalanced-learn

---
## Future Improvements

- Use advanced ensemble methods (e.g., XGBoost, LightGBM)
- Threshold tuning for better precision-recall balance
- Deploy using Flask or Streamlit for web integration

---

## Authors

- Harshit Saini  
- Harsh Dubey  
- Nikhil  
- Disha  
- Nivedita  

**Supervisor**: Abhishek Shukla  
**Institution**: KIET Group of Institutions, Ghaziabad  
**Session**: 2024–25 (CSE - AIML)

---

## References

- [Kaggle - Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

## License

This project is part of an academic submission and is intended for educational purposes.
