# ML Project: Basketball Player Performance Analysis

## Overview
This project uses machine learning techniques to analyze basketball player performance based on various features. The model is trained using decision trees and other machine learning techniques to classify player performance and predict future trends.

## Features of the Code
1. **Data Preprocessing**
   - The dataset is loaded and cleaned.
   - Features like height, weight, nationality, draft position, and player statistics are considered.
   - Unnecessary columns such as season are removed.
   
2. **Feature Engineering**
   - Players' performance data is grouped by key attributes.
   - The number of seasons per player is calculated to reflect their playing tenure.
   
3. **Machine Learning Model**
   - A **Decision Tree Classifier** is used to predict player performance.
   - The dataset is split into training and testing sets using **train_test_split** with stratification.
   - The model is trained on the training set and evaluated on the test set.
   
4. **Evaluation Metrics**
   - The **classification report** is generated, including precision, recall, F1-score, and accuracy.
   - Confusion matrix analysis is performed to assess model performance.

## How to Run
1. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ML-Project(Basketball).ipynb
   ```
3. Execute the cells in order to preprocess data, train the model, and evaluate results.

## Expected Output
- A trained decision tree classifier that predicts player performance.
- Classification report showing model accuracy and performance metrics.
- Insights into player statistics based on basketball data.

## Next Steps
- Improve model performance by tuning hyperparameters (e.g., max_depth, min_samples_split).
- Experiment with other ML models like RandomForest or SVM.
- Enhance data preprocessing and feature selection.

---
**Author:** ofer ashkenazy
**Date:** 05-03-2025
