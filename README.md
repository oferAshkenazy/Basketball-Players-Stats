# Machine Learning Project: Basketball Performance Prediction

## Overview
This project applies machine learning techniques to analyze and predict basketball player performance based on various features. Multiple classification models are trained and evaluated to determine the best approach for predicting player outcomes.

## Features of the Code
### 1. **Data Preprocessing**
- Loads basketball player performance data into a **Pandas DataFrame**.
- Identifies **categorical** and **numerical** features.
- Handles missing values using **imputation techniques** or by removing incomplete rows.
- Filters out unnecessary columns to improve model efficiency.

### 2. **Exploratory Data Analysis (EDA)**
- Uses **Seaborn** and **Matplotlib** to visualize data distributions.
- **Count plots** are used to analyze categorical variables (e.g., team, position, nationality).
- **Histograms and box plots** help in understanding the spread of numerical features (e.g., height, weight, points per game).
- **Correlation heatmaps** determine relationships between numerical features and target labels.

### 3. **Feature Engineering**
- Encodes categorical variables using **one-hot encoding** or **label encoding**.
- Normalizes numerical features to ensure better performance in models like **SVM and KNN**.
- Creates new meaningful features, such as **BMI, experience years, efficiency rating**, etc.
- Drops highly correlated features to prevent redundancy in model training.

### 4. **Machine Learning Models**
The following classification models are implemented and compared:
- **Logistic Regression**: Simple and interpretable model for binary classification.
- **Random Forest Classifier**: An ensemble learning method that improves accuracy by combining multiple decision trees.
- **Support Vector Machine (SVM)**: Finds the optimal hyperplane to separate classes.
- **Decision Tree Classifier**: A tree-based model that works well with categorical and numerical data.
- **K-Nearest Neighbors (KNN)**: A distance-based classifier that predicts based on closest neighbors.
- **Naïve Bayes Classifier**: Assumes independence among predictors; useful for categorical data.
- **AdaBoost Classifier**: A boosting algorithm that improves weak classifiers.
- **Gradient Boosting Classifier**: Another boosting technique that optimizes classification performance over multiple iterations.

### 5. **Model Training & Evaluation**
- Splits the dataset into **training (80%)** and **testing (20%)** subsets using `train_test_split`.
- Trains all models using **Scikit-Learn**.
- Evaluates models using:
  - **Accuracy Score**: Percentage of correct predictions.
  - **Classification Report**: Includes **Precision, Recall, F1-score, and Support** for each class.
  - **Confusion Matrix**: Displays true positives, false positives, false negatives, and true negatives.
- Compares performance across models and selects the best one based on metrics.

