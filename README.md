# Predicting Electric Vehicle Purchases

Supervised Machine Learning Project by Xplug

## About the Project

This project uses machine learning to predict whether a customer is likely to purchase an Electric Vehicle (EV).

The goal is to use customer information such as income, environmental concern, charging access, subsidy availability, and range anxiety to predict EV purchase interest.

## Main Question

Can machine learning predict whether a customer will purchase an Electric Vehicle based on their personal, financial, transportation, and charging-related information?

## What I Did

* Explored and cleaned the dataset

* Analyzed customer and EV purchase patterns

* Studied relationships between important features and EV purchase behavior

* Built a Logistic Regression baseline model

* Tested CatBoost, LightGBM, and XGBoost models

* Used 5-Fold Stratified Cross-Validation

* Compared different XGBoost tree depths

* Selected XGBoost with a maximum tree depth of 5

* Generated probability predictions for the Kaggle test dataset

* Submitted multiple models to Kaggle and compared their performance

## Result

The best model was **XGBoost with max_depth=5**.

5-Fold Cross-Validation ROC-AUC:

**0.94195**

Kaggle Public Score:

**0.94172**

The Depth 5 model performed slightly better than the Depth 6 model on the Kaggle leaderboard.

## Tools Used

* Python

* Pandas

* NumPy

* Matplotlib

* Seaborn

* Scikit-learn

* XGBoost

* CatBoost

* LightGBM

* Git

* GitHub

* Kaggle

XGBoost was able to learn meaningful patterns in customer information and predict Electric Vehicle purchase interest with a strong ROC-AUC score.

## Author

Xplug
