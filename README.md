 Stroke Prediction Using Machine Learning with a Web Application

This project builds an end-to-end machine learning–based stroke prediction system using clinical and lifestyle attributes. It follows a strict no-data-leakage pipeline, applies advanced class balancing techniques, and deploys the best-performing model (XGBoost) through a Flask web application for real-time stroke-risk assessment.

 Project Overview

Stroke is a major global health concern, and early prediction can help in timely diagnosis and prevention.
This system analyzes patient attributes such as age, hypertension, heart disease, glucose level, BMI, and smoking status to predict the likelihood of stroke.

The project includes:

Data preprocessing & missing value handling

Exploratory Data Analysis (EDA)

Hybrid class balancing (Under-Sampling + SMOTE)

Feature selection (ANOVA, Chi-Square, Mutual Information)

Model training & evaluation

Web application deployment (Flask)

 Key Features

No Data Leakage: All preprocessing is fitted on training data only, ensuring unbiased evaluation.

Balanced Dataset: Combines Random Undersampling and SMOTE to handle class imbalance.

Multiple Algorithms Compared: LR, SVM, KNN, Random Forest, Naïve Bayes, XGBoost.

Best Model — XGBoost: Achieved ~87% accuracy 

Interactive Web App: Users enter health details to instantly receive stroke-risk predictions.

 Technologies Used

Python, Pandas, NumPy, Scikit-Learn, XGBoost

Imbalanced-Learn (SMOTE)

Flask Web Framework

HTML, CSS (Frontend UI)

 Workflow Pipeline

Load and clean dataset

Split into train/test sets (stratified)

Apply preprocessing (encoding, scaling) on training set

Address class imbalance using Under-Sampling + SMOTE

Perform feature selection

Train & evaluate multiple ML models

Select and save the best model

Deploy via Flask web interface


📦 Project Structure
|-- app.py
|-- home.html
|-- result.html
|-- error.html
|-- healthcare-dataset-stroke-data.csv
|-- best_model_xgboost.pkl
|-- preprocessing_objects.pkl
|-- static/
|-- templates/

🔍 How to Run the Project

Clone this repository

Install dependencies

Run the Flask application

Open the local URL to access the UI

🎯 Result

A fully functional machine learning system capable of real-time stroke prediction with high accuracy, a clean preprocessing pipeline, and a user-friendly web interface.
