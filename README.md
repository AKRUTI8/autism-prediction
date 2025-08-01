🧠 Autism Prediction Model
This project is a machine learning pipeline designed to predict the likelihood of Autism Spectrum Disorder (ASD) based on questionnaire data. It involves data preprocessing, exploratory data analysis (EDA), handling imbalanced datasets, model building, hyperparameter tuning, and evaluation using multiple classifiers.

📌 Features
Data Cleaning and Preprocessing

Removal of unnecessary columns (ID, age_desc)

Conversion of data types

Country name mapping and value replacement

Label encoding of categorical variables

Exploratory Data Analysis (EDA)

Distribution plots for age and result

Outlier detection using IQR method

Boxplots and statistical summary

Handling Class Imbalance

Use of SMOTE (Synthetic Minority Over-sampling Technique)

Model Training and Evaluation

Models used:

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Hyperparameter tuning using RandomizedSearchCV

Performance evaluation using:

Accuracy Score

Confusion Matrix

Classification Report

Model Saving

Final trained model is saved as a .pkl file using pickle.

🚀 Technologies Used
Python 3.x

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn (SMOTE)

XGBoost

Pickle

📂 Dataset
Input CSV file: train.csv

Contains features related to individual responses for Autism screening tests.

Target column: Class/ASD (binary classification: Yes/No).

⚙️ Workflow
Data Loading: Read CSV dataset using Pandas.

Preprocessing: Clean and encode data for machine learning models.

Exploratory Analysis: Visualize data distribution and detect outliers.

Balancing Dataset: Apply SMOTE to handle class imbalance.

Model Building: Train multiple classifiers and tune hyperparameters.

Model Evaluation: Compare models based on accuracy and classification metrics.

Save Best Model: Export the best-performing model for future predictions.

▶️ How to Run
Clone the repository or download the notebook.

Install required dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
Place the train.csv file in the working directory.

Open and run Autism_Prediction.ipynb in Jupyter Notebook or VS Code.

The trained model (autism_model.pkl) will be generated for future predictions.

📊 Output
Graphical visualizations of data distribution and outliers.

Performance metrics for different classifiers.

A saved .pkl file containing the final trained model.
