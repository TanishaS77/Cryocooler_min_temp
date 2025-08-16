# Cryocooler ML Model for predicting minimum temperature

This repository contains machine learning models for predicting cryocooler performance, specifically for the first stage cryocooler of a PT415 Pulse Tube Cryocooler used in a Dilution Refrigerator. The data is from CFD analysis of varying loads for five seconds on ANSYS Fluent. 

## 📂 Project Structure
   cryocooler_min_temp_prediction/
│
├── src/
│ └── Cryocooler_code.py
│
├── figures/ 
│
├── data/ # Local datasets (not uploaded)
│ └── .gitkeep
│
├── README.md 
├── requirements.txt 
├── .gitignore 
└── LICENSE 


## Features
- Preprocessing and feature engineering (lagged variables, transitions)
- Multiple regression models:
  - Linear Regression
  - Random Forest (optimized with GridSearchCV)
  - XGBoost (optimized with GridSearchCV)
  - Stacking Regressor (ensemble of RF + XGBoost with Linear Regression as meta-learner)
- Model evaluation using MAE and R²
- Visualization:
  - Actual vs Predicted plots
  - Residual plots
  - Feature importance plots

## Installation

Clone the repository:

```bash
git clone https://https://github.com/TanishaS77/Cryocooler_min_temp
cd Cryocooler_min_temp

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

python src/Cryocooler_code.py

 