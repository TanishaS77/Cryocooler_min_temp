# Cryocooler ML – Predicting Minimum Temperature

This repository contains machine learning models for predicting cryocooler performance, specifically for the first stage cryocooler of a PT415 Pulse Tube Cryocooler used in a Dilution Refrigerator.
The dataset originates from CFD simulations (ANSYS Fluent) of varying loads over five seconds.

## 📂 Project Structure
   cryocooler_min_temp_prediction/
│
├── src/                   
│   └── Cryocooler_code.py
│
├── figures/               
│
├── data/                  # Local datasets (not uploaded)
│   └── .gitkeep
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

git clone https://github.com/TanishaS77/Cryocooler_min_temp.git
cd Cryocooler_min_temp

### Create virtual environment
python -m venv venv

### Activate
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

### Install dependencies
pip install -r requirements.txt

### Run training
python src/Cryocooler_code.py


##📊 Results
-Actual vs Predicted plots
-Residual plots
-Feature importance (RF, XGBoost)

##License
This project is licensed under the MIT License – see the LICENSE file for details.

##Contributions, issues, and feature requests are welcome!


 