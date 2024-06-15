# Heart Disease Prediction Model

This repository contains a machine learning model developed using logistic regression to predict the likelihood of a person developing heart disease in the future. The model is trained on the Framingham Heart Study dataset, containing various health-related parameters, and uses these features to output a binary prediction indicating the presence or absence of heart disease.

## Features:
- **Algorithm**: Logistic Regression
- **Programming Language**: Python
- **Libraries Used**: Pandas, NumPy, Scikit-learn, Joblib
- **Dataset**: Framingham Heart Study dataset

## Dataset:
The dataset used in this project is the <u>Framingham Heart Study dataset</u>. It includes the following features:
- `male`: Gender (1 = Male, 0 = Female)
- `age`: Age of the individual
- `currentSmoker`: Current smoking status (1 = Smoker, 0 = Non-smoker)
- `cigsPerDay`: Number of cigarettes smoked per day
- `totChol`: Total cholesterol level
- `sysBP`: Systolic blood pressure
- `diaBP`: Diastolic blood pressure
- `BMI`: Body Mass Index
- `heartRate`: Heart rate
- `glucose`: Glucose level
- `TenYearCHD`: 10-year risk of coronary heart disease (1 = Yes, 0 = No)

## Usage:
1. **Data Preparation**: The dataset is preprocessed to handle missing values and normalize features.
2. **Model Training**: The logistic regression model is trained using the training dataset.
3. **Prediction**: The trained model is used to make predictions on the test dataset and new individual data.
4. **Evaluation**: The model's performance is evaluated using accuracy metrics.

## Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/imArjunMalik/heart-disease-prediction.git
