import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
dataset = pd.read_csv("framingham.csv")
dataset.drop(['education'], inplace=True, axis=1)  # dropped because not necessary for our prediction

# Drop rows with NaN or Null values
dataset.dropna(axis=0, inplace=True)
print(dataset.shape)

# Split dataset into training and testing
X = np.asarray(dataset[['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(dataset['TenYearCHD'])

# Normalize the dataset
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Train-and-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Save the trained model
joblib.dump(logreg, 'logreg_model.pkl')

# Predict on test set to check accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of model is =', accuracy_score(y_test, y_pred))

# Load the scaler and model for prediction
scaler = joblib.load('scaler.pkl')
logreg = joblib.load('logreg_model.pkl')

# Input data for a new individual
# Example: age=50, male=1, cigsPerDay=10, totChol=200, sysBP=120, glucose=85
new_individual = np.array([[70, 5, 10, 180, 120, 85]])

# Normalize the input data using the loaded scaler
new_individual = scaler.transform(new_individual)

# Predict the probability of the individual having heart disease
probability = logreg.predict_proba(new_individual)

# Output the probability of heart disease
print('Probability of heart disease:', probability[0][1])
