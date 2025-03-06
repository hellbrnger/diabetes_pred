import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
diab_data = pd.read_csv('diabetes.csv')

# Handle missing values if any
diab_data.fillna(diab_data.mean(), inplace=True)

# Splitting features & target
X = diab_data.drop(columns='Outcome', axis=1)
Y = diab_data['Outcome']

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
X_test_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, X_test_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump((classifier, scaler), model_file)
