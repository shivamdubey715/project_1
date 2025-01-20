# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import shap
import pickle

# Step 1: Load and Explore the Dataset
print("Loading dataset...")
diabetes_data = pd.read_csv('../dataset/diabetes.csv')

# Display dataset information
print(f"Dataset Loaded: {diabetes_data.shape[0]} rows and {diabetes_data.shape[1]} columns.")
print("Statistical Summary of Features:")
print(diabetes_data.describe())

# Step 2: Data Preparation
print("\nPreparing data for training...")

# Separate features (X) and target variable (Y)
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

print(f"Training Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")

# Step 3: Train the Model
print("\nTraining the Support Vector Machine (SVM) model...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, Y_train)

# Evaluate the model on training data
train_accuracy = accuracy_score(Y_train, svm_model.predict(X_train))
print(f"Training Accuracy: {train_accuracy:.2f}")

# Evaluate the model on testing data
test_accuracy = accuracy_score(Y_test, svm_model.predict(X_test))
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Step 4: Save the Trained Model
model_path = '../models/diabetes_model_advanced.sav'
pickle.dump(svm_model, open(model_path, 'wb'))
print(f"Model saved at: {model_path}")

# Step 5: Explain Predictions with SHAP
print("\nInitializing SHAP explanations...")
explainer = shap.Explainer(svm_model, X_train)
shap_values = explainer(X_test)

# Visualize SHAP values for the first test sample
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Step 6: Predict for Sample Input
print("\nTesting the model with sample input...")
sample_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # Modify as needed
sample_array = np.array(sample_data).reshape(1, -1)

# Make a prediction and explain
prediction = svm_model.predict(sample_array)[0]
proba = svm_model.predict_proba(sample_array)

if prediction == 0:
    print("Result: The person is NOT diabetic.")
else:
    print("Result: The person is DIABETIC.")

# SHAP explanation for the sample
shap_sample_values = explainer(sample_array)
shap.waterfall_plot(shap_sample_values[0], max_display=10)

# Step 7: Feature Importance Visualization
print("\nFeature Importance:")
shap.summary_plot(shap_values, X_test)
