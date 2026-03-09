# main_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# --- 1. Introduction and Goal ---
print("Credit Card Fraud Detection Model Training")
print("="*40)
print("Goal: To train a model that can accurately identify fraudulent credit card transactions.")
print("Dataset: Kaggle Credit Card Fraud Detection Dataset.")
print("\n")

# --- 2. Data Loading and Initial Exploration ---
print("[INFO] Loading dataset...")
# Load the dataset from the provided CSV file.
# The dataset can be downloaded from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
try:
    data = pd.read_csv('creditcard.csv')
    print("[SUCCESS] Dataset loaded successfully.")
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    print("\n")
except FileNotFoundError:
    print("[ERROR] 'creditcard.csv' not found. Please download it from Kaggle and place it in the same directory.")
    exit()

# --- 3. Exploratory Data Analysis (EDA) ---
print("[INFO] Performing Exploratory Data Analysis (EDA)...")

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum().max() == 0)
print("No missing values in the dataset.")

# Check the distribution of the target variable 'Class'
class_distribution = data['Class'].value_counts(normalize=True) * 100
print("\nClass Distribution:")
print(class_distribution)

# Visualize the class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data, palette=['#3498db', '#e74c3c'])
plt.title('Class Distribution (0: Non-Fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
# Save the plot
if not os.path.exists('static/images'):
    os.makedirs('static/images')
plt.savefig('static/images/class_distribution.png')
print("\n[INFO] Class distribution plot saved as 'static/images/class_distribution.png'")
# plt.show() # Uncomment to display plot during script execution

print("\n[ANALYSIS] The dataset is highly imbalanced. Fraudulent transactions represent only {:.2f}% of the data.".format(class_distribution[1]))
print("This means accuracy alone is not a good metric. We must focus on Precision, Recall, and F1-Score.")
print("\n")


# --- 4. Data Preprocessing ---
print("[INFO] Preprocessing data...")

# The 'Time' and 'Amount' columns are not on the same scale as the anonymized 'V' features.
# We need to scale them to avoid giving them undue weight in the model.
scaler = StandardScaler()
data['scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Rearrange columns to have the target 'Class' at the end
data = data[['scaled_Time', 'scaled_Amount'] + [col for col in data.columns if col not in ['scaled_Time', 'scaled_Amount', 'Class']] + ['Class']]

print("Data after scaling 'Time' and 'Amount' and rearranging columns:")
print(data.head())
print("\n")

# Define features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
# We use stratify=y to ensure the class distribution is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("\n")


# --- 5. Model Training ---
print("[INFO] Training the Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("[SUCCESS] Model training completed.")
print("\n")


# --- 6. Model Evaluation ---
print("[INFO] Evaluating the model...")
y_pred = model.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance on Test Data:")
print(f"  - Accuracy:  {accuracy:.4f}")
print(f"  - Precision: {precision:.4f} (Of all transactions predicted as fraud, what percentage were actually fraud?)")
print(f"  - Recall:    {recall:.4f} (Of all actual fraud transactions, what percentage did we correctly identify?)")
print(f"  - F1-Score:  {f1:.4f} (The harmonic mean of precision and recall)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Fraudulent', 'Fraudulent']))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig('static/images/confusion_matrix.png')
print("\n[INFO] Confusion matrix plot saved as 'static/images/confusion_matrix.png'")
# plt.show() # Uncomment to display plot


# --- 7. Model and Scaler Persistence ---
print("\n[INFO] Saving the trained model and scaler...")

# Create a directory to save the models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model to a file
joblib.dump(model, 'models/fraud_detection_model.joblib')
# Save the scaler object as well, as it is needed to process new data
joblib.dump(scaler, 'models/scaler.joblib')

print("[SUCCESS] Model saved as 'models/fraud_detection_model.joblib'")
print("[SUCCESS] Scaler saved as 'models/scaler.joblib'")
print("\n")
print("="*40)
print("Phase 1 Complete. You can now proceed to build the Flask API.")

