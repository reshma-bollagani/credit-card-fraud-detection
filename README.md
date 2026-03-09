💳 End-to-End Credit Card Fraud Detector

<p align="center">
<em>A full-stack, AI-enhanced web application for detecting fraudulent credit card transactions in real-time.</em>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask Badge">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript Badge">
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5 Badge">
<img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS Badge">
</p>

✨ Project Overview
This project demonstrates a complete, end-to-end machine learning application designed to solve a real-world problem: credit card fraud. It features a robust Python backend powered by Flask that serves a Scikit-learn model, coupled with a responsive and interactive frontend.

What sets this project apart is its integration with the Google Gemini API. When a transaction is flagged as fraudulent, the application can generate an AI-powered risk report, explaining the potential risks and providing clear, actionable steps for the user.

Note: This is a demonstration project. The V1-V28 features are anonymized, so the primary way to test the application is by using the provided "Sample Data" buttons.

🚀 Key Features
🧠 ML-Powered Predictions: Utilizes a Logistic Regression model trained on a highly imbalanced dataset to classify transactions with high precision.

🖥️ Interactive Web UI: A clean, modern, and fully responsive user interface built with Tailwind CSS allows for seamless interaction on any device.

🤖 AI-Powered Risk Analysis: On detecting fraud, a user can generate an intelligent report from the Gemini API that provides context and recommended actions.

⚡ Real-Time API: A robust Flask API exposes the ML model's prediction capabilities over a /predict endpoint, responding with JSON.

📊 Data Preprocessing: Demonstrates key ML concepts like feature scaling (StandardScaler) and strategies for handling imbalanced data.

✅ Easy Demonstration: Includes "Normal" and "Fraud" sample data buttons to allow anyone (including recruiters!) to easily test and verify the model's functionality.

Gemini AI Report in Action

🛠️Technology                                                                                                                                                                                                  
Backend & ML

Python, Flask, Scikit-learn, Pandas, NumPy, Joblib                                                                                                   
Frontend

HTML5, Tailwind CSS, JavaScript                                                                                               
AI Integration
Google Gemini API                                                                                                                                                                                                     
📈 Machine Learning Model
The core of this project is a Logistic Regression model trained to distinguish between legitimate and fraudulent transactions.

Model: Logistic Regression was chosen as a strong, interpretable baseline.

Dataset: The model was trained on the Kaggle Credit Card Fraud dataset, which contains 284,807 transactions, with only 492 (0.172%) being fraudulent.

Challenge: The extreme class imbalance makes 'accuracy' a misleading metric. Therefore, the model's performance was evaluated using Precision, Recall, and F1-Score, which are more suitable for this type of problem.

Performance on Test Data:

Precision (Fraud): 0.86 (When it predicts fraud, it's correct 86% of the time).

Recall (Fraud): 0.60 (It correctly identifies 60% of all actual frauds).

F1-Score (Fraud): 0.71 (A harmonic mean of Precision and Recall).

📂 Project Structure
credit-fraud-detector/
├── models/
│   ├── fraud_detection_model.joblib  # The trained ML model
│   └── scaler.joblib                 # The data scaler
├── static/
│   └── images/                       # EDA outputs
├── templates/
│   └── index.html                    # The main frontend file
├── app.py                            # Flask backend server
├── main_model.py                     # Script to train and save the ML model
├── .gitignore                        # Tells Git to ignore large files
└── README.md                         # You are here!

⚙️ How to Run This Project Locally
To set up and run this project on your local machine, follow these steps.

1. Prerequisites
Python 3.x installed

pip (Python package installer)

2. Clone the Repository
git clone https://github.com/Hasadhika/credit-card-fraud-detection.git
cd credit-card-fraud-detection

3. Set Up a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

(You will need to create a requirements.txt file for this to work. See below.)

5. Create requirements.txt
Create a file named requirements.txt in the root directory and add the following lines:

Flask
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn

6. Download the Dataset
Download the "Credit Card Fraud Detection" dataset from Kaggle.

Place the creditcard.csv file in the root of the project directory.

7. Train the Model
Run the training script. This will create the models and static/images directories.

python main_model.py

8. Run the Flask Application
This will start the backend server.

python app.py

9. View the Application
Open your web browser and navigate to: http://127.0.0.1:5000

Congratulations on building this project!
