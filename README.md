# Diabetes Prediction App (Machine Learning + Streamlit)

## This project is an end-to-end Diabetes Prediction System built using:
1. Python
2. Logistic Regression
3. Streamlit interactive UI
4. Automated EDA, Model Training, Evaluation
5. PDF Report Generator
6. Patient History + Excel Export
7. Future-ready for deployment

# Features
# Machine Learning

Cleaned + preprocessed Pima Diabetes dataset

StandardScaler + Logistic Regression

Saved model as .pkl for production

Full pipeline: EDA → Model Training → Evaluation → App

# Interactive Streamlit App

Enter patient medical parameters

Get:

Probability of diabetes

Risk level (Low/Medium/High)

Model classification

Clean UI with themes

# PDF Report Generator

Auto-generated medical-style PDF

Patient ID + QR Code

Prediction summary

Parameters table

Timestamp

# Patient History

Every prediction stored in history.csv

View last 5 results

Export history to Excel (professional formatting)

# Utility Features

“Next Patient” button resets form instantly

“Reset All” clears history + last prediction

Easy configuration via config.py

# Project Structure
project-folder/
1. app.py                         # Streamlit main app
2.  main.py                        # Full pipeline runner
3. requirements.txt               # Dependencies
4. models
4.1 diabetes_log_reg.pkl     # Trained model
5. reports
5.1 history.csv              # Patient prediction history
5.2 *.pdf                    # Generated medical reports
6. src
6.1 config.py
6.2 data_prep.py
6.3 eda.py
6.4 train_model.py
6.5 evaluate_model.py
7. data
7.1 diabetes.csv

# Installation & Setup
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux

3. Install Requirements
pip install -r requirements.txt

4. Run the App
Run full ML pipeline + launch the app
python -m src.main


or start only Streamlit:

streamlit run app.py

# Model Performance
Metric	Score
Accuracy	~0.71
Precision	~0.60
Recall	~0.50
ROC-AUC	~0.81

Confusion matrix, feature importance, and reports are generated automatically.

# Screenshots



# Deployment (Streamlit Cloud)

To deploy:

Push project to GitHub

Go to https://share.streamlit.io

Select your repository

Set main file: app.py

Deploy — done

# Contributing

Pull requests are welcome.
Open issues for bugs, new features, or suggestions.

# License

This project is for educational & demonstration purposes.
