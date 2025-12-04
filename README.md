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
<img width="200" height="200" alt="Screenshot 2025-12-04 at 21 21 23" src="https://github.com/user-attachments/assets/7ca0eeff-8dbc-4bcd-931f-67a8cf3ec768" />
<img width="200" height="200" alt="Screenshot 2025-12-04 at 21 21 35" src="https://github.com/user-attachments/assets/3ef227c4-b237-4573-9630-5ee006e00d50" />
<img width="200" height="200" alt="Screenshot 2025-12-04 at 21 22 19" src="https://github.com/user-attachments/assets/ba4bd6d5-94e2-4cc4-a657-eac2e3e735cd" />
<img width="200" height="200" alt="Screenshot 2025-12-04 at 21 19 58" src="https://github.com/user-attachments/assets/98a8012d-a38f-447d-870d-9ae31e5679fa" />
<img width="200" height="200" alt="Screenshot 2025-12-04 at 21 30 19" src="https://github.com/user-attachments/assets/ec08eee4-f9c1-4a73-a9ac-4279d4b685ea" />


# App link
https://diabetespredictionapp-ffcfgbmn3xxxe9ah7dl3rw.streamlit.app/

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
