
import subprocess
from pathlib import Path
#from .eda import run_eda          # if you don't want EDA every time, you can remove this import
from .train_model import train_model
from .evaluate_model import evaluate_model
from .config import PROJECT_ROOT


def main():
    print("STEP 1: EDA ")
    # Comment out this line if  don't want plots every time
    #run_eda()
    #print("EDA finished.\n")

    print("STEP 2: Training model ")
    train_model()   # trains and saves models/diabetes_log_reg.pkl
    print("Training finished.\n")

    print("STEP 3: Evaluating model ")
    evaluate_model()
    print("Evaluation finished.\n")

    print("STEP 4: Starting Streamlit app ")
    # This starts the Streamlit server and opens the browser
    subprocess.run(["streamlit", "run", "app.py"],cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
