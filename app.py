import csv
from datetime import datetime, date
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF
from src.config import MODEL_PATH

from openpyxl.styles import Font, PatternFill, Alignment

# QR Code Support
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False


# =========================================================
#                 STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
#                   PATH / CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
HISTORY_CSV = REPORTS_DIR / "history.csv"


# =========================================================
#                   MODEL LOADING
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
#               DEFAULT FORM VALUES
# =========================================================
DEFAULTS = {
    "pregnancies": 1,
    "glucose": 120.0,
    "blood_pressure": 70.0,
    "skin_thickness": 20.0,
    "insulin": 80.0,
    "bmi": 25.0,
    "dpf": 0.5,
    "age": 30,
}


def init_session_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("patient_name", "")
    st.session_state.setdefault("patient_id", "")
    st.session_state.setdefault("gender", "Male")
    st.session_state.setdefault("doctor_name", "")
    st.session_state.setdefault("test_date", date.today())

    st.session_state.setdefault("symptoms", "")
    st.session_state.setdefault("notes", "")

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("history", [])

    st.session_state.setdefault("reset_for_next", False)


def apply_reset_if_needed():
    if st.session_state.get("reset_for_next", False):

        # Reset numeric fields
        for key, value in DEFAULTS.items():
            st.session_state[key] = value

        # Reset patient info
        st.session_state["patient_name"] = ""
        st.session_state["patient_id"] = ""
        st.session_state["gender"] = "Male"
        st.session_state["doctor_name"] = ""
        st.session_state["test_date"] = date.today()
        st.session_state["symptoms"] = ""
        st.session_state["notes"] = ""

        st.session_state["reset_for_next"] = False


# =========================================================
#                   RISK CLASSIFICATION
# =========================================================
def classify_risk(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    return "High"


# =========================================================
#                   HISTORY LOG
# =========================================================
def log_history(record: dict):
    st.session_state["history"].append(record)

    fieldnames = [
        "timestamp", "patient_name", "patient_id", "gender",
        "doctor_name", "test_date", "age",
        "symptoms", "notes", "risk_level",
        "probability", "prediction", "pdf_name",
        "pregnancies", "glucose", "blood_pressure",
        "skin_thickness", "insulin", "bmi", "dpf"
    ]

    new_file = not HISTORY_CSV.exists()
    with HISTORY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


# =========================================================
#                   QR CODE
# =========================================================
def generate_qr_image(text: str, out_path: Path) -> Path:
    if not QR_AVAILABLE:
        return None
    img = qrcode.make(text)
    img.save(out_path)
    return out_path


# =========================================================
#             PDF REPORT GENERATION  (ASCII SAFE)
# =========================================================
def create_pdf_report(input_df: pd.DataFrame, prob, pred_class, patient_info, risk_level):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = REPORTS_DIR / f"diabetes_report_{ts}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "ABC Diagnostics Laboratory", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, "123 Health Street, Wellness City, India", ln=True, align="C")
    pdf.cell(0, 6, "Phone: +91-9876543210 | Email: info@abcdiagnostics.com", ln=True, align="C")

    pdf.ln(3)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "DIABETES RISK ASSESSMENT REPORT", ln=True, align="C")
    pdf.ln(3)

    # Timestamp
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="R")

    # Separator
    pdf.line(10, pdf.get_y() + 1, 200, pdf.get_y() + 1)
    pdf.ln(5)

    # QR Code
    qr_path = None
    if QR_AVAILABLE:
        qr_text = f"{patient_info['name']} | Risk: {risk_level} | Prob: {prob:.2f}"
        qr_path = REPORTS_DIR / f"qr_{ts}.png"
        generate_qr_image(qr_text, qr_path)
        pdf.image(str(qr_path), x=165, y=30, w=30)

    # Patient Info
    row = input_df.iloc[0]
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 11)

    pdf.cell(100, 6, f"Patient Name: {patient_info['name']}", ln=0)
    pdf.cell(0, 6, f"Patient ID: {patient_info['id']}", ln=1)
    pdf.cell(100, 6, f"Gender: {patient_info['gender']}", ln=0)
    pdf.cell(0, 6, f"Age: {row['Age']}", ln=1)
    pdf.cell(100, 6, f"Referring Doctor: {patient_info['doctor']}", ln=0)
    pdf.cell(0, 6, f"Test Date: {patient_info['test_date']}", ln=1)

    pdf.ln(3)

    # Clinical Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Clinical Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"Symptoms: {patient_info.get('symptoms','Not provided')}")
    pdf.ln(2)
    pdf.multi_cell(0, 6, f"Notes: {patient_info.get('notes','Not provided')}")

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Assessment Summary", ln=True)
    pdf.set_font("Arial", "", 11)

    pred_text = "Diabetes (Positive)" if pred_class == 1 else "No Diabetes (Negative)"
    pdf.cell(0, 6, f"Probability: {prob:.2f}", ln=True)
    pdf.cell(0, 6, f"Classification: {pred_text}", ln=True)
    pdf.cell(0, 6, f"Risk Category: {risk_level}", ln=True)

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Measured Parameters", ln=True)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(60, 7, "Parameter", border=1)
    pdf.cell(40, 7, "Value", border=1)
    pdf.cell(40, 7, "Units", border=1)
    pdf.cell(40, 7, "Reference Range", border=1, ln=1)

    pdf.set_font("Arial", "", 11)

    def add_row(n, v, u, r):
        pdf.cell(60, 7, n, border=1)
        pdf.cell(40, 7, str(v), border=1)
        pdf.cell(40, 7, u, border=1)
        pdf.cell(40, 7, r, border=1, ln=1)

    # NOTE: all ranges below now use ASCII "-" only
    add_row("Pregnancies", row["Pregnancies"], "-", "0-10")
    add_row("Glucose", row["Glucose"], "mg/dL", "70-140")
    add_row("Blood Pressure", row["BloodPressure"], "mmHg", "90-120 / 60-80")
    add_row("Skin Thickness", row["SkinThickness"], "mm", "10-40")
    add_row("Insulin", row["Insulin"], "uU/mL", "2-25 (fasting)")
    add_row("BMI", row["BMI"], "kg/m2", "18.5-24.9")
    add_row("Diabetes Pedigree", row["DiabetesPedigreeFunction"], "-", "Family history index")
    add_row("Age", row["Age"], "years", "-")

    pdf.ln(8)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Disclaimer: This automated report is for educational/demo purposes "
        "and should not replace professional medical evaluation.",
    )

    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, "Authorized Signatory: ____________________", ln=True)

    pdf.output(str(file_path))
    return file_path


# =========================================================
#                       UI LAYOUT
# =========================================================
def main():
    init_session_state()
    apply_reset_if_needed()
    model = load_model()

    # Title
    st.title("🩺 Diabetes Risk Assessment")
    st.caption("Machine-learning-powered clinical decision support tool")

    # ---------------------------------------------
    #            SIDEBAR CONTROL PANEL
    # ---------------------------------------------
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🔄 Reset All"):
            st.session_state["last_result"] = None
            st.session_state["history"] = []
            st.session_state["reset_for_next"] = True
            st.rerun()

        st.markdown("---")
        st.subheader("📁 Recent Patients")

        if st.session_state["history"]:
            df = (
                pd.DataFrame(st.session_state["history"])
                .sort_values("timestamp", ascending=False)
                .head(5)
            )
            for _, row in df.iterrows():
                st.write(
                    f"**{row['timestamp']}** — {row['patient_id']} | "
                    f"{row['risk_level']} ({row['probability']:.2f})"
                )
        else:
            st.info("No history yet.")

        if not QR_AVAILABLE:
            st.warning("QR Code unavailable. Install via: `pip install qrcode[pil]`")

    # ---------------------------------------------
    #                  MAIN TABS
    # ---------------------------------------------
    tab_predict, tab_history, tab_about = st.tabs([
        "🧪 New Assessment",
        "📚 Patient History",
        "ℹ️ About the Model"
    ])

    # ========================================================
    #                   TAB 1: PREDICTION
    # ========================================================
    with tab_predict:

        st.header("Patient Information")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name", key="patient_name")
            pid = st.text_input("Patient ID", key="patient_id")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")

        with col2:
            doctor = st.text_input("Referring Doctor", key="doctor_name")
            tdate = st.date_input("Test Date", key="test_date")

        st.subheader("Clinical Notes")
        symptoms = st.text_area("Symptoms / Complaints", key="symptoms")
        notes = st.text_area("Additional Notes", key="notes")

        st.header("Measurements")
        c1, c2 = st.columns(2)

        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, key="pregnancies")
            glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, key="glucose")
            bp = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, key="blood_pressure")
            skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, key="skin_thickness")

        with c2:
            insulin = st.number_input("Insulin (uU/mL)", 0.0, 900.0, key="insulin")
            bmi = st.number_input("BMI (kg/m2)", 0.0, 70.0, key="bmi")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, key="dpf")
            age = st.number_input("Age (years)", 1, 120, key="age")

        input_df = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                }
            ]
        )

        patient_info = {
            "name": name,
            "id": pid,
            "gender": gender,
            "doctor": doctor,
            "test_date": tdate.strftime("%Y-%m-%d"),
            "symptoms": symptoms,
            "notes": notes,
        }

        if st.button("Predict Diabetes Risk"):
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(model.predict(input_df)[0])
            risk = classify_risk(prob)

            pdf_path = create_pdf_report(input_df, prob, pred, patient_info, risk)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            record = {
                "timestamp": timestamp,
                "patient_name": name,
                "patient_id": pid,
                "gender": gender,
                "doctor_name": doctor,
                "test_date": tdate.strftime("%Y-%m-%d"),
                "age": age,
                "symptoms": symptoms,
                "notes": notes,
                "risk_level": risk,
                "probability": prob,
                "prediction": pred,
                "pdf_name": pdf_path.name,
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": bp,
                "skin_thickness": skin,
                "insulin": insulin,
                "bmi": bmi,
                "dpf": dpf,
            }

            log_history(record)

            st.session_state["last_result"] = {
                "prob": prob,
                "pred": pred,
                "risk": risk,
                "pdf_bytes": pdf_bytes,
                "pdf_name": pdf_path.name,
            }

        # -----------------------------
        #      SHOW RESULTS
        # -----------------------------
        if st.session_state["last_result"] is not None:
            r = st.session_state["last_result"]

            st.subheader("Assessment Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Probability", f"{r['prob']:.2f}")
            c2.metric("Risk Level", r["risk"])
            c3.metric("Classification", "Positive" if r["pred"] == 1 else "Negative")

            if r["pred"] == 1:
                st.error("Diabetes (Positive)")
            else:
                st.success("No Diabetes Detected")

            st.download_button(
                "Download PDF Report",
                data=r["pdf_bytes"],
                file_name=r["pdf_name"],
                mime="application/pdf",
            )

            if st.button("Next Patient"):
                st.session_state["last_result"] = None
                st.session_state["reset_for_next"] = True
                st.rerun()

    # ========================================================
    #                   TAB 2: HISTORY
    # ========================================================
    with tab_history:
        st.header("Patient History")

        if not st.session_state["history"]:
            st.info("No predictions yet.")
        else:
            df = pd.DataFrame(st.session_state["history"])
            st.dataframe(df)

            st.subheader("Export as Excel")

            excel_buffer = BytesIO()

            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Predictions", index=False)
                ws = writer.sheets["Predictions"]

                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                header_font = Font(bold=True, color="FFFFFF")
                header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

                for cell in ws[1]:
                    cell.font = header_font
                    cell.fill = header_fill

                for col in ws.columns:
                    max_len = max(len(str(c.value)) for c in col)
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

            excel_buffer.seek(0)

            st.download_button(
                "Download Excel",
                data=excel_buffer,
                file_name="history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ========================================================
#                   TAB 3: ABOUT THE MODEL
# ========================================================
    with tab_about:
        st.header("ℹ️ About the Diabetes Prediction Model")

        st.subheader("📘 Model Overview")
        st.write("""
        This application uses a **Logistic Regression** model to estimate the probability that
        a patient may have diabetes based on clinical measurements.
    
        Logistic Regression is a statistical machine-learning model commonly used in healthcare
        risk prediction because it is explainable and reliable for binary classification tasks.
        """)

        st.subheader("📊 Dataset Used")
        st.write("""
        The model is trained on the **PIMA Indians Diabetes Dataset**, a widely used benchmark
        dataset in medical ML research. It includes clinical features such as:
    
        - Pregnancies  
        - Glucose  
        - Blood Pressure  
        - Skin Thickness  
        - Insulin  
        - BMI  
        - Diabetes Pedigree Function  
        - Age  
        """)

        st.subheader("⚙️ How Predictions Work")
        st.write("""
        The model outputs a **probability between 0 and 1**:
    
        - **0.00 – 0.32 → Low Risk**  
        - **0.33 – 0.65 → Moderate Risk**  
        - **0.66 – 1.00 → High Risk**
    
        These thresholds are configurable and used only for **screening**, not diagnosis.
        """)

        st.subheader("🚨 Important Disclaimer")
        st.warning("""
        This tool is for **educational and demonstration purposes only**.
        It is **NOT a medical device** and must not be used for clinical decision-making.
        Always consult qualified healthcare professionals for medical advice.
        """)

        st.info("Model Version: 1.0.0  |  Algorithm: Logistic Regression  |  Maintainer: Pranav Gujjar")

if __name__ == "__main__":
    main()
