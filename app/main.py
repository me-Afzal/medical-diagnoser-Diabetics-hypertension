import streamlit as st
from preprocess import extract_text_from_file, clean_text
from inference import predict_diabetes, predict_hyper

from llm_extractors.diabetics_llm_extractor import LabReportExtractor as DiabetesExtractor
from llm_extractors.hypertension_llm_extractor import LabReportExtractor as HypertensionExtractor


# ---- Load Secrets ----
API_KEY = st.secrets["api_key"]

# ---- UI THEME ----
st.set_page_config(
    page_title="Medical Diagnoser",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: black; color: white; }
    input, textarea { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center; margin-top: -10px;'>🩺 Medical Diagnoser</h1>
    <p style='text-align: center; font-size:18px;'>
        AI-powered predictions for Diabetes & Hypertension
    </p>
    <hr style="border: 1px solid gray;">
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------------------------------------
# ---------------------------------------  STREAMLIT UI --------------------------------------------------
tabs = st.tabs(["🩸 Diabetes Prediction", "💓 Hypertension Prediction"])


# ---------------- TAB 1: DIABETES -------------------
with tabs[0]:
    st.header("Diabetes Prediction")

    mode = st.radio("Input Method", ["Upload Lab Report", "Manual Entry"])

    extractor = DiabetesExtractor(API_KEY)

    if mode == "Upload Lab Report":
        report_file = st.file_uploader(
            "Upload PDF / DOCX / TXT",
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="diabetes_file"
        )

        if report_file and st.button("Extract & Predict"):
            text = extract_text_from_file(report_file)

            if not text:
                st.error("Couldn't extract text from the file.")
            else:
                text = clean_text(text)

                # LLM extraction
                data = extractor.extract(text)
                st.subheader("Extracted Fields")
                st.json(data)

                # ML Prediction
                result = predict_diabetes(data)
                center = st.columns([1, 2, 1])[1]
                with center:
                    st.markdown(
                        f"""<h3 style='text-align:center; margin-top:15px;'>
                        Result: <span style='color:white;'>{result}</span>
                        </h3>""",
                        unsafe_allow_html=True
                    )

    else:
        manual = {}
        gender = st.selectbox("Gender", ["Male", "Female"])
        manual["Gender"] = 1 if gender == "Male" else 0
        manual["AGE"] = st.number_input("Age", 1, 100, 30, key="diabetes_age")

        cols = st.columns(3)
        with cols[0]: manual["Urea"] = st.number_input("Urea (mmol/L)", 0.0, 30.0)
        with cols[1]: manual["Cr"] = st.number_input("Creatinine", 0.0, 300.0)
        with cols[2]: manual["HbA1c"] = st.number_input("HbA1c", 0.0, 18.0)
        with cols[0]: manual["Chol"] = st.number_input("Chol", 0.0, 20.0)
        with cols[1]: manual["TG"] = st.number_input("TG", 0.0, 20.0)
        with cols[2]: manual["HDL"] = st.number_input("HDL", 0.0, 20.0)
        with cols[0]: manual["LDL"] = st.number_input("LDL", 0.0, 20.0)
        with cols[1]: manual["VLDL"] = st.number_input("VLDL", 0.0, 20.0)
        with cols[2]: manual["BMI"] = st.number_input("BMI", 0.0, 60.0)

        if st.button("Predict"):
            result = predict_diabetes(manual)
            center = st.columns([1, 2, 1])[1]
            with center:
                st.markdown(
                    f"""<h3 style='text-align:center; margin-top:15px;'>
                    Result: <span style='color:white;'>{result}</span>
                    </h3>""",
                    unsafe_allow_html=True
                )


# ---------------- TAB 2: HYPERTENSION -------------------
with tabs[1]:
    st.header("Hypertension Prediction")

    mode = st.radio("Input Method", ["Upload Lab Report", "Manual Entry"], key="hyper_mode")

    extractor = HypertensionExtractor(API_KEY)

    if mode == "Upload Lab Report":
        report_file = st.file_uploader(
            "Upload PDF / DOCX / TXT",
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="hypertension_file"
        )

        if report_file and st.button("Extract & Predict", key="hyper_extract"):
            text = extract_text_from_file(report_file)

            if not text:
                st.error("Couldn't extract text from the file.")
            else:
                text = clean_text(text)

                # LLM extraction
                data = extractor.extract(text)
                st.subheader("Extracted Fields")
                st.json(data)

                # ML Prediction
                result = predict_hyper(data)
                center = st.columns([1, 2, 1])[1]
                with center:
                    st.markdown(
                        f"""<h3 style='text-align:center; margin-top:15px;'>
                        Result: <span style='color:white;'>{result}</span>
                        </h3>""",
                        unsafe_allow_html=True
                    )
                

    else:
        manual = {}
        manual["Age"] = st.number_input("Age", 1, 100, 30, key="hyper_age")
        manual["Salt_Intake"] = st.number_input("Salt Intake (grams/day)", 0.0, 50.0)
        manual["Stress_Score"] = st.slider("Stress Score (0–10)", 0, 10, 5)
        manual["Sleep_Duration"] = st.number_input("Sleep Duration (hours)", 0.0, 24.0)
        manual["BMI"] = st.number_input("BMI", 0.0, 60.0)

        manual["BP_History"] = st.selectbox("Blood Pressure History", ["Normal", "Prehypertension", "Hypertension"])
        manual["Medication"] = st.selectbox("Medication", ["No", "ACE Inhibitor", "Others"])
        manual["Exercise_Level"] = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
        manual["Smoking_Status"] = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
        manual["Family_History"] = st.selectbox("Family History", ["No", "Yes"])

        if st.button("Predict", key="hyper_btn"):
            result = predict_hyper(manual)
            center = st.columns([1, 2, 1])[1]
            with center:
                st.markdown(
                    f"""<h3 style='text-align:center; margin-top:15px;'>
                    Result: <span style='color:white;'>{result}</span>
                    </h3>""",
                    unsafe_allow_html=True
                )
