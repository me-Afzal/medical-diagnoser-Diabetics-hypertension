import numpy as np
import pandas as pd
import pickle


# Load ML Models
diabetes_model = pickle.load(open("app/models/diabetic_rf_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("app/models/diabetic_scaler.pkl", "rb"))

hyper_model = pickle.load(open("app/models/hypertension_xgb_model.pkl", "rb"))
hyper_scaler = pickle.load(open("app/models/hypertension_scaler.pkl", "rb"))
o_encoder_bp = pickle.load(open("app/models/hypertension_o_encoder_bp.pkl", "rb"))
o_encoder_exercise = pickle.load(open("app/models/hypertension_o_encoder_exercise.pkl", "rb"))


def predict_diabetes(data_dict):
    num_cols = ["Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
    cat_cols = ["Gender", "Age_Grp"]

    # ---------- Check missing required numeric features ----------
    required_features = num_cols  # same list
    missing = [f for f in required_features if f not in data_dict or data_dict[f] is None or str(data_dict[f]).strip() == ""]
    if missing:
        return f"Not enough details, missing fields: {', '.join(missing)}"

    df = pd.DataFrame([data_dict])
    df["AGE"] = df["AGE"].fillna(22) if "AGE" in df else 22
    df["Gender"] = df["Gender"].fillna(0) if "Gender" in df else 0

    # Age grouping
    def age_group(age):
        return "Child" if age < 18 else "Adult" if age < 65 else "Senior"

    df["Age_Grp"] = df["AGE"].apply(age_group)
    df.drop(columns=["AGE"], inplace=True)
    df["Age_Grp"] = df["Age_Grp"].map({"Child": 0, "Adult": 1, "Senior": 2})
    df["Cr"] = df["Cr"].clip(lower=15, upper=200)

    df_scaled_num = diabetes_scaler.transform(df[num_cols])
    X = np.concatenate((df_scaled_num, df[cat_cols].values), axis=1)

    pred = diabetes_model.predict(X)[0]
    return {0: "No Diabetes", 1: "Close to Diabetes", 2: "Diabetes"}[pred]


def predict_hyper(data):
    # ---- Required features check ----
    required_features = ['Salt_Intake', 'BMI', 'BP_History']
    missing = [f for f in required_features if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return f"Not enough details, missing fields: {', '.join(missing)}"

    df = pd.DataFrame([data])

    df["Age"] = df["Age"].fillna(22) if "Age" in df else 22
    df["Stress_Score"] = df["Stress_Score"].fillna(5) if "Stress_Score" in df else 5
    df["Sleep_Duration"] = df["Sleep_Duration"].fillna(6) if "Sleep_Duration" in df else 6
    df["Medication"] = df["Medication"].fillna("No") if "Medication" in df else "No"
    df["Exercise_Level"] = df["Exercise_Level"].fillna("Moderate") if "Exercise_Level" in df else "Moderate"
    df["Smoking_Status"] = df["Smoking_Status"].fillna("Non-Smoker") if "Smoking_Status" in df else "Non-Smoker"
    df["Family_History"] = df["Family_History"].fillna("No") if "Family_History" in df else "No"

    df["Medication"] = df["Medication"].apply(lambda x: 0 if x == "No" else 1)
    df["BP_History"] = o_encoder_bp.transform(df[["BP_History"]]).astype(int)
    df["Exercise_Level"] = o_encoder_exercise.transform(df[["Exercise_Level"]]).astype(int)
    df["Smoking_Status"] = df["Smoking_Status"].map({"Non-Smoker": 0, "Smoker": 1})
    df["Family_History"] = df["Family_History"].map({"No": 0, "Yes": 1})

    num_cols = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]
    df[num_cols] = hyper_scaler.transform(df[num_cols])

    final_order = [
        "Age", "Salt_Intake", "Stress_Score", "BP_History",
        "Sleep_Duration", "BMI", "Medication", "Family_History",
        "Exercise_Level", "Smoking_Status"
    ]
    df = df[final_order]

    return "Hypertension" if hyper_model.predict(df)[0] == 1 else "No Hypertension"
