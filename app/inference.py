import numpy as np
import pandas as pd
import pickle


# Load ML Models
diabetes_model = pickle.load(open("models/diabetic_rf_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("models/diabetic_scaler.pkl", "rb"))

hyper_model = pickle.load(open("models/hypertension_xgb_model.pkl", "rb"))
hyper_scaler = pickle.load(open("models/hypertension_scaler.pkl", "rb"))
o_encoder_bp = pickle.load(open("models/hypertension_o_encoder_bp.pkl", "rb"))
o_encoder_exercise = pickle.load(open("models/hypertension_o_encoder_exercise.pkl", "rb"))


def predict_diabetes(data_dict):
    num_cols = ["Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
    cat_cols = ["Gender", "Age_Grp"]

    df = pd.DataFrame([data_dict])
    df["AGE"] = df["AGE"].fillna(22)
    df["Gender"] = df["Gender"].fillna(0)

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
    df = pd.DataFrame([data])

    df["Age"] = df["Age"].fillna(22)
    df["Stress_Score"] = df["Stress_Score"].fillna(5)
    df["Sleep_Duration"] = df["Sleep_Duration"].fillna(6)
    df["Medication"] = df["Medication"].fillna("No")
    df["Exercise_Level"] = df["Exercise_Level"].fillna("Moderate")
    df["Smoking_Status"] = df["Smoking_Status"].fillna("Non-Smoker")
    df["Family_History"] = df["Family_History"].fillna("No")

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
