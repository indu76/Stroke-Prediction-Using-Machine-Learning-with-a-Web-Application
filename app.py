# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import joblib
import os

app = Flask(__name__)

# ---------- CONFIG ----------
PREPROCESS_FILENAME = "preprocessing_objects.pkl"
MODEL_FILENAME = "best_model_xgboost.pkl"   # <-- change this to your actual model filename if different
# ----------------------------

def load_models():
    prep_objects = None
    model = None
    # Load preprocessing objects
    if os.path.exists(PREPROCESS_FILENAME):
        try:
            with open(PREPROCESS_FILENAME, "rb") as f:
                prep_objects = pickle.load(f)
        except Exception as e:
            print("Failed to load preprocessing_objects:", e)
    else:
        print(f"Warning: {PREPROCESS_FILENAME} not found.")

    # Load ML model
    if os.path.exists(MODEL_FILENAME):
        try:
            model = joblib.load(MODEL_FILENAME)
        except Exception as e:
            print("Failed to load model:", e)
    else:
        print(f"Warning: {MODEL_FILENAME} not found.")

    return prep_objects, model

preprocessing_objects, model = load_models()

def preprocess_input(input_data: dict):
    """
    Accepts raw input_data dictionary (strings/floats) and returns preprocessed
    DataFrame ready for model prediction using preprocessing_objects.
    """
    if preprocessing_objects is None:
        # fallback: simple conversion to dataframe
        return pd.DataFrame([input_data])

    df = pd.DataFrame([input_data])
    # handle BMI missing
    if "bmi" in df.columns and pd.isna(df["bmi"].iloc[0]):
        if preprocessing_objects.get("bmi_mean_train") is not None:
            df["bmi"].fillna(preprocessing_objects["bmi_mean_train"], inplace=True)

    # Label encode categorical using saved label_encoders
    le_dict = preprocessing_objects.get("label_encoders", {})
    for col, le in le_dict.items():
        if col in df.columns:
            # map unseen values to nearest fallback: if unseen, add code len(le.classes_) (consistent with training)
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                # fallback mapping: unknown -> last index (or 0)
                mapping = {cls: i for i, cls in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(lambda x: mapping[x] if x in mapping else len(mapping))

    # Ensure column order and presence
    feature_names = preprocessing_objects.get("feature_names")
    if feature_names:
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]
    else:
        # keep current df
        pass

    # Scale using saved scaler if present
    scaler = preprocessing_objects.get("scaler")
    if scaler is not None:
        try:
            arr = scaler.transform(df)
            df_scaled = pd.DataFrame(arr, columns=df.columns, index=df.index)
            return df_scaled
        except Exception as e:
            print("Scaler transform failed:", e)
            return df
    else:
        return df

def predict_stroke_probability(input_data):
    try:
        X = preprocess_input(input_data)
        if model is None:
            return None, None
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0][1]
            except Exception:
                proba = None
        else:
            # try decision_function fallback
            try:
                scores = model.decision_function(X)
                proba = (scores - scores.min()) / (scores.max() - scores.min())
                proba = float(proba[0])
            except Exception:
                proba = None
        return int(pred), float(proba) if proba is not None else None
    except Exception as e:
        print("Prediction error:", e)
        return None, None

# Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        def get_str(name):
            return request.form.get(name)

        def get_float(name, default=np.nan):
            v = request.form.get(name)
            try:
                return float(v) if v is not None and str(v).strip() != "" else default
            except:
                return default

        data = {
            "gender": get_str("gender"),
            "age": get_float("age"),
            "hypertension": int(request.form.get("hypertension", 0)),
            "heart_disease": int(request.form.get("heart_disease", 0)),
            "ever_married": get_str("ever_married"),
            "work_type": get_str("work_type"),
            "Residence_type": get_str("Residence_type"),
            "avg_glucose_level": get_float("avg_glucose_level"),
            "bmi": get_float("bmi", default=np.nan),
            "smoking_status": get_str("smoking_status")
        }

        pred, proba = predict_stroke_probability(data)
        if pred is None:
            return render_template("error.html", error_message="Prediction failed. Check server logs.")

        result = {
            "prediction": pred,
            "probability": proba,
            "input_data": data
        }
        return render_template("result.html", result=result)
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    if preprocessing_objects is None or model is None:
        print("WARNING: preprocessing objects or model not loaded. Check that:")
        print(" -", PREPROCESS_FILENAME)
        print(" -", MODEL_FILENAME)
    app.run(debug=True, host="0.0.0.0", port=5000)
