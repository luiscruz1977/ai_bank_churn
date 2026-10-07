import joblib
import pandas as pd

MODEL_PATH = "ml/model.pkl"

model = joblib.load(MODEL_PATH)


def predict_churn(customer: dict) -> dict:

    data = pd.DataFrame([customer])

    data = data.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    probability = model.predict_proba(data)[0][1]

    if probability < 0.30:
        risk = "LOW"
    elif probability < 0.55:
        risk = "MEDIUM"
    elif probability < 0.80:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "churn_probability": round(float(probability), 4),
        "risk": risk
    }