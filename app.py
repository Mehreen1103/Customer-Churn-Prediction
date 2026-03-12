from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse

# Load model
model = joblib.load("churn_model.pkl")

app = FastAPI()

class Features(BaseModel):
    features: list  # 18 raw inputs from UI

@app.get("/")
def home():
    return FileResponse("index.html")

def preprocess(f: list) -> np.ndarray:
    """
    Map 18 UI inputs → 25 model features.

    UI input order (matches index.html fields f1–f18):
      f1  = Gender              (Male / Female)
      f2  = Partner             (Yes / No)
      f3  = Dependents          (Yes / No)
      f4  = Phone Service       (Yes / No)
      f5  = Multiple Lines      (Yes / No / No phone service)
      f6  = Internet Service    (DSL / Fiber optic / No)
      f7  = Online Security     (Yes / No / No internet service)
      f8  = Online Backup       (Yes / No / No internet service)
      f9  = Device Protection   (Yes / No / No internet service)
      f10 = Tech Support        (Yes / No / No internet service)
      f11 = Streaming TV        (Yes / No / No internet service)
      f12 = Streaming Movies    (Yes / No / No internet service)
      f13 = Contract            (Month-to-month / One year / Two year)
      f14 = Paperless Billing   (Yes / No)
      f15 = Payment Method      (Electronic check / Mailed check /
                                  Bank transfer (automatic) /
                                  Credit card (automatic))
      f16 = Tenure (months)     (numeric)
      f17 = Monthly Charges     (numeric)
      f18 = Total Charges       (numeric)

    NOTE: SeniorCitizen is NOT in the UI — it defaults to 0 (No).
    Add a SeniorCitizen field to index.html if you want to collect it.

    Model feature order (25 columns, must match training exactly):
      0  gender
      1  SeniorCitizen
      2  Partner
      3  Dependents
      4  tenure
      5  PhoneService
      6  PaperlessBilling
      7  MonthlyCharges
      8  TotalCharges
      9  MultipleLines_Yes
      10 InternetService_Fiber optic
      11 InternetService_No
      12 OnlineSecurity_Yes
      13 OnlineBackup_Yes
      14 DeviceProtection_Yes
      15 TechSupport_Yes
      16 StreamingTV_Yes
      17 StreamingMovies_Yes
      18 Contract_One year
      19 Contract_Two year
      20 PaymentMethod_Credit card (automatic)
      21 PaymentMethod_Electronic check
      22 PaymentMethod_Mailed check
      23 No_internet_service
      24 No_phone_service
    """

    # ── helpers ──────────────────────────────────────────────────────────
    yn   = lambda v: 1 if v == "Yes" else 0
    eq   = lambda v, t: 1 if v == t else 0
    no_i = lambda v: 1 if v == "No internet service" else 0

    gender           = 1 if f[0] == "Male" else 0
    senior_citizen   = int(f[18]) if len(f) > 18 else 0   # appended by UI
    partner          = yn(f[1])
    dependents       = yn(f[2])
    tenure           = float(f[15])
    phone_service    = yn(f[3])
    paperless        = yn(f[13])
    monthly_charges  = float(f[16])
    total_charges    = float(f[17])

    # MultipleLines
    multi_yes        = eq(f[4], "Yes")
    no_phone_svc     = eq(f[4], "No phone service")

    # InternetService
    inet_fiber       = eq(f[5], "Fiber optic")
    inet_no          = eq(f[5], "No")

    # Add-ons: "No internet service" on ANY of these → shared flag
    no_inet_svc      = (
        no_i(f[6]) or no_i(f[7]) or no_i(f[8]) or
        no_i(f[9]) or no_i(f[10]) or no_i(f[11])
    )
    no_inet_svc      = 1 if no_inet_svc else 0

    online_sec_yes   = eq(f[6],  "Yes")
    online_bk_yes    = eq(f[7],  "Yes")
    dev_prot_yes     = eq(f[8],  "Yes")
    tech_sup_yes     = eq(f[9],  "Yes")
    stream_tv_yes    = eq(f[10], "Yes")
    stream_mv_yes    = eq(f[11], "Yes")

    # Contract
    contract_1y      = eq(f[12], "One year")
    contract_2y      = eq(f[12], "Two year")

    # PaymentMethod (Electronic check was baseline → dropped during training)
    pay_credit       = eq(f[14], "Credit card (automatic)")
    pay_electronic   = eq(f[14], "Electronic check")
    pay_mailed       = eq(f[14], "Mailed check")

    row = [
        gender, senior_citizen, partner, dependents,
        tenure, phone_service, paperless,
        monthly_charges, total_charges,
        multi_yes,
        inet_fiber, inet_no,
        online_sec_yes, online_bk_yes, dev_prot_yes,
        tech_sup_yes, stream_tv_yes, stream_mv_yes,
        contract_1y, contract_2y,
        pay_credit, pay_electronic, pay_mailed,
        no_inet_svc, no_phone_svc,
    ]

    return np.array(row, dtype=float).reshape(1, -1)


@app.post("/predict")
def predict(data: Features):
    print("Received features:", data.features)  # 👈 see what arrives
    print("Feature count:", len(data.features))
    
    try:
        arr = preprocess(data.features)
        print("Preprocessed shape:", arr.shape)  # should be (1, 25)
        
        prediction = model.predict(arr)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(arr)[0][1])

        return {"prediction": int(prediction), "confidence": confidence}
    
    except Exception as e:
        print("ERROR:", e)        # 👈 see the actual error
        return {"error": str(e)}