from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_pipeline.joblib"

FEATURES = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
            "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.title("Customer segment prediction")
st.write("Enter a customer profile to predict the segment.")

values = {}
for col in FEATURES:
    values[col] = st.number_input(col, min_value=0.0, value=0.0, step=10.0)

if st.button("Predict"):
    new_customer = pd.DataFrame([values], columns=FEATURES)

    segment = pipeline.predict(new_customer)[0]
    st.subheader(f"Segment: {segment}")
