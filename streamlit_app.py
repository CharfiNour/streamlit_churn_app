import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page config FIRST
st.set_page_config(page_title="Churn Prediction", page_icon="🔮")

# Load model safely
model = None
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please ensure 'model.pkl' is in the app directory.")

# Load dataset to get label encodings for categorical features
df = None
if os.path.exists("Expresso_churn_dataset.csv"):
    df = pd.read_csv("Expresso_churn_dataset.csv")
else:
    st.error("Dataset file not found. Please ensure 'Expresso_churn_dataset.csv' is in the app directory.")

def get_label_dict(column):
    unique_vals = sorted(df[column].dropna().unique())
    return {val: i for i, val in enumerate(unique_vals)}

st.title("🔮 Customer Churn Prediction")
st.markdown(
    "Enter the customer features below to predict the likelihood of churn. "
    "This model uses six key features."
)

if model is not None and df is not None:
    # Get label encodings for categorical features
    region_dict = get_label_dict("REGION")
    toppack_dict = get_label_dict("TOP_PACK")

    col1, col2 = st.columns(2)
    with col1:
        REGULARITY = st.number_input("REGULARITY", min_value=0.0)
        st.markdown("**Active for 90 days straight**")
        FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", min_value=0.0)
        st.markdown("**Top pack packages activations count**")
        REGION = st.selectbox("REGION", list(region_dict.keys()))
        st.markdown("**The location of the client**")
    with col2:
        ORANGE = st.number_input("ORANGE", min_value=0.0)
        st.markdown("**Call to Orange**")
        TIGO = st.number_input("TIGO", min_value=0.0)
        st.markdown("**Call to Tigo**")
        TOP_PACK = st.selectbox("TOP_PACK", list(toppack_dict.keys()))
        st.markdown("**The most active packs**")

    # Prepare input in the same order as model training
    input_data = np.array([[REGULARITY, FREQ_TOP_PACK, ORANGE, TIGO, REGION_enc, TOP_PACK_enc]])

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)
        label = "✅ Churn" if prediction[0] == 1 else "❌ No Churn"
        st.success(f"Prediction: {label}")
