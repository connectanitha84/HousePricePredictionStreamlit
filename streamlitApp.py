import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load saved model and preprocessors
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
# encoder = pickle.load(open("encoder.pkl", "rb"))

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below to get the estimated price:")

# ----------------------------
# User input form
# ----------------------------
with st.form("prediction_form"):
    area = st.number_input("Area (sq ft):", min_value=100, max_value=10000, step=50, value=1000)
    bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1, value=2)
    bathrooms = st.number_input("Number of Bathrooms:", min_value=1, max_value=10, step=1, value=1)
    location = st.selectbox("Location:", ["downtown", "suburb"])
    if location=='downtown':
        location_downtown = 1
        location_suburb = 0

    else:
        location_downtown = 0
        location_suburb = 1
    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    # 1️⃣ Create input DataFrame

    input_df = pd.DataFrame([[area, bedrooms, bathrooms, location_downtown,location_suburb]],
                            columns=['area', 'bedrooms', 'bathrooms', 'location_downtown','location_suburb'])

    # 2️⃣ Preprocess numeric features
    numeric_features = ['area', 'bedrooms', 'bathrooms']
    # categorical_features =['location']

    # Scale numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])


    # Encode categorical features
    # input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)

    # input_df = input_df.reindex(columns=X_preprocessed.columns, fill_value=0)

    # 5️⃣ Predict price
    predicted_price = model.predict(input_df)[0]

    # 6️⃣ Display results
    st.markdown("---")
    st.success(f"💰 Estimated House Price: ₹{predicted_price:,.0f}")
    st.write("**Entered Details:**")
    st.write(f"Area: {area} sq ft")
    st.write(f"Bedrooms: {bedrooms}")
    st.write(f"Bathrooms: {bathrooms}")
    st.write(f"Location: {location}")
