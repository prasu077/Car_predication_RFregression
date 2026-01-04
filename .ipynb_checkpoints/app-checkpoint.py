import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("car_price_model.pkl", "rb"))
feature_cols = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.write("Predict the resale price of a car using Machine Learning")

# ---------------- USER INPUT ----------------
milage = st.number_input("Mileage (in miles)", min_value=0, step=1000)
car_age = st.number_input("Car Age (years)", min_value=0, step=1)
engine_hp = st.number_input("Engine Power (HP)", min_value=50, step=10)

brand = st.selectbox(
    "Brand",
    ['Audi','BMW','Bentley','Buick','Bugatti','Alfa','Aston','Ford','Hyundai','Lexus','Other']
)

fuel_type = st.selectbox(
    "Fuel Type",
    ['Gasoline','Diesel','Hybrid','Electric','Other']
)

transmission = st.selectbox(
    "Transmission",
    ['Automatic','Manual']
)

accident = st.selectbox(
    "Accident History",
    ['Yes','No']
)

clean_title = st.selectbox(
    "Clean Title",
    ['Yes','No']
)

# ---------------- CREATE INPUT DATAFRAME ----------------
input_dict = {
    'milage': milage,
    'car_age': car_age,
    'engine_hp': engine_hp,
    'accident_Yes': 1 if accident == 'Yes' else 0,
    'clean_title_Yes': 1 if clean_title == 'Yes' else 0
}

# initialize all feature columns with 0
input_df = pd.DataFrame(columns=feature_cols)
input_df.loc[0] = 0

# fill numeric values
for key, value in input_dict.items():
    if key in input_df.columns:
        input_df.at[0, key] = value

# handle brand
brand_col = f"brand_{brand}"
if brand_col in input_df.columns:
    input_df.at[0, brand_col] = 1

# handle fuel
fuel_col = f"fuel_type_{fuel_type}"
if fuel_col in input_df.columns:
    input_df.at[0, fuel_col] = 1

# handle transmission
trans_col = f"transmission_{transmission}"
if trans_col in input_df.columns:
    input_df.at[0, trans_col] = 1

# ---------------- PREDICTION ----------------
if st.button("Predict Price 💰"):
    log_price = model.predict(input_df)[0]
    price = np.expm1(log_price)

    st.success(f"Estimated Car Price: ₹ {price:,.0f}")
