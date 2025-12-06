import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("bangalore_house_price_prediction_rfr_model.pkl")

# Exact feature list your model expects
model_columns = ['bath', 'balcony', 'total_sqft_int', 'bhk', 'price_per_sqft',
 'area_typeSuper built-up  Area', 'area_typeBuilt-up  Area', 'area_typePlot  Area',
 'availability_Ready To Move', 'location_Whitefield', 'location_Sarjapur  Road',
 'location_Electronic City', 'location_Raja Rajeshwari Nagar', 'location_Haralur Road',
 'location_Marathahalli', 'location_Hennur Road', 'location_Bannerghatta Road',
 'location_Uttarahalli', 'location_Thanisandra', 'location_Electronic City Phase II',
 'location_Hebbal', 'location_7th Phase JP Nagar', 'location_Yelahanka',
 'location_Kanakpura Road', 'location_KR Puram', 'location_Sarjapur',
 'location_Rajaji Nagar', 'location_Kasavanhalli', 'location_Bellandur',
 'location_Begur Road', 'location_Kothanur', 'location_Banashankari',
 'location_Hormavu', 'location_Harlur', 'location_Akshaya Nagar', 'location_Jakkur',
 'location_Electronics City Phase 1', 'location_Varthur', 'location_Chandapura',
 'location_Hennur', 'location_HSR Layout', 'location_Ramamurthy Nagar',
 'location_Kaggadasapura', 'location_Ramagondanahalli', 'location_Kundalahalli',
 'location_Koramangala', 'location_Hoodi', 'location_Budigere', 'location_Hulimavu',
 'location_Malleshwaram', 'location_Hegde Nagar', 'location_JP Nagar',
 'location_8th Phase JP Nagar', 'location_Yeshwanthpur', 'location_Gottigere',
 'location_Bisuvanahalli', 'location_Channasandra', 'location_Indira Nagar',
 'location_Vittasandra', 'location_Old Airport Road', 'location_Hosa Road',
 'location_Kengeri', 'location_Vijayanagar', 'location_Sahakara Nagar',
 'location_Brookefield', 'location_Green Glen Layout', 'location_Balagere',
 'location_Bommasandra', 'location_Kudlu Gate', 'location_Old Madras Road',
 'location_Rachenahalli', 'location_Panathur', 'location_Kadugodi',
 'location_Talaghattapura', 'location_Ambedkar Nagar', 'location_Mysore Road',
 'location_Thigalarapalya', 'location_Jigani', 'location_Yelahanka New Town',
 'location_Attibele', 'location_Dodda Nekkundi', 'location_Frazer Town',
 'location_Devanahalli', 'location_Kanakapura', 'location_Ananth Nagar',
 'location_5th Phase JP Nagar', 'location_TC Palaya', 'location_Anekal',
 'location_Nagarbhavi', 'location_Lakshminarayana Pura', 'location_Kudlu',
 'location_Jalahalli', 'location_Kengeri Satellite Town', 'location_CV Raman Nagar',
 'location_Bhoganhalli', 'location_Horamavu Agara', 'location_Doddathoguru',
 'location_Kalena Agrahara', 'location_Subramanyapura', 'location_Hebbal Kempapura',
 'location_BTM 2nd Stage', 'location_Vidyaranyapura', 'location_Hosur Road',
 'location_Domlur', 'location_Horamavu Banaswadi', 'location_Mahadevpura',
 'location_Tumkur Road'
]

# UI
st.set_page_config(page_title="Bangalore House Price Prediction", layout="wide")
st.title("🏠 Bangalore House Price Prediction")

st.write("Provide details below to estimate the house price (₹ Lakhs)")

# Numeric Inputs
bath = st.number_input("Bathrooms", 1, 10, 2)
balcony = st.number_input("Balcony", 0, 5, 1)
total_sqft_int = st.number_input("Total Sqft", 200, 10000, 1200)
bhk = st.number_input("BHK", 1, 10, 2)
price_per_sqft = st.number_input("Price per Sqft (Average Market)", 100.0, 20000.0, 6000.0)

# Area Type
area_type = st.selectbox("Area Type", [
    "Super built-up  Area", "Built-up  Area", "Plot  Area"
])

# Availability
st.info("Availability fixed to: Ready To Move (only supported by model)")

# Location
location = st.selectbox("Location", sorted([col.replace("location_","") for col in model_columns if col.startswith("location_")]))

# Predict Button
if st.button("Predict Price 💰"):
    # Build input row with zeros
    input_row = pd.DataFrame([[0]*len(model_columns)], columns=model_columns)

    # Fill numeric values
    input_row.loc[0, 'bath'] = bath
    input_row.loc[0, 'balcony'] = balcony
    input_row.loc[0, 'total_sqft_int'] = total_sqft_int
    input_row.loc[0, 'bhk'] = bhk
    input_row.loc[0, 'price_per_sqft'] = price_per_sqft

    # Set selected categorical flags
    input_row[f"area_type{area_type}"] = 1
    input_row["availability_Ready To Move"] = 1
    input_row[f"location_{location}"] = 1

    pred_price = model.predict(input_row)[0]

    st.success(f"🏷 Predicted Price: **₹ {round(pred_price, 2)} Lakhs**")
