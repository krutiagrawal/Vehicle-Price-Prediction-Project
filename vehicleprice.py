import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the saved model and preprocessor
model_path = 'C:/Users/Kruti Agrawal/Desktop/Projects/vehicle_price_prediction/model.h5'
preprocessor_path = 'C:/Users/Kruti Agrawal/Desktop/Projects/vehicle_price_prediction/preprocessor.pkl'

model = load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

# Function to predict vehicle price
def predict_price(input_data):
    # Transform input data using the preprocessor
    transformed_data = preprocessor.transform(input_data)
    # Make prediction
    price = model.predict(transformed_data)
    return price[0][0]

# Streamlit app layout
st.title("Vehicle Price Prediction")
st.write("This app predicts the price of a vehicle based on its specifications.")

# User input fields
make = st.selectbox("Make", ["Ford", "Toyota", "BMW", "Chevrolet", "Honda", "Nissan"])  # Add more options as needed
model_name = st.text_input("Model")
year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
cylinders = st.number_input("Cylinders", min_value=1, max_value=12, value=4)
fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
mileage = st.number_input("Mileage (in miles)", min_value=0, value=15000)
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
trim = st.text_input("Trim")
body = st.selectbox("Body Style", ["SUV", "Sedan", "Pickup Truck", "Coupe", "Hatchback"])  # Add more options as needed
doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)
exterior_color = st.text_input("Exterior Color")
interior_color = st.text_input("Interior Color")
drivetrain = st.selectbox("Drivetrain", ["All-wheel Drive", "Front-wheel Drive", "Rear-wheel Drive"])

# Collect input data into a DataFrame
input_data = pd.DataFrame({
    "make": [make],
    "model": [model_name],
    "year": [year],
    "cylinders": [cylinders],
    "fuel": [fuel],
    "mileage": [mileage],
    "transmission": [transmission],
    "trim": [trim],
    "body": [body],
    "doors": [doors],
    "exterior_color": [exterior_color],
    "interior_color": [interior_color],
    "drivetrain": [drivetrain]
})

exchange_rate = 83

# Button to trigger prediction
if st.button("Predict Price"):
    price = predict_price(input_data)
    st.write(f"The predicted price of the vehicle is: ${price:,.2f}")
    predicted_price_inr = price * exchange_rate
    st.success(f'Predicted Price: ₹{predicted_price_inr:,.2f} INR')


