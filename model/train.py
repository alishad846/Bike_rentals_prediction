import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the trained model and scaler
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: scaler.pkl not found. Make sure it's in the same directory.")
    st.stop()

try:
    model = tf.keras.models.load_model('bike_rental_model.h5')
except FileNotFoundError:
    st.error("Error: bike_rental_model.h5 not found. Make sure it's in the same directory.")
    st.stop()

# Streamlit app setup
st.title("Bike Rental Prediction")
st.write("This app predicts bike rental counts based on weather and time-related features.")

# Input form for user to enter values
season = st.selectbox("Season", [1, 2, 3, 4])  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
year = st.selectbox("Year", [2011, 2012])
month = st.selectbox("Month", list(range(1, 13)))
hour = st.selectbox("Hour", list(range(0, 24)))
weather = st.selectbox("Weather", [1, 2, 3])  # 1=Clear, 2=Cloudy, 3=Rainy
temp = st.slider("Temperature (C)", min_value=-10.0, max_value=40.0, value=20.0)
atemp = st.slider("Apparent Temperature (C)", min_value=-10.0, max_value=40.0, value=20.0)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
windspeed = st.slider("Windspeed (m/s)", min_value=0.0, max_value=50.0, value=5.0)
holiday = st.selectbox("Holiday", [0, 1])  # 0=No, 1=Yes
working_day = st.selectbox("Working Day", [0, 1])  # 0=No, 1=Yes

# Collect all input features into a DataFrame
input_df = pd.DataFrame([[season, year, month, hour, weather, temp, atemp, humidity, windspeed, holiday, working_day]],
                          columns=['season', 'yr', 'mnth', 'hr', 'weathersit', 'temp', 'atemp', 'humidity', 'windspeed', 'holiday', 'workingday'])

# Identify categorical columns
categorical_cols = ['season', 'yr', 'mnth', 'hr', 'weathersit', 'holiday', 'workingday']
numerical_cols = ['temp', 'atemp', 'humidity', 'windspeed']

# Apply one-hot encoding to categorical columns
encoded_data = pd.get_dummies(input_df[categorical_cols], drop_first=True)

# Concatenate the encoded data with the numerical features
input_data_processed = pd.concat([input_df[numerical_cols].reset_index(drop=True), encoded_data.reset_index(drop=True)], axis=1)

# Print the column names after processing (for debugging)
print("Processed input columns (after encoding and concatenation):", input_data_processed.columns)

# Reindex the processed input data to match the columns used during training (scaler)
input_data_processed = input_data_processed.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Print the columns after reindexing (for verification)
print("Reindexed input columns:", input_data_processed.columns)

# Scale the input data using the scaler that was fitted during training
input_data_scaled = scaler.transform(input_data_processed)

# Make the prediction using the trained model
prediction = model.predict(input_data_scaled)

# Display the prediction result
st.write(f"Predicted bike rental count: {prediction[0][0]:.2f}")