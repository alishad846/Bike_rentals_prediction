import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load trained model and scaler
try:
    model = load_model("C:\\Users\\asus\\OneDrive\\Desktop\\Machine learning\\ML projects\\Bike_rental\\model\\bike_rental_model.h5")
except FileNotFoundError:
    st.error("Error: bike_rental_model.h5 not found. Please check the path.")
    st.stop()

try:
    scaler = joblib.load("C:/Users/asus/OneDrive/Desktop/Machine learning/ML projects/Bike_rental/model/scaler.pkl")
except FileNotFoundError:
    st.error("Error: scaler.pkl not found. Please check the path.")
    st.stop()

# Streamlit app layout
st.title("Bike Rental Prediction")
st.write("Predicts bike rental counts based on various features.")

# Get the current year
current_year = datetime.now().year
future_years = list(range(current_year, current_year + 13))  # Current year + next 12

# User-friendly input form with custom titles and explanations (tooltips)
season_title = "Season of the Year"
season_options = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
season_help = "Select the season."
season_display = st.selectbox(season_title, list(season_options.values()), index=0, help=season_help)
season_value = [k for k, v in season_options.items() if v == season_display][0]

year_title = "Year"
year_options = {year: str(year) for year in future_years}
year_help = "Select the year."
year_display = st.selectbox(year_title, list(year_options.values()), index=0 if 2011 in future_years else future_years.index(current_year), help=year_help)
year_value = int(year_display)

month_title = "Month"
month_options = {i: f"{i:02d}" for i in range(1, 13)}
month_help = "Select the month (01-12)."
month_display = st.selectbox(month_title, list(month_options.values()), index=0, help=month_help)
month_value = int(month_display)

hour_title = "Hour of the Day"
hour_options = {i: f"{i:02d}:00" for i in range(0, 24)}
hour_help = "Select the hour (00-23)."
hour_display = st.selectbox(hour_title, list(hour_options.values()), index=0, help=hour_help)
hour_value = int(hour_display.split(':')[0])

weather_title = "Weather Condition"
weather_options = {
    1: "Clear, Few clouds, Partly cloudy, Cloudy",
    2: "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
    3: "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
    4: "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
}
weather_help = "Select the weather condition."
weather_display = st.selectbox(weather_title, list(weather_options.values()), index=0, help=weather_help)
weather_value = [k for k, v in weather_options.items() if v == weather_display][0]

temp_title = "Temperature (°C)"
temp_help = "Enter the temperature in Celsius."
temp = st.slider(temp_title, min_value=-10.0, max_value=40.0, value=20.0, help=temp_help)

atemp_title = "Apparent Temperature (°C)"
atemp_help = "Enter the apparent (feels-like) temperature in Celsius."
atemp = st.slider(atemp_title, min_value=-10.0, max_value=40.0, value=20.0, help=atemp_help)

humidity_title = "Humidity (%)"
humidity_help = "Enter the humidity percentage (0-100)."
humidity = st.slider(humidity_title, min_value=0, max_value=100, value=50, help=humidity_help)

windspeed_title = "Wind Speed (m/s)"
windspeed_help = "Enter the wind speed in meters per second."
windspeed = st.slider(windspeed_title, min_value=0.0, max_value=50.0, value=5.0, help=windspeed_help)

holiday_title = "Holiday"
holiday_options = {0: "No", 1: "Yes"}
holiday_help = "Is it a holiday?"
holiday_display = st.selectbox(holiday_title, list(holiday_options.values()), index=0, help=holiday_help)
holiday_value = [k for k, v in holiday_options.items() if v == holiday_display][0]

working_day_title = "Working Day"
working_day_options = {0: "No", 1: "Yes"}
working_day_help = "Is it a working day (neither weekend nor holiday)?"
working_day_display = st.selectbox(working_day_title, list(working_day_options.values()), index=0, help=working_day_help)
working_day_value = [k for k, v in working_day_options.items() if v == working_day_display][0]

# Collect input features with original numerical values
input_data = pd.DataFrame({
    'season': [season_value],
    'yr': [year_value],
    'mnth': [month_value],
    'hr': [hour_value],
    'weathersit': [weather_value],
    'temp': [temp],
    'atemp': [atemp],
    'hum': [humidity / 100],  # Normalize humidity
    'windspeed': [windspeed / 67],  # Normalize windspeed
    'holiday': [holiday_value],
    'workingday': [working_day_value]
})

if st.button("Predict Bike Rentals"):
    # Define categorical columns
    categorical_cols = ['season', 'yr', 'mnth', 'hr', 'weathersit', 'holiday', 'workingday']

    # One-hot encode categorical features
    input_data_processed = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Reindex to match training data columns
    input_data_processed = input_data_processed.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_processed)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display result
    st.write(f"Predicted bike rental count for {year_value}: {prediction[0][0]:.2f}")