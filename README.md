# Bike Rental Prediction

## Project Overview

The Bike Rental Prediction project is designed to predict the number of bike rentals based on various factors like weather, time of day, and holidays. Using a dataset that includes features such as temperature, humidity, windspeed, and working day status, the model uses machine learning techniques to forecast bike rental demand.

This project aims to provide valuable insights for bike rental businesses, helping them optimize their operations based on predicted rental trends.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Description](#model-description)
- [Requirements](#requirements)
- [License](#license)

## Installation

To get started with the project, clone the repository and set up the environment:

### Step 1: Clone the Repository
```bash
git clone https://github.com/alishad846/Bike_rentals_prediction.git
cd Bike_rentals_prediction
Step 2: Create and Activate Virtual Environment
For Python 3.x, create a virtual environment to manage dependencies:

bash
Copy
Edit
python -m venv bike_rental_env
Activate the virtual environment:

Windows:

bash
Copy
Edit
.\bike_rental_env\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source bike_rental_env/bin/activate
Step 3: Install Dependencies
Install the required libraries using pip:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Running the Project
Once the environment is set up and dependencies are installed, you can start the project. The following commands can be used to run the Streamlit app or train the model.

1. Start the Streamlit Web Application:
bash
Copy
Edit
streamlit run app.py
This will start a local web server and you can view the app in your browser by going to http://localhost:8501.

2. Training the Model:
To train the machine learning model, use the following command:

bash
Copy
Edit
python model/train.py
This will train the model on the provided dataset and save the model for later use.

Project Structure
bash
Copy
Edit
Bike_rentals_prediction/
│
├── bike_rental_env/          # Virtual environment (not tracked in Git)
│
├── data/                     # Data files (e.g., CSVs)
│   └── bike_rental_data.csv
│
├── model/                    # Model training scripts
│   ├── train.py
│   └── model.py
│
├── app.py                    # Streamlit app
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
Model Description
The model used in this project predicts the number of bike rentals based on the following features:

season: The season of the year (spring, summer, fall, winter)

year: The year (2011 or 2012)

month: The month of the year

hour: The hour of the day

weather: The weather condition (clear, mist, cloudy, etc.)

temp: The temperature in Celsius

atemp: The "feels like" temperature in Celsius

humidity: The humidity level

windspeed: The wind speed in km/h

holiday: Whether it is a holiday or not (binary)

working day: Whether it is a working day or not (binary)

The project uses TensorFlow to build a regression model that learns from historical data and predicts future bike rental demand.

Requirements
To run this project, you need the following Python version and libraries:

Python 3.12.4

TensorFlow 2.17.0

Keras 3.2.1

Streamlit

pandas

numpy

scikit-learn

matplotlib

You can install the required libraries with:

bash
Copy
Edit
pip install -r requirements.txt
