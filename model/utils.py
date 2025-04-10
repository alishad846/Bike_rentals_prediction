import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to build the model
def build_model(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer (for regression)
    ])
    return model
