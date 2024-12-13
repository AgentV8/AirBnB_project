import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cloudpickle
import tensorflow as tf
from tensorflow.keras.models import load_model

@tf.keras.utils.register_keras_serializable()
class IndexLayer(tf.keras.layers.Layer):
    def __init__(self, index=None, indices=None, **kwargs):
        super(IndexLayer, self).__init__(**kwargs)
        self.index = index
        self.indices = indices

    def call(self, inputs):
        if self.index is not None:
            return inputs[:, :, self.index]
        elif self.indices is not None:
            return tf.gather(inputs, self.indices, axis=2)
        else:
            raise ValueError("Either 'index' or 'indices' must be provided.")

    def compute_output_shape(self, input_shape):
        batch_size, seq_length, _ = input_shape
        if self.index is not None:
            return (batch_size, seq_length)
        elif self.indices is not None:
            return (batch_size, seq_length, len(self.indices))


# -----------------------------
# File Paths (Update these paths as needed)
# -----------------------------
CSV_PATH = r"C:\Users\prabh\Downloads\New folder (2)\sequential_listings_12000.csv"
SCALER_PATH = r"C:\Users\prabh\Downloads\New folder (3)\scaler.joblib"
NEIGHBOURHOOD_ENCODER_PATH = r"C:\Users\prabh\Downloads\New folder (3)\label_encoder_neighbourhood_cleansed.joblib"
PROPERTY_ENCODER_PATH = r"C:\Users\prabh\Downloads\New folder (3)\label_encoder_property_type.joblib"
MODEL_PKL_PATH = r"C:\Users\prabh\Downloads\New folder (3)\best_lstm_regressor.pkl"

SEQ_LENGTH = 3  # Sequence length used during training

# -----------------------------
# Load Preprocessing Objects
# -----------------------------
scaler = joblib.load(SCALER_PATH)
neighbourhood_le = joblib.load(NEIGHBOURHOOD_ENCODER_PATH)
property_type_le = joblib.load(PROPERTY_ENCODER_PATH)

# -----------------------------
# Load the Trained Model
# -----------------------------
with open(MODEL_PKL_PATH, 'rb') as f:
    best_regressor = cloudpickle.load(f)

# -----------------------------
# Load and Sort Data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df = df.sort_values(by=['id', 'data_year', 'data_month'])

# Create a copy of the raw data before any preprocessing
df_raw = df.copy()

# -----------------------------
# Define Feature Columns
# -----------------------------
exclude_cols = ['id', 'price', 'data_year', 'data_month']
feature_columns = [col for col in df.columns if col not in exclude_cols]

categorical_cols = ['neighbourhood_cleansed', 'property_type']
# Recompute numerical_cols in the same way as training
numerical_cols = df[feature_columns].select_dtypes(include=['int16','float16','int64','float32','float64']).columns.tolist()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Airbnb Price Prediction Demo")
st.write("Predict next month's price for a given listing after setting `availability_30` to 3.")

# Dropdown to select Listing ID
unique_ids = df['id'].unique()
listing_id = st.selectbox("Select a Listing ID:", options=unique_ids)

# Button to trigger prediction
if st.button("Predict Next Month's Price"):
    # Filter data for the selected Listing ID
    listing_data = df[df['id'] == listing_id].sort_values(by=['data_year', 'data_month'])
    
    # Check if there's enough historical data
    if len(listing_data) < SEQ_LENGTH:
        st.error("Not enough historical data for this listing.")
    else:
        # Prepare the last SEQ_LENGTH months of data
        input_seq = listing_data.iloc[-SEQ_LENGTH:].copy()

        # -----------------------------
        # Encode Categorical Features
        # -----------------------------
        if input_seq['neighbourhood_cleansed'].dtype not in ['int16','int32','int64']:
            input_seq['neighbourhood_cleansed'] = neighbourhood_le.transform(input_seq['neighbourhood_cleansed'])

        if input_seq['property_type'].dtype not in ['int16','int32','int64']:
            input_seq['property_type'] = property_type_le.transform(input_seq['property_type'])

        # -----------------------------
        # Scale Numerical Features
        # -----------------------------
        # Ensure numerical_cols are consistent with training
        input_seq[numerical_cols] = scaler.transform(input_seq[numerical_cols])

        # -----------------------------
        # Modify availability_30
        # -----------------------------
        if 'availability_30' in feature_columns:
            input_seq.loc[input_seq.index[-1], 'availability_30'] = 3
            # Re-scale the last row if availability_30 is in numerical_cols
            if 'availability_30' in numerical_cols:
                temp_row = input_seq.loc[input_seq.index[-1], numerical_cols].to_frame().T
                input_seq.loc[input_seq.index[-1], numerical_cols] = scaler.transform(temp_row)[0]
        else:
            st.warning("'availability_30' not found in feature_columns.")

        # -----------------------------
        # Prepare Input for Prediction
        # -----------------------------
        X_input = np.array(input_seq[feature_columns].values).reshape(1, SEQ_LENGTH, len(feature_columns))

        # -----------------------------
        # Make Prediction
        # -----------------------------
        y_pred = best_regressor.predict(X_input)
        predicted_price = float(y_pred[0])
        st.success(f"Predicted Next Month's Price for Listing {listing_id}: ${predicted_price:.2f}")


        # -----------------------------
        # Display All Raw Data for Selected Listing ID
        # -----------------------------
        st.subheader("All Data for Selected Listing ID (Raw)")
        raw_listing_data = df_raw[df_raw['id'] == listing_id].sort_values(by=['data_year', 'data_month'])
        st.write(raw_listing_data)
