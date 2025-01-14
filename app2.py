import streamlit as st
import numpy as np
import pickle
import sqlite3
import pandas as pd

# Load the trained Random Forest model
model = pickle.load(open('final_forest_model.pkl', 'rb'))

# Connect to the SQLite database
conn = sqlite3.connect('predictiondb.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    specific_gravity REAL,
    hypertension INTEGER,
    haemoglobin REAL,
    diabetes_mellitus INTEGER,
    albumin INTEGER,
    red_blood_cell_count REAL,
    packed_cell_volume INTEGER,
    prediction INTEGER
)
''')
conn.commit()

# Streamlit app title
st.title('Chronic Kidney Disease Prediction')

# Input form for the user to enter data
specific_gravity = st.number_input("Specific Gravity", min_value=1.005, max_value=1.030, step=0.001)
hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
haemoglobin = st.number_input("Hemoglobin (g/dl)", min_value=0.0, max_value=20.0, step=0.1)
diabetes_mellitus = st.selectbox("Diabetes Mellitus", options=["Yes", "No"])
albumin = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5])
red_blood_cell_count = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, step=0.1)
packed_cell_volume = st.selectbox("Packed Cell Volume", options=["Normal", "Abnormal"])

# Convert categorical inputs to numeric values
hypertension = 1 if hypertension == "Yes" else 0
diabetes_mellitus = 1 if diabetes_mellitus == "Yes" else 0
packed_cell_volume = 1 if packed_cell_volume == "Abnormal" else 0

# Input data for prediction
input_data = np.array([[specific_gravity, hypertension, haemoglobin, diabetes_mellitus, albumin,
                        red_blood_cell_count, packed_cell_volume]]).astype(np.float64)

# Make the prediction
if st.button("Predict"):
    # Display the input values for debugging
    st.write(f"Input values: {specific_gravity}, {hypertension}, {haemoglobin}, {diabetes_mellitus}, "
             f"{albumin}, {red_blood_cell_count}, {packed_cell_volume}")
    
    prediction = model.predict(input_data)
    st.write(f"Prediction output: {prediction}")  # Debugging the output
    prediction_label = "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"
    
    # Display the result
    st.subheader(f"Prediction: {prediction_label}")

    # Store the inputs and prediction in the database
    c.execute('''INSERT INTO predictions (specific_gravity, hypertension, haemoglobin, diabetes_mellitus, 
                 albumin, red_blood_cell_count, packed_cell_volume, prediction)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                 (specific_gravity, hypertension, haemoglobin, diabetes_mellitus, albumin,
                  red_blood_cell_count, packed_cell_volume, prediction[0]))
    conn.commit()

    # Show stored data from the database
    st.subheader("Stored Predictions")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.dataframe(df)

# Footer
st.markdown("<br><br><p style='text-align: center;'>©2021 Sagar Dhandare</p>", unsafe_allow_html=True)
