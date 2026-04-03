import streamlit as st
import pandas as pd
import pickle

# Load model & columns
model = pickle.load(open('flight_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

df = pd.read_csv('C:\\Users\\tejas\\OneDrive\\Desktop\\RTP\\Data_Train.csv')

st.title("✈️ Flight Price Prediction")

# Inputs
airline = st.selectbox("Airline", df['Airline'].unique())
source = st.selectbox("Source", df['Source'].unique())
destination = st.selectbox("Destination", df['Destination'].unique())
total_stops = st.selectbox("Total Stops", df['Total_Stops'].unique())
additional_info = st.selectbox("Additional Info", df['Additional_Info'].unique())
# ONLY ONE date input (fixed)
journey_date = st.date_input("Journey Date")

# Time inputs (clean way)
dep_time = st.time_input("Departure Time")
arrival_time = st.time_input("Arrival Time")

# Duration
duration_hour = st.slider("Duration Hour", 0, 20)
duration_min = st.slider("Duration Minute", 0, 59)

# Prediction button
if st.button("Predict Price 💰"):

    # Feature extraction
    journey_day = journey_date.day
    journey_month = journey_date.month

    dep_hour = dep_time.hour
    dep_min = dep_time.minute

    arrival_hour = arrival_time.hour
    arrival_min = arrival_time.minute

    # Create input dictionary
    input_dict = {
        'Journey_day': journey_day,
        'Journey_month': journey_month,
        'Dep_hour': dep_hour,
        'Dep_min': dep_min,
        'Arrival_hour': arrival_hour,
        'Arrival_min': arrival_min,
        'Duration_hour': duration_hour,
        'Duration_min': duration_min,
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Total_Stops': total_stops,
        'Additional_Info': additional_info  
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Remove extra columns
    input_df = input_df[model_columns]
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Flight Price: ₹ {round(prediction, 2)}")