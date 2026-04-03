import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load data
df = pd.read_csv('C:\\Users\\tejas\\OneDrive\\Desktop\\RTP\\Data_Train.csv')

# -----------------------------
# PREPROCESSING
# -----------------------------

# Date
df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.day
df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.month
df.drop(['Date_of_Journey'], axis=1, inplace=True)

# Dep time
df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
df.drop(['Dep_Time'], axis=1, inplace=True)

# Arrival time
df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop(['Arrival_Time'], axis=1, inplace=True)

# Duration
df['Duration_hour'] = df['Duration'].str.extract(r'(\d+)h').fillna(0).astype(int)
df['Duration_min'] = df['Duration'].str.extract(r'(\d+)m').fillna(0).astype(int)
df.drop(['Duration'], axis=1, inplace=True)

# Stops
df['Total_Stops'] = df['Total_Stops'].replace({
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
})

# Drop unwanted
df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

# Encoding
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# TRAIN
# -----------------------------

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# -----------------------------
# SAVE
# -----------------------------

pickle.dump(model, open('flight_model.pkl', 'wb'))

print("✅ Model trained & saved!")