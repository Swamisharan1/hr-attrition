import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import io

st.title("Employee Attrition Prediction")

# GitHub raw CSV URL
github_raw_csv_url = 'https://raw.githubusercontent.com/Swamisharan1/hr-attrition/main/HR-Employee-Attrition.csv'

# Function to fetch data from GitHub
def fetch_data_from_github(url):
    response = requests.get(url)
    return response.text

# Fetch the CSV data from GitHub
csv_data = fetch_data_from_github(github_raw_csv_url)

# Load data into a DataFrame
try:
    df = pd.read_csv(io.StringIO(csv_data))
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Data preprocessing
labelencoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == object:
        df[column] = labelencoder.fit_transform(df[column])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Check if there's enough data for splitting
if len(df) < 2:
    st.error("Not enough data for splitting. Make sure your dataset contains multiple rows.")
    st.stop()

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
except Exception as e:
    st.error(f"Error splitting data: {str(e)}")
    st.stop()

# Check if there's enough data in the train/test sets
if len(X_train) == 0 or len(X_test) == 0:
    st.error("Not enough data in train/test sets. Check your dataset or splitting parameters.")
    st.stop()

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# User input for prediction
st.header("Predict Employee Attrition")
st.write("Enter employee data for prediction:")

# Create input fields for the user to enter data
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col)

if st.button("Predict"):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    for column in input_df.columns:
        if input_df[column].dtype == object:
            input_df[column] = labelencoder.transform(input_df[column])

    # Make predictions
    prediction = logreg.predict(input_df)

    # Display the prediction result
    if prediction[0] == 1:
        st.warning("Employee is likely to leave (Attrition: Yes).")
    else:
        st.success("Employee is likely to stay (Attrition: No).")
