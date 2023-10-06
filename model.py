import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import requests  # Added for fetching data from GitHub
import io

# Title and description
st.title("Employee Attrition Prediction")
st.write("This app predicts employee attrition using a Random Forest Classifier.")

# GitHub raw CSV URL
github_raw_csv_url = 'https://raw.githubusercontent.com/Swamisharan1/hr-attrition/main/HR-Employee-Attrition.csv'

# Function to fetch data from GitHub
def fetch_data_from_github(url):
    response = requests.get(url)
    return response.text

# Fetch the CSV data from GitHub
csv_data = fetch_data_from_github(github_raw_csv_url)

# Read the CSV data into a DataFrame
try:
    df1 = pd.read_csv(io.StringIO(csv_data))
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Copy the DataFrame
df = df1.copy()

# Drop the 'StandardHours' column
df.drop(columns=['StandardHours'], inplace=True)

# Label encode the 'Attrition' column
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# One-hot encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Drop unnecessary columns
df.drop(columns=['EmployeeCount', 'EmployeeNumber'], inplace=True)

# Split the dataset into features X and target variable y
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split the balanced dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# User input for prediction
st.header("Predict Employee Attrition")
st.write("Enter employee data for prediction:")

# Create input fields for the user to enter data (integer values)
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, step=1)

if st.button("Predict"):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions
    prediction = rf_classifier.predict(input_df)

    # Display the prediction result
    if prediction[0] == 1:
        st.warning("Employee is likely to leave (Attrition: Yes).")
    else:
        st.success("Employee is likely to stay (Attrition: No).")
