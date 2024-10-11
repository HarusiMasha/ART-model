import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Sample data
num_samples = 100  # Total number of samples
data = {
    'Age Group': ['0-14', '15-19', '20-34', '35-49', '50+'] * (num_samples // 5),
    'Sex': ['M', 'F'] * (num_samples // 2),
    'Population Type': ['General Population', 'Key Population'] * (num_samples // 2),
    'Last VL Result': ['LDL', 'min_value=0', 'max_value=50000'] * (num_samples // 3),
    'Active in OVC': ['Yes', 'No'] * (num_samples // 2),
    'VL Category': ['Supressed', 'Non-Suppressed'] * (num_samples // 2)  # Target variable
}

# Ensure all arrays are the same length
for key, value in data.items():
    remainder = num_samples - len(value)
    if remainder > 0:
        data[key].extend(value[:remainder])

df = pd.DataFrame(data)

# Encode categorical features for prediction
le_age = LabelEncoder()
le_sex = LabelEncoder()
le_population = LabelEncoder()
le_last_vl = LabelEncoder()
le_active_ovc = LabelEncoder()
le_vl_category = LabelEncoder()  # For target variable encoding

df['Age Group'] = le_age.fit_transform(df['Age Group'])
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Population Type'] = le_population.fit_transform(df['Population Type'])
df['Last VL Result'] = le_last_vl.fit_transform(df['Last VL Result'])
df['Active in OVC'] = le_active_ovc.fit_transform(df['Active in OVC'])
df['VL Category'] = le_vl_category.fit_transform(df['VL Category'])  # Encode target variable

# Prepare features and target variable
X = df.drop('VL Category', axis=1)  # Features
y = df['VL Category']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI for user input
st.title("VL Category Prediction")

# User inputs using Streamlit widgets
age_group = st.selectbox("Select Age Group:", options=['0-14', '15-19', '20-34', '35-49', '50+'])
sex = st.selectbox("Select Sex:", options=['M', 'F'])
population_type = st.selectbox("Select Population Type:", options=['General Population', 'Key Population'])

# Slider for Last VL Result with "LDL" corresponding to 0
last_vl_result = st.slider("Select Last VL Result (0 for LDL, 0 - 50000):", min_value=0, max_value=50000, value=0)

active_in_ovc = st.selectbox("Active in OVC:", options=['Yes', 'No'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age Group': [le_age.transform([age_group])[0]], 
    'Sex': [le_sex.transform([sex])[0]],
    'Population Type': [le_population.transform([population_type])[0]],
    'Last VL Result': [last_vl_result],  # Use the slider value directly
    'Active in OVC': [le_active_ovc.transform([active_in_ovc])[0]]
})

# Make prediction with error handling
try:
    prediction = model.predict(input_data)
    predicted_label = le_vl_category.inverse_transform(prediction)[0]  # Inverse transform for readable output
except ValueError as e:
    st.error(f"Prediction error: {str(e)}")
    predicted_label = "Unknown"  # Default value if there's an error

# Display the prediction result
st.subheader("Prediction Result")
st.write(f"The predicted value for 'VL Category' is: {predicted_label}")
