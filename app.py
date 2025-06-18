import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('data.csv')
X = data[['Hours']]
y = data['Marks']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Streamlit App
st.title("Student Marks Prediction App")
st.write("## Predict marks based on study hours")

hours = st.number_input("Enter number of study hours:", 0.0, 10.0, step=0.5)

if st.button("Predict"):
    result = model.predict([[hours]])
    st.success(f"Predicted Marks: {result[0]:.2f}")

st.write("## Dataset")
st.dataframe(data)
