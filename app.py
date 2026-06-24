import streamlit as st
import pickle
import os

st.title("Heart Disease Prediction")

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    st.success("Model loaded successfully!")

except FileNotFoundError:
    st.error(
        f"model.pkl not found.\n\nFiles in current directory:\n{os.listdir('.')}"
    )
    st.stop()

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2, feature3]])
    st.success(f"Prediction: {prediction[0]}")
