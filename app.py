import streamlit as st
import pickle

with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2, feature3]])
    st.success(f"Prediction: {prediction[0]}")
