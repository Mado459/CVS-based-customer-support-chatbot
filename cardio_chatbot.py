import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import joblib

# Load LLM components (BioBERT model and tokenizer)
bio_bert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bio_bert_model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load pre-trained prediction model (RandomForestClassifier) and scaler
scaler = joblib.load('scaler.joblib')  # Assumes you have saved the scaler
rf_model = joblib.load('rf_model.joblib')  # Assumes you have saved the RF model

# Define query function for BioBERT
def query_biobert(user_input):
    inputs = bio_bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bio_bert_model(**inputs)
    logits = outputs.last_hidden_state.mean(dim=1)
    return logits

# Define prediction function for RF model
def predict_health_risk(input_data):
    scaled_data = scaler.transform([input_data])
    prediction = rf_model.predict(scaled_data)
    return prediction[0]

# Define recommendation function
def generate_recommendation(prediction):
    if prediction == 1:
        return "It is recommended to consult a healthcare provider for further cardiovascular risk assessment and management."
    else:
        return "Keep up with your current health habits. Continue regular check-ups to monitor cardiovascular health."

# Streamlit App
st.title("Cardiovascular Health Chatbot")
st.write("This chatbot helps assess cardiovascular risk based on user input and provides recommendations.")

# User input
user_query = st.text_input("Enter your health-related question or information:", "")

# Input fields for numerical data if user wants an assessment
st.write("Provide additional health details for an accurate assessment.")
age = st.number_input("Age:", min_value=0)
height = st.number_input("Height (in cm):", min_value=0)
weight = st.number_input("Weight (in kg):", min_value=0)
cholesterol = st.selectbox("Cholesterol Level:", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
gluc = st.selectbox("Glucose Level:", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
ap_hi = st.number_input("Systolic blood pressure (ap_hi):", min_value=0)
ap_lo = st.number_input("Diastolic blood pressure (ap_lo):", min_value=0)

# Action button
if st.button("Get Assessment and Recommendation"):
    # Ensure that all numeric values are provided
    if not user_query.strip():
        st.write("Please enter a query for assessment.")
    elif age == 0 or height == 0 or weight == 0 or ap_hi == 0 or ap_lo == 0:
        st.write("Please provide complete health details for assessment.")
    else:
        # Run LLM for text-based cardiovascular analysis
        bio_response = query_biobert(user_query)
        st.write("**BioBERT Cardiovascular Analysis:**")
        st.write(f"Risk assessment logits: {bio_response}")

        # Prediction model
        input_data = [age, height, weight, cholesterol, gluc, ap_hi, ap_lo]
        prediction = predict_health_risk(input_data)
        prediction_text = "High Cardiovascular Risk" if prediction == 1 else "Low Cardiovascular Risk"
        st.write(f"**Prediction Model Output:** {prediction_text}")

        # Health recommendation
        recommendation = generate_recommendation(prediction)
        st.write("**Recommendation:**")
        st.write(recommendation)

