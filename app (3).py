
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("fake_news_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Fake vs Real News Prediction")

st.write("Enter the details below:")

# Inputs
title = st.number_input("Title (encoded value)")
author = st.number_input("Author (encoded value)")
credibility = st.slider("Source Credibility Score", 0.0, 1.0)
citations = st.number_input("Number of Citations")
sentiment = st.slider("Sentiment Score", -1.0, 1.0)
clickbait = st.slider("Clickbait Score", 0.0, 1.0)
readability = st.number_input("Readability Score")

if st.button("Predict"):
    sample = np.array([[title, author, credibility, citations, sentiment, clickbait, readability]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    if prediction[0] == 1:
        st.success("This is REAL News")
    else:
        st.error("This is FAKE News")
