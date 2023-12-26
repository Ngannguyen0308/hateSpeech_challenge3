
import streamlit as st
import numpy as np
import torch
from tensorflow.keras.models import load_model
from transformers import AutoModel, AutoTokenizer

# Load PhoBERT model and tokenizer
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load your trained LSTM model
model = load_model("vietnam_model.h5")

# Function to convert text to PhoBERT embedding
def text_to_phobert_embedding(text, max_length=256):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = phobert_model(**tokens)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Streamlit UI
st.title("Hate Speech Detection in Vietnamese")

# User input
user_input = st.text_area("Enter text for detection:", "đuỹ cái")

# Convert input to PhoBERT embedding
sample_embedding = text_to_phobert_embedding(user_input)

# Make prediction
prediction = model.predict(np.array([sample_embedding]))[0, 0]

# Display prediction result
st.write(f"Text: {user_input}")
st.write(f"Prediction: {'Tục tiểu' if prediction > 0.5 else 'Không tục'}")

