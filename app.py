import streamlit as st
import numpy as np
import torch
import joblib
from brainrot_model import BrainrotNet

#Load model and scaler
model = BrainrotNet()
model.load_state_dict(torch.load("brainrot_model.pt"))
model.eval()
scaler = joblib.load("scaler.pkl")

st.title("📱 Brainrot Predictor")
st.write("Find out how cooked your brain is🔥🔥")

#Input fields
screen_time_hours = st.number_input("Screen Time (hours per day)", 0.0, 24.0, 5.0)
music_time_hours = st.number_input("Music Listening Time (hours per day)", 0.0, 24.0, 2.0)

# Unchanged Inputs
sleep_hours = st.number_input("Sleep hours", 0.0, 12.0, 6.5)
study_hours = st.number_input("Study hours", 0.0, 12.0, 2.0)
phone_pickups = st.number_input("Phone pickups per day", 0, 500, 90)

# Convert HOURS → MINUTES for model input
screen_time = screen_time_hours * 60
music_time = music_time_hours * 60

# Prediction Button
if st.button("Predict Brainrot Score"):
    user_data = np.array([[screen_time, sleep_hours, study_hours, phone_pickups, music_time]])
    scaled = scaler.transform(user_data)
    tensor_input = torch.tensor(scaled, dtype=torch.float32)
    pred = model(tensor_input).item()

    #Brainrot Label
    if pred < 20:
        label = "🧘 Zen Monk — your soul is intact."
    elif pred < 40:
        label = "😌 Light Scroller — manageable."
    elif pred < 60:
        label = "🤳 Moderate Brainrot — careful."
    elif pred < 80:
        label = "💀 Severe Brainrot — you need help."
    else:
        label = "🧟 Certified Brainrotted — RIP."

    st.subheader("Your Brainrot Score")
    st.progress(int(pred))  # turns the score into a bar (0-100)
    st.write(f"Score: {pred:.2f}/100")
    st.write(label)
    