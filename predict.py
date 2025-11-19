import torch
import numpy as np
import joblib
from brainrot_model import BrainrotNet

# Load model and scaler 
model = BrainrotNet()
model.load_state_dict(torch.load("brainrot_model.pt"))
model.eval()
scaler = joblib.load("scaler.pkl")

# User Inputs (HOURS → MINUTES)
screen_time_hours = float(input("Screen time per day (in hours): "))
sleep_hours = float(input("Sleep hours: "))
study_hours = float(input("Study hours: "))
phone_pickups = float(input("Phone pickups per day: "))
music_time_hours = float(input("Music time (in hours): "))

# Convert hours → minutes
screen_time = screen_time_hours * 60
music_time = music_time_hours * 60

# Prepare data
your_data = np.array([[screen_time, sleep_hours, study_hours, phone_pickups, music_time]])
scaled = scaler.transform(your_data)
input_tensor = torch.tensor(scaled, dtype=torch.float32)

# Predict
pred = model(input_tensor).item()

# Label
if pred < 20:
    label = "Zen Monk Vibes 🧘‍♀️ — you're clean."
elif pred < 40:
    label = "Casual Doomscroller 😌 — watch yourself."
elif pred < 60:
    label = "Moderate Brainrot 🤳 — risky!"
elif pred < 80:
    label = "Severe Scrolling Disorder 💀 — put the phone DOWN."
else:
    label = "Certified Brainrotted & Gone 🧟‍♀️ — seek help."

print(f"\nPredicted Brainrot Score: {pred:.2f}/100")
print(label)
