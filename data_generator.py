import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "screen_time": np.random.randint(60, 600, 100), #minutes on phone
    "sleep_hours": np.random.uniform(4, 9, 100),
    "study_hours": np.random.uniform(0, 6, 100),
    "phone_pickups": np.random.randint(30, 200, 100),
    "music_time": np.random.uniform(0, 180, 100),
}

df = pd.DataFrame(data)

#Brainrot score (higher = more doomscrolling)

df["brainrot_score"] = (
    0.5 * (df["screen_time"] / 600)
    + 0.3 * (df["phone_pickups"] / 200)
    -0.2 * (df["study_hours"] / 6)
    -0.1 * (df["sleep_hours"] / 9)
) * 100
df["brainrot_score"] = df["brainrot_score"].clip(0,100)

df.to_csv("data/habits.csv",  index=False)
print("Dummy data created!")