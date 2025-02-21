import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load data and define target variable (swing)
df = pd.read_csv("devers_data.csv")
df["swing"] = df["description"].isin(["swinging_strike", "foul", "hit_into_play"]).astype(int)

pitch_type_mapping = {
    "CH": "Changeup", "CU": "Curveball", "FC": "Cutter", "FF": "Four-seam Fastball",
    "FS": "Splitter", "KC": "Knuckle-curve", "KN": "Knuckleball", "SI": "Sinker",
    "SL": "Slider", "ST": "Sweeper", "SV": "Slurve"
}

df["pitch_type"] = df["pitch_type"].map(pitch_type_mapping)

# desired features
features = ["pitch_type", "plate_x", "plate_z", "release_speed", "spin_axis", 
            "balls", "strikes", "inning", "outs_when_up", "bat_score", "fld_score"]

df = df.dropna(subset=features + ["swing"])

# data processing
label_encoders = {}
for col in ["pitch_type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 

X = df[features]
y = df["swing"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model trianing - random forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# EVERYTHING BELOW IS UI
st.title("Rafael Devers Swing Analysis")

# define inputs
st.sidebar.header("Swing Probability Prediction")
pitch_type = st.sidebar.selectbox("Pitch Type", label_encoders["pitch_type"].classes_)
plate_x = st.sidebar.slider("Horizontal Location (ft. from center of strike zone)", -2.0, 2.0, 0.0)
plate_z = st.sidebar.slider("Vertical Location (ft. above ground)", 0.0, 4.0, 2.0)
release_speed = st.sidebar.slider("Pitch Speed (MPH)", 70, 100, 90)
spin_axis = st.sidebar.slider("Spin Axis", 0, 360, 180)
balls = st.sidebar.slider("Balls Count", 0, 3, 0)
strikes = st.sidebar.slider("Strikes Count", 0, 2, 0)
inning = st.sidebar.slider("Inning", 1, 9, 1)
outs_when_up = st.sidebar.slider("Outs", 0, 2, 0)
bat_score = st.sidebar.slider("Batting Team Score", 0, 20, 0)
fld_score = st.sidebar.slider("Fielding Team Score", 0, 20, 0)

input_data = pd.DataFrame([[pitch_type, plate_x, plate_z, release_speed, spin_axis, 
                            balls, strikes, inning, outs_when_up, bat_score, fld_score]], 
                          columns=features)

input_data["pitch_type"] = label_encoders["pitch_type"].transform([pitch_type])[0]

swing_prob = rf_model.predict_proba(input_data)[0][1]
st.sidebar.write(f"### Swing Probability: {swing_prob:.2%}")

# HEATMAP VISUALIZATION

st.header("Swing Heatmap")

# options
swing_filter = st.selectbox("Show pitches where Devers:", 
                            ["All Pitches", "Swung", "Did Not Swing", "Hit a Home Run",
                             "Recorded a Base Hit", "Called Strike", "Swinging Strike"])

filtered_df = df.copy()

if swing_filter == "Swung":
    filtered_df = filtered_df[filtered_df["swing"] == 1]
elif swing_filter == "Did Not Swing":
    filtered_df = filtered_df[filtered_df["swing"] == 0]
elif swing_filter == "Hit Home Run":
    filtered_df = filtered_df[filtered_df["events"] == "home_run"]
elif swing_filter == "Recorded a Base Hit":
    filtered_df = filtered_df[filtered_df["events"].isin(["single", "double", "triple", "home_run"])]
elif swing_filter == "Called Strike":
    filtered_df = filtered_df[filtered_df["description"] == "called_strike"]
elif swing_filter == "Swinging Strike":
    filtered_df = filtered_df[filtered_df["description"] == "swinging_strike"]

# make heatmap with strike zone box
fig, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(data=filtered_df, x="plate_x", y="plate_z", cmap="Reds", fill=True, levels=50, alpha=0.7, ax=ax)

ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2, edgecolor="black", fill=False, lw=2))

ax.set_xlim(-2, 2)
ax.set_ylim(0, 4)
ax.set_xlabel("Horizontal Location (plate_x)")
ax.set_ylabel("Vertical Location (plate_z)")
ax.set_title(f"Heatmap of Pitches - {swing_filter}")

st.pyplot(fig)
