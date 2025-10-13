import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# ⚽ App Title
# -----------------------------
st.title("⚽ Football Match Result Predictor")

# -----------------------------
# 🔹 Load Model & Scaler
# -----------------------------
rf = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Try to get feature names used during training
try:
    expected_features = scaler.feature_names_in_
except AttributeError:
    st.error("⚠️ Could not detect scaler feature names. Please ensure sklearn >= 1.0 was used for training.")
    st.stop()

# -----------------------------
# 🧩 Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Match Features")
form_points_diff = st.sidebar.number_input("Form Points Difference", value=0.0)
goal_scored_diff = st.sidebar.number_input("Goal Scored Difference", value=0.0)
goal_conceded_diff = st.sidebar.number_input("Goal Conceded Difference", value=0.0)
home_team_enc = st.sidebar.number_input("Home Team Code", value=0)
away_team_enc = st.sidebar.number_input("Away Team Code", value=0)

# -----------------------------
# 🧮 Build Input DataFrame
# -----------------------------
# Create a template DataFrame with all expected columns set to 0
input_df = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)

# Fill only the relevant input columns
for col, val in {
    "form_points_diff": form_points_diff,
    "goal_scored_diff": goal_scored_diff,
    "goal_conceded_diff": goal_conceded_diff,
    "home_team_enc": home_team_enc,
    "away_team_enc": away_team_enc
}.items():
    if col in input_df.columns:
        input_df[col] = val

# -----------------------------
# 🧠 Scale & Predict
# -----------------------------
scaled = scaler.transform(input_df)
pred = rf.predict(scaled)[0]

# -----------------------------
# 🏁 Display Result
# -----------------------------
mapping = {2: "🏠 Home Win", 1: "🤝 Draw", 0: "🚩 Away Win"}

st.subheader("Predicted Result:")
st.success(mapping.get(pred, "Unknown Outcome"))

# -----------------------------
# 📊 Optional: Show Input Data
# -----------------------------
with st.expander("🔍 View Input Data Used for Prediction"):
    st.dataframe(input_df)
