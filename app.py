import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("C:\\Users\\STAR\\Downloads\\ipl-next-over-runs\\models\\next_over_runs_model.pkl")
features = joblib.load("C:\\Users\\STAR\\Downloads\\ipl-next-over-runs\\models\\features.pkl")

st.title("IPL Next Over Runs Predictor")

st.write("Data-driven ML model for T20 next-over run prediction")

# Inputs
over = st.slider("Current Over (0–19)", 0, 19, 6)
runs_in_over = st.number_input("Runs in Last Over", 0, 36, 7)
runs_last_3 = st.number_input("Avg Runs Last 3 Overs", 0.0, 20.0, 8.0)
wickets_last_3 = st.number_input("Wickets Last 3 Overs", 0, 6, 1)
current_rr = st.number_input("Current Run Rate", 3.0, 15.0, 7.5)
wickets_remaining = st.slider("Wickets Remaining", 0, 10, 7)
over_phase = st.selectbox("Over Phase", [0, 1, 2])

# Create dataframe
input_df = pd.DataFrame([[
    over,
    runs_in_over,
    runs_last_3,
    wickets_last_3,
    current_rr,
    wickets_remaining,
    over_phase
]], columns=features)

# Prediction
pred = model.predict(input_df)[0]

st.subheader(f"Predicted Next Over Runs: {pred:.2f}")

# Trade suggestion
if pred - runs_in_over >= 1.5:
    st.success("Trade Signal: OVER")
elif pred - runs_in_over <= -1.5:
    st.error("Trade Signal: UNDER")
else:
    st.warning("No Trade")
