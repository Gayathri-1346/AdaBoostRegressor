import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spotify Popularity Prediction",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fbff);
}

h1 {
    color: #4b4bff;
    text-align: center;
}

.stSidebar {
    background-color: #ffffff;
}

.metric-box {
    padding: 10px;
    border-radius: 10px;
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎵 Spotify Popularity Prediction")
st.write("Predict Spotify Song Popularity using AdaBoost Regressor")

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "SpotifyFeatures.csv"
)")

df = pd.read_csv(DATA_PATH)

# ---------------- DATA CLEANING ----------------
df['explicit'] = df['explicit'].astype(int)

# ---------------- FEATURES ----------------
X = df[[
    'duration_ms',
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]]

y = df['popularity']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = AdaBoostRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎧 Enter Song Features")

duration_ms = st.sidebar.slider(
    "Duration (ms)",
    0,
    500000,
    200000
)

danceability = st.sidebar.slider(
    "Danceability",
    0.0,
    1.0,
    0.5
)

energy = st.sidebar.slider(
    "Energy",
    0.0,
    1.0,
    0.5
)

key = st.sidebar.slider(
    "Key",
    0,
    11,
    5
)

loudness = st.sidebar.slider(
    "Loudness",
    -60.0,
    5.0,
    -10.0
)

mode = st.sidebar.selectbox(
    "Mode",
    [0, 1]
)

speechiness = st.sidebar.slider(
    "Speechiness",
    0.0,
    1.0,
    0.1
)

acousticness = st.sidebar.slider(
    "Acousticness",
    0.0,
    1.0,
    0.5
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0,
    1.0,
    0.0
)

liveness = st.sidebar.slider(
    "Liveness",
    0.0,
    1.0,
    0.2
)

valence = st.sidebar.slider(
    "Valence",
    0.0,
    1.0,
    0.5
)

tempo = st.sidebar.slider(
    "Tempo",
    50.0,
    250.0,
    120.0
)

# ---------------- USER INPUT ----------------
input_data = pd.DataFrame({
    'duration_ms': [duration_ms],
    'danceability': [danceability],
    'energy': [energy],
    'key': [key],
    'loudness': [loudness],
    'mode': [mode],
    'speechiness': [speechiness],
    'acousticness': [acousticness],
    'instrumentalness': [instrumentalness],
    'liveness': [liveness],
    'valence': [valence],
    'tempo': [tempo]
})

# ---------------- SCALING INPUT ----------------
input_scaled = scaler.transform(input_data)

# ---------------- PREDICT BUTTON ----------------
if st.sidebar.button("Predict Popularity"):

    prediction = model.predict(input_scaled)[0]

    st.subheader("🎯 Predicted Popularity")

    st.success(f"Predicted Popularity Score: {prediction:.2f}")

    # -------- Popularity Meter --------
    st.subheader("🔥 Popularity Meter")

    progress_value = int(max(0, min(prediction, 100)))

    st.progress(progress_value)

    if prediction >= 80:
        st.success("This song may become a HIT 🎉")

    elif prediction >= 50:
        st.info("This song has Average Popularity 🎵")

    else:
        st.warning("This song may have Lower Popularity 📉")

    # -------- User Input Graph --------
    st.subheader("📊 Your Song Features")

    feature_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Value": input_data.iloc[0].values
    })

    fig_user, ax_user = plt.subplots(figsize=(10,5))

    ax_user.bar(
        feature_df["Feature"],
        feature_df["Value"]
    )

    plt.xticks(rotation=45)

    ax_user.set_title("Input Features")

    st.pyplot(fig_user)

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MSE", f"{mse:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

with col3:
    st.metric("MAE", f"{mae:.2f}")

with col4:
    st.metric("R² Score", f"{r2:.2f}")

# ---------------- ACTUAL VS PREDICTED ----------------
st.subheader("📉 Actual vs Predicted Popularity")

fig1, ax1 = plt.subplots(figsize=(7,5))

ax1.scatter(y_test, y_pred)

ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax1.set_xlabel("Actual Popularity")
ax1.set_ylabel("Predicted Popularity")
ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("🎵 Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(rotation=45)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

# ---------------- DATASET PREVIEW ----------------
st.subheader("🗂 Dataset Preview")

st.dataframe(df.head())
