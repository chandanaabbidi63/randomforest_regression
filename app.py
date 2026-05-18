
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =========================
# Title
# =========================

st.markdown("""
<h1 style='text-align:center; color:#FF4B4B;'>
🚗 Car Price Prediction Using Random Forest Regression
</h1>
""", unsafe_allow_html=True)

st.write("---")

# =========================
# Load Dataset
# =========================

@st.cache_data
def load_data():

    data = pd.read_csv("car_price_prediction.csv")

    return data

try:
    df = load_data()

except:
    st.error("Dataset file not found.")
    st.stop()

# =========================
# Display Dataset
# =========================

st.subheader("📌 Dataset Preview")

st.dataframe(df.head())

# =========================
# Dataset Information
# =========================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Dataset Shape")

    st.write(df.shape)

with col2:

    st.subheader("📋 Dataset Columns")

    st.write(df.columns.tolist())

# =========================
# Missing Values
# =========================

st.subheader("🔍 Missing Values")

missing = df.isnull().sum()

st.dataframe(missing)

# =========================
# Statistical Summary
# =========================

st.subheader("📈 Statistical Summary")

st.dataframe(df.describe())

# =========================
# Check Dataset Columns
# =========================

st.subheader("📌 Column Names")

st.write(df.columns)

# ============================================================
# IMPORTANT:
# CHANGE THESE COLUMN NAMES ACCORDING TO YOUR DATASET
# ============================================================

# Example Assumptions:
# Target Column = Price
# Other Columns = features

# =========================
# Encoding Categorical Columns
# =========================

le = LabelEncoder()

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = le.fit_transform(df[col])

# =========================
# Select Features and Target
# =========================

# CHANGE 'Price' TO YOUR TARGET COLUMN NAME

target_column = "Price"

if target_column not in df.columns:

    st.error(f"Target column '{target_column}' not found in dataset.")

    st.stop()

X = df.drop(target_column, axis=1)

y = df[target_column]

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Model
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================

y_pred = model.predict(X_test)

# =========================
# Evaluation Metrics
# =========================

st.subheader("🤖 Model Evaluation")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

m1, m2, m3, m4 = st.columns(4)

m1.metric("MAE", round(mae, 2))

m2.metric("MSE", round(mse, 2))

m3.metric("RMSE", round(rmse, 2))

m4.metric("R² Score", round(r2, 2))

# =========================
# Actual vs Predicted Graph
# =========================

st.subheader("📉 Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(8,5))

ax1.scatter(y_test, y_pred)

ax1.set_xlabel("Actual Values")

ax1.set_ylabel("Predicted Values")

ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)

# =========================
# Feature Importance
# =========================

st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance,
    ax=ax2
)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

# =========================
# User Input Section
# =========================

st.sidebar.header("🚘 Enter Car Details")

input_data = {}

for column in X.columns:

    if X[column].dtype == 'int64':

        value = st.sidebar.number_input(
            f"{column}",
            value=int(X[column].mean())
        )

    else:

        value = st.sidebar.number_input(
            f"{column}",
            value=float(X[column].mean())
        )

    input_data[column] = value

# =========================
# Convert Input into DataFrame
# =========================

input_df = pd.DataFrame([input_data])

# =========================
# Prediction Button
# =========================

if st.sidebar.button("Predict Price"):

    prediction = model.predict(input_df)

    st.subheader("💰 Predicted Car Price")

    st.success(f"Estimated Price: {prediction[0]:.2f}")

# =========================
# Footer
# =========================

st.write("---")

st.markdown("""
<h4 style='text-align:center;'>
Machine Learning Project using Random Forest Regression
</h4>
""", unsafe_allow_html=True)