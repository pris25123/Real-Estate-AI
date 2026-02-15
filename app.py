import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------
# Load Data & Models
# --------------------------------

df = pd.read_csv("clean_bangalore_real_estate.csv")
model_rf = joblib.load("price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(model_rf)

residual_std = 38.60   # Your residual std
Z_VALUE = 1.28         # 80% confidence interval


# --------------------------------
# Helper Functions
# --------------------------------

def prepare_input_df(location, bhk, sqft, bath=2, balcony=1):
    input_dict = {
        'total_sqft': sqft,
        'bath': bath,
        'balcony': balcony,
        'bhk': bhk
    }

    for col in feature_columns:
        if col.startswith("location_"):
            input_dict[col] = 1 if col == f"location_{location}" else 0

    input_df = pd.DataFrame([input_dict])

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[feature_columns]


def predict_price(location, bhk, sqft):
    input_df = prepare_input_df(location, bhk, sqft)
    log_pred = model_rf.predict(input_df)[0]
    return np.exp(log_pred)


def get_prediction_interval(predicted_price):
    error_margin = Z_VALUE * residual_std
    lower = max(0, predicted_price - error_margin)
    upper = predicted_price + error_margin
    return round(lower, 2), round(upper, 2)


def explain_prediction(location, bhk, sqft):
    input_df = prepare_input_df(location, bhk, sqft)
    shap_values = explainer.shap_values(input_df, check_additivity=False)
    shap_series = pd.Series(shap_values[0], index=feature_columns)

    shap_series = shap_series.abs().sort_values(ascending=False).head(3)

    explanations = []
    for feature in shap_series.index:
        if feature == "total_sqft":
            text = "Built-up area"
        elif feature == "bhk":
            text = "Number of bedrooms"
        elif feature == "bath":
            text = "Bathrooms"
        elif feature.startswith("location_"):
            text = feature.replace("location_", "") + " location"
        else:
            text = feature
        explanations.append(text)

    return explanations


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(page_title="Find Your Space AI", page_icon="🏠", layout="wide")

st.title("🏠 Find Your Space – Intelligent Real Estate Advisor")

tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "📊 Market Analytics", "🧠 Model Diagnostics"])


# --------------------------------
# SIDEBAR FILTERS
# --------------------------------

st.sidebar.header("🔎 Manual Filters")

selected_location = st.sidebar.selectbox(
    "Location",
    sorted(df["location"].unique())
)

selected_bhk = st.sidebar.selectbox(
    "BHK",
    sorted(df["bhk"].unique())
)

selected_budget = st.sidebar.slider(
    "Max Budget (Lakhs)",
    min_value=0,
    max_value=int(df["price"].max()),
    value=100
)

compare_mode = st.sidebar.checkbox("Enable Comparison Mode")


# --------------------------------
# TAB 1: AI ASSISTANT
# --------------------------------

with tab1:

    st.subheader("Search Listings")

    filtered_df = df[
        (df["location"] == selected_location) &
        (df["bhk"] == selected_bhk) &
        (df["price"] <= selected_budget)
    ]

    if filtered_df.empty:
        st.warning("No matching properties found.")
    else:
        st.success(f"{len(filtered_df)} matching properties found.")

        display_df = filtered_df.head(5)
        st.dataframe(display_df[["location","bhk","total_sqft","bath","balcony","price"]])

    st.divider()

    st.subheader("💰 Instant Price Estimator")

    sqft_input = st.number_input("Enter Sqft", value=1500)

    if st.button("Predict Price"):
        predicted_price = predict_price(selected_location, selected_bhk, sqft_input)
        lower, upper = get_prediction_interval(predicted_price)

        st.markdown(f"### Estimated Price: ₹ {round(predicted_price,2)} Lakhs")
        st.markdown(f"80% Confidence Range: ₹ {lower} – {upper} Lakhs")

        explanations = explain_prediction(selected_location, selected_bhk, sqft_input)

        st.markdown("#### Key Influencing Factors:")
        for e in explanations:
            st.write(f"- {e}")

        # Visualization
        fig, ax = plt.subplots()
        ax.bar(["Lower","Predicted","Upper"], [lower, predicted_price, upper])
        ax.set_ylabel("Price (Lakhs)")
        st.pyplot(fig)


# --------------------------------
# TAB 2: MARKET ANALYTICS
# --------------------------------

with tab2:

    st.subheader("Location Price Trends")

    avg_price = df.groupby("location")["price"].mean().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8,5))
    avg_price.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Price (Lakhs)")
    ax.set_title("Top 15 Locations by Average Price")
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig2, ax2 = plt.subplots()
    df["price"].hist(bins=50, ax=ax2)
    ax2.set_xlabel("Price (Lakhs)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)


# --------------------------------
# TAB 3: MODEL DIAGNOSTICS
# --------------------------------

with tab3:

    st.subheader("Model Summary")

    st.markdown("""
    - Model: Random Forest Regressor  
    - Target: Log(Price)  
    - R² ≈ 0.77  
    - Residual Std ≈ 38.6 Lakhs  
    - 80% Prediction Interval used  
    """)

    st.subheader("Feature Importance")

    importances = model_rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)

    fig3, ax3 = plt.subplots()
    ax3.barh(importance_df["Feature"], importance_df["Importance"])
    ax3.invert_yaxis()
    st.pyplot(fig3)
