import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --------------------------------
# Streamlit Page Setup
# --------------------------------

st.set_page_config(
    page_title="Find Your Space AI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Find Your Space – Intelligent Real Estate Advisor")

# --------------------------------
# CACHED LOADING (Performance Boost)
# --------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("price_model.pkl")
    features = joblib.load("feature_columns.pkl")
    explainer_obj = shap.TreeExplainer(model)
    return model, features, explainer_obj

@st.cache_data
def load_data():
    return pd.read_csv("clean_bangalore_real_estate.csv")

df = load_data()
model_rf, feature_columns, explainer = load_model()

residual_std = 38.60  # from training residuals

# --------------------------------
# Sidebar Filters (Professional UI)
# --------------------------------

st.sidebar.header("🔎 Manual Filters")

selected_location = st.sidebar.selectbox(
    "Select Location",
    options=["Any"] + sorted(df["location"].unique().tolist())
)

selected_bhk = st.sidebar.selectbox(
    "Select BHK",
    options=["Any"] + sorted(df["bhk"].unique().tolist())
)

selected_budget = st.sidebar.slider(
    "Max Budget (Lakhs)",
    min_value=0,
    max_value=int(df["price"].max()),
    value=int(df["price"].max())
)

# --------------------------------
# Helper Functions
# --------------------------------

def apply_sidebar_filters():
    results = df.copy()

    if selected_location != "Any":
        results = results[results["location"] == selected_location]

    if selected_bhk != "Any":
        results = results[results["bhk"] == selected_bhk]

    results = results[results["price"] <= selected_budget]

    return results

def prepare_input_df(location, bhk, sqft, bath=2, balcony=1):
    input_dict = {
        "total_sqft": sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk
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
    return round(np.exp(log_pred), 2)

def explain_prediction(location, bhk, sqft):
    input_df = prepare_input_df(location, bhk, sqft)
    shap_values = explainer.shap_values(input_df, check_additivity=False)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(fig)

# --------------------------------
# Tabs Layout
# --------------------------------

tab1, tab2, tab3 = st.tabs(["💬 AI Assistant", "📊 Market Analytics", "🧠 Model Diagnostics"])

# --------------------------------
# TAB 1 – AI Assistant
# --------------------------------

with tab1:

    query = st.text_input("Ask about listings, comparisons, or price estimates:")

    if st.button("Submit") and query:

        lower_query = query.lower()

        if any(keyword in lower_query for keyword in [
            "estimate", "how much", "price of", "cost of"
        ]):

            location_match = next((loc for loc in df["location"].unique() if loc.lower() in lower_query), None)
            bhk_match = re.search(r"(\d+)\s*bhk", lower_query)
            sqft_match = re.search(r"(\d+)\s*sqft", lower_query)

            if location_match and bhk_match:
                bhk = int(bhk_match.group(1))
                sqft = int(sqft_match.group(1)) if sqft_match else 1500

                predicted_price = predict_price(location_match, bhk, sqft)

                z_value = 1.28
                error_margin = z_value * residual_std

                lower_bound = round(max(0, predicted_price - error_margin), 2)
                upper_bound = round(predicted_price + error_margin, 2)

                st.markdown(f"""
                ### 💰 Estimated Price (80% Confidence)
                **{lower_bound} – {upper_bound} Lakhs**
                """)

                st.markdown("### 🔍 SHAP Explanation")
                explain_prediction(location_match, bhk, sqft)

            else:
                st.warning("Please specify location and BHK clearly.")

        else:
            results = apply_sidebar_filters()

            if results.empty:
                st.warning("No matching properties found.")
            else:
                st.dataframe(results.head(10))

# --------------------------------
# TAB 2 – Market Analytics
# --------------------------------

with tab2:

    st.subheader("📊 Average Price by Location")

    avg_price = df.groupby("location")["price"].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    avg_price.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)

    st.subheader("📈 Price Distribution")

    fig2, ax2 = plt.subplots()
    ax2.hist(df["price"], bins=30)
    ax2.set_xlabel("Price (Lakhs)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

# --------------------------------
# TAB 3 – Model Diagnostics
# --------------------------------

with tab3:

    st.subheader("🌲 Feature Importance")

    importances = model_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(10)

    fig3, ax3 = plt.subplots()
    feat_imp.plot(kind="barh", ax=ax3)
    ax3.invert_yaxis()
    st.pyplot(fig3)

    st.subheader("📉 Residual Distribution")

    preds_log = model_rf.predict(prepare_input_df(df.iloc[0]["location"], df.iloc[0]["bhk"], df.iloc[0]["total_sqft"]))
    # simple placeholder since full residual set not saved
    st.info("Residual STD used for interval: 38.60 Lakhs")
