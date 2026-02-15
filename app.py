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

residual_std = 38.60
Z_VALUE = 1.28  # 80% interval


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


def get_interval(price):
    error = Z_VALUE * residual_std
    return max(0, price - error), price + error


def explain_prediction(location, bhk, sqft):
    input_df = prepare_input_df(location, bhk, sqft)
    shap_values = explainer.shap_values(input_df, check_additivity=False)
    shap_series = pd.Series(shap_values[0], index=feature_columns)
    shap_series = shap_series.abs().sort_values(ascending=False).head(3)

    readable = []
    for f in shap_series.index:
        if f == "total_sqft":
            readable.append("Built-up area")
        elif f == "bhk":
            readable.append("Number of bedrooms")
        elif f == "bath":
            readable.append("Bathrooms")
        elif f.startswith("location_"):
            readable.append(f.replace("location_", "") + " location")
    return readable


def detect_intent(query):
    q = query.lower()

    if "compare" in q:
        return "compare"
    if "average" in q or "avg" in q:
        return "average"
    if "estimate" in q or "how much" in q or "cost" in q or "price of" in q:
        return "estimate"
    if "cheapest" in q or "lowest" in q:
        return "cheapest"
    if "overpriced" in q or "reasonable" in q:
        return "valuation"
    return "search"


def extract_location(query):
    for loc in df["location"].unique():
        if loc.lower() in query.lower():
            return loc
    return None


def extract_bhk(query):
    match = re.search(r'(\d+)\s*bhk', query.lower())
    return int(match.group(1)) if match else None


def extract_sqft(query):
    match = re.search(r'(\d+)\s*sqft', query.lower())
    return int(match.group(1)) if match else 1500


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(page_title="Find Your Space AI", layout="wide")
st.title("🏠 Find Your Space – Intelligent Real Estate Advisor")

tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "📊 Market Analytics", "🧠 Model Diagnostics"])


# --------------------------------
# Sidebar Filters
# --------------------------------

st.sidebar.header("🔎 Manual Filters")

selected_location = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
selected_bhk = st.sidebar.selectbox("BHK", sorted(df["bhk"].unique()))
selected_budget = st.sidebar.slider("Max Budget (Lakhs)", 0, int(df["price"].max()), 100)


# --------------------------------
# TAB 1 – CONVERSATIONAL AI
# --------------------------------

with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask about listings, price estimates, comparisons...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent = detect_intent(user_input)
        location = extract_location(user_input) or selected_location
        bhk = extract_bhk(user_input) or selected_bhk
        sqft = extract_sqft(user_input)

        response = ""

        if intent == "estimate":
            price = predict_price(location, bhk, sqft)
            lower, upper = get_interval(price)
            factors = explain_prediction(location, bhk, sqft)

            response = f"""
💰 Estimated price for {bhk} BHK in {location} ({sqft} sqft):

**₹ {round(price,2)} Lakhs**

80% Range: ₹ {round(lower,2)} – {round(upper,2)} Lakhs

Key Drivers:
- {factors[0]}
- {factors[1]}
- {factors[2]}
"""

        elif intent == "average":
            avg = df[df["location"] == location]["price"].mean()
            response = f"📊 Average price in {location}: ₹ {round(avg,2)} Lakhs."

        elif intent == "cheapest":
            results = df[(df["location"] == location) & (df["bhk"] == bhk)]
            results = results.sort_values("price").head(3)
            response = "💸 Cheapest options:\n\n"
            for _, r in results.iterrows():
                response += f"- ₹ {r['price']} Lakhs ({int(r['total_sqft'])} sqft)\n"

        elif intent == "compare":
            avg_prices = df.groupby("location")["price"].mean()
            locations = [l for l in df["location"].unique() if l.lower() in user_input.lower()]
            if len(locations) >= 2:
                response = "📊 Comparison:\n\n"
                for l in locations[:2]:
                    response += f"{l}: ₹ {round(avg_prices[l],2)} Lakhs average\n"
            else:
                response = "Please mention two locations to compare."

        elif intent == "valuation":
            price = predict_price(location, bhk, sqft)
            response = f"Predicted fair value: ₹ {round(price,2)} Lakhs.\nIf market price is significantly higher, it may be overpriced."

        else:
            results = df[
                (df["location"] == location) &
                (df["bhk"] == bhk) &
                (df["price"] <= selected_budget)
            ].head(5)

            if results.empty:
                response = "No matching properties found."
            else:
                response = "🏘 Matching properties:\n\n"
                for _, r in results.iterrows():
                    response += f"- ₹ {r['price']} Lakhs | {int(r['total_sqft'])} sqft\n"

        st.session_state.messages.append({"role": "assistant", "content": response})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# --------------------------------
# TAB 2 – MARKET ANALYTICS
# --------------------------------

with tab2:

    st.subheader("Top Locations by Average Price")

    avg_price = df.groupby("location")["price"].mean().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8,5))
    avg_price.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig2, ax2 = plt.subplots()
    df["price"].hist(bins=40, ax=ax2)
    st.pyplot(fig2)


# --------------------------------
# TAB 3 – MODEL DIAGNOSTICS
# --------------------------------

with tab3:

    st.markdown("""
    **Model:** Random Forest  
    **Target:** Log(Price)  
    **R²:** ~0.77  
    **Residual Std:** 38.6 Lakhs  
    **Prediction Interval:** 80% Confidence  
    """)

    importances = model_rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)

    fig3, ax3 = plt.subplots()
    ax3.barh(importance_df["Feature"], importance_df["Importance"])
    ax3.invert_yaxis()
    st.pyplot(fig3)
