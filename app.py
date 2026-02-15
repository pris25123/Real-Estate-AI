import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
import matplotlib.pyplot as plt
import requests

warnings.filterwarnings("ignore")

# --------------------------------
# Secure HuggingFace Token
# --------------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def call_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            return "⚠️ LLM returned unexpected format."
        else:
            return None

    except Exception:
        return None


# --------------------------------
# Load Data & Models
# --------------------------------

df = pd.read_csv("clean_bangalore_real_estate.csv")
model_rf = joblib.load("price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

residual_std = 38.60
Z_VALUE = 1.28


# --------------------------------
# ML Helper Functions
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
    log_pred = model_rf.predict(prepare_input_df(location, bhk, sqft))[0]
    return np.exp(log_pred)


def prediction_interval(price):
    error = Z_VALUE * residual_std
    return max(0, price - error), price + error


# --------------------------------
# Conversational Intent Detection
# --------------------------------

def detect_intent(query):
    q = query.lower()
    if "cheapest" in q or "lowest" in q:
        return "cheapest"
    if "range" in q:
        return "range"
    if "average" in q:
        return "average"
    if "compare" in q:
        return "compare"
    if "estimate" in q or "cost" in q or "price of" in q or "how much" in q:
        return "estimate"
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
# UI Layout
# --------------------------------

st.set_page_config(page_title="Find Your Space", layout="wide")
st.title("🏠 Find Your Space – Intelligent Real Estate Advisor")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Assistant", "📊 Market Analytics", "🏗 Investment Studio", "🧠 Model Diagnostics"]
)


# =================================
# 💬 TAB 1 – Assistant
# =================================

with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask about listings, estimates, comparisons...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent = detect_intent(user_input)
        location = extract_location(user_input)
        bhk = extract_bhk(user_input)
        sqft = extract_sqft(user_input)

        structured_response = ""

        if intent == "cheapest" and location:
            results = df[df["location"] == location]
            if bhk:
                results = results[results["bhk"] == bhk]
            results = results.sort_values("price").head(3)

            for _, r in results.iterrows():
                structured_response += f"- ₹ {r['price']} Lakhs | {int(r['total_sqft'])} sqft | {r['bhk']} BHK\n"

        elif intent == "range" and location:
            loc_df = df[df["location"] == location]
            structured_response = f"Price range in {location}: ₹ {loc_df['price'].min()} – {loc_df['price'].max()} Lakhs"

        elif intent == "average" and location:
            avg = df[df["location"] == location]["price"].mean()
            structured_response = f"Average price in {location}: ₹ {round(avg,2)} Lakhs"

        elif intent == "estimate" and location and bhk:
            price = predict_price(location, bhk, sqft)
            lower, upper = prediction_interval(price)
            structured_response = f"Estimated price: ₹ {round(price,2)} Lakhs (80% Range: ₹ {round(lower,2)} – {round(upper,2)})"

        elif location:
            results = df[df["location"] == location].head(5)
            for _, r in results.iterrows():
                structured_response += f"- ₹ {r['price']} Lakhs | {int(r['total_sqft'])} sqft | {r['bhk']} BHK\n"

        else:
            structured_response = "Please mention a valid Bangalore location."

        prompt = f"""
You are a professional Bangalore real estate advisor.

User question:
{user_input}

Verified database results:
{structured_response}

Explain clearly and professionally.
Do NOT invent information.
If structured data is empty, say no data available.
"""

        llm_response = call_llm(prompt)

        final_response = llm_response if llm_response else structured_response

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# =================================
# 📊 TAB 2 – Market Analytics
# =================================

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


# =================================
# 🏗 TAB 3 – Investment Studio
# =================================

with tab3:

    st.header("Investment & ROI Calculator")

    location = st.selectbox("Select Location", sorted(df["location"].unique()))
    bhk = st.selectbox("Select BHK", sorted(df["bhk"].unique()))
    sqft = st.number_input("Built-up Area (sqft)", value=1500)
    expected_rent = st.number_input("Expected Monthly Rent (₹)", value=30000)
    holding_years = st.slider("Investment Horizon (Years)", 1, 20, 5)
    annual_appreciation = st.slider("Expected Annual Appreciation (%)", 0, 15, 5)

    if st.button("Calculate Investment Metrics"):

        predicted_price = predict_price(location, bhk, sqft)
        lower, upper = prediction_interval(predicted_price)

        st.write(f"Predicted Value: ₹ {round(predicted_price,2)} Lakhs")
        st.write(f"80% Range: ₹ {round(lower,2)} – {round(upper,2)} Lakhs")

        annual_rent = expected_rent * 12
        rental_yield = (annual_rent / (predicted_price * 100000)) * 100
        st.write(f"Rental Yield: {round(rental_yield,2)} %")

        future_value = predicted_price * ((1 + annual_appreciation/100) ** holding_years)
        st.write(f"Projected Value After {holding_years} Years: ₹ {round(future_value,2)} Lakhs")


# =================================
# 🧠 TAB 4 – Model Diagnostics
# =================================

with tab4:

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

    fig4, ax4 = plt.subplots()
    ax4.barh(importance_df["Feature"], importance_df["Importance"])
    ax4.invert_yaxis()
    st.pyplot(fig4)
