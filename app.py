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

HF_TOKEN = st.secrets.get("HF_TOKEN", None)
st.write("Token loaded:", bool(HF_TOKEN))
API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def call_llm(prompt):
    if not HF_TOKEN:
        return None

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=40
        )

        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]

            return "⚠️ Unexpected LLM response format."

        elif response.status_code == 503:
            return "⏳ Model is warming up. Try again in a few seconds."

        else:
            return f"⚠️ LLM Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"⚠️ LLM Exception: {str(e)}"



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
# Data Retrieval Logic
# --------------------------------

def get_structured_data(query):
    query_lower = query.lower()
    structured = ""

    locations_found = [loc for loc in df["location"].unique()
                       if loc.lower() in query_lower]

    bhk_match = re.search(r'(\d+)\s*bhk', query_lower)
    bhk = int(bhk_match.group(1)) if bhk_match else None

    sqft_match = re.search(r'(\d+)\s*sqft', query_lower)
    sqft = int(sqft_match.group(1)) if sqft_match else 1500

    if "cheapest" in query_lower and locations_found:
        loc = locations_found[0]
        results = df[df["location"] == loc]
        if bhk:
            results = results[results["bhk"] == bhk]
        results = results.sort_values("price").head(3)

        for _, r in results.iterrows():
            structured += f"- ₹ {r['price']} Lakhs | {int(r['total_sqft'])} sqft | {r['bhk']} BHK\n"

    elif "range" in query_lower and locations_found:
        loc = locations_found[0]
        loc_df = df[df["location"] == loc]
        structured = f"Price range in {loc}: ₹ {loc_df['price'].min()} – {loc_df['price'].max()} Lakhs"

    elif "average" in query_lower and locations_found:
        loc = locations_found[0]
        avg = df[df["location"] == loc]["price"].mean()
        structured = f"Average price in {loc}: ₹ {round(avg,2)} Lakhs"

    elif ("estimate" in query_lower or "how much" in query_lower) and locations_found and bhk:
        loc = locations_found[0]
        price = predict_price(loc, bhk, sqft)
        lower, upper = prediction_interval(price)
        structured = f"Estimated value for {bhk} BHK in {loc} ({sqft} sqft): ₹ {round(price,2)} Lakhs (Range: ₹ {round(lower,2)} – {round(upper,2)})"

    elif locations_found:
        loc = locations_found[0]
        results = df[df["location"] == loc].head(5)
        for _, r in results.iterrows():
            structured += f"- ₹ {r['price']} Lakhs | {int(r['total_sqft'])} sqft | {r['bhk']} BHK\n"

    else:
        structured = "No structured data available for this query."

    return structured


# --------------------------------
# UI Layout
# --------------------------------

st.set_page_config(page_title="Find Your Space", layout="wide")
st.title("🏠 Find Your Space – Intelligent Real Estate Advisor")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Assistant", "📊 Market Analytics", "🏗 Investment Studio", "🧠 Model Diagnostics"]
)


# =================================
# 💬 Assistant (LLM Reasoning Layer)
# =================================

with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask about listings, investment advice, comparisons...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        structured_data = get_structured_data(user_input)

        prompt = f"""
You are a professional Bangalore real estate advisor.

User question:
{user_input}

Verified structured data from database:
{structured_data}

Rules:
- Use ONLY the structured data provided.
- Do NOT invent numbers.
- If no data available, clearly say so.
- Provide advisory reasoning where appropriate.
- Be concise but insightful.
"""

        llm_response = call_llm(prompt)

        final_response = llm_response if llm_response else structured_data

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# =================================
# 📊 Market Analytics
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
# 🏗 Investment Studio
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
# 🧠 Model Diagnostics
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




