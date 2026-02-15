import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap

# --------------------------------
# Load Data & Models
# --------------------------------

df = pd.read_csv("clean_bangalore_real_estate.csv")
model_rf = joblib.load("price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Create SHAP explainer dynamically (DO NOT load from file)
explainer = shap.TreeExplainer(model_rf)

mae_lakhs = 25  # Replace with your actual MAE in Lakhs

# --------------------------------
# Helper Functions
# --------------------------------

conversation_memory = {}

def extract_filters(query):
    global conversation_memory
    filters = conversation_memory.copy()

    bhk_match = re.search(r'(\d+)\s*bhk', query.lower())
    if bhk_match:
        filters['bhk'] = int(bhk_match.group(1))

    price_match = re.search(r'under\s*(\d+)', query.lower())
    if price_match:
        filters['max_price'] = int(price_match.group(1))

    for loc in df['location'].unique():
        if loc.lower() in query.lower():
            filters['location'] = loc
            break

    conversation_memory = filters
    return filters


def search_properties(query):
    filters = extract_filters(query)
    results = df.copy()

    if 'location' in filters:
        results = results[results['location'] == filters['location']]

    if 'bhk' in filters:
        results = results[results['bhk'] == filters['bhk']]

    if 'max_price' in filters:
        results = results[results['price'] <= filters['max_price']]

    return results.head(5)


def format_properties(properties_df):
    formatted_output = []
    for _, row in properties_df.iterrows():
        formatted_output.append(
            f"- Location: {row['location']}, "
            f"BHK: {int(row['bhk'])}, "
            f"Sqft: {row['total_sqft']:.0f}, "
            f"Bath: {int(row['bath'])}, "
            f"Balcony: {int(row['balcony'])}, "
            f"Price: {row['price']:.2f} Lakhs"
        )
    return "\n".join(formatted_output)


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

    input_df = input_df[feature_columns]
    return input_df


def predict_price(location, bhk, sqft, bath=2, balcony=1):
    input_df = prepare_input_df(location, bhk, sqft, bath, balcony)
    log_pred = model_rf.predict(input_df)[0]
    return round(np.exp(log_pred), 2)


def explain_prediction(location, bhk, sqft, bath=2, balcony=1):
    input_df = prepare_input_df(location, bhk, sqft, bath, balcony)

    shap_values = explainer.shap_values(input_df, check_additivity=False)
    shap_series = pd.Series(shap_values[0], index=feature_columns)

    meaningful_features = []

    for feature in shap_series.index:
        if feature.startswith("location_") and input_df[feature].iloc[0] == 1:
            meaningful_features.append(feature)
        elif feature in ['total_sqft', 'bhk', 'bath', 'balcony']:
            meaningful_features.append(feature)

    shap_series = shap_series[meaningful_features]
    top_features = shap_series.abs().sort_values(ascending=False).head(3)

    explanations = []

    for feature in top_features.index:
        impact = shap_series[feature]
        direction = "increased" if impact > 0 else "reduced"

        if feature == "total_sqft":
            text = "Larger built-up area"
        elif feature == "bhk":
            text = "Number of bedrooms"
        elif feature == "bath":
            text = "Number of bathrooms"
        elif feature == "balcony":
            text = "Balcony count"
        elif feature.startswith("location_"):
            text = feature.replace("location_", "") + " location"
        else:
            text = feature

        explanations.append(f"{text} {direction} the estimated price")

    return explanations


def generate_response(query):
    lower_query = query.lower()

    if "estimate" in lower_query or "how much" in lower_query:
        filters = extract_filters(query)

        sqft_match = re.search(r'(\d+)\s*sqft', lower_query)
        sqft = int(sqft_match.group(1)) if sqft_match else 1500

        if 'location' in filters and 'bhk' in filters:
            predicted_price = predict_price(filters['location'], filters['bhk'], sqft)

            error_margin = mae_lakhs * 0.5
            lower_bound = round(predicted_price - error_margin, 2)
            upper_bound = round(predicted_price + error_margin, 2)

            explanations = explain_prediction(filters['location'], filters['bhk'], sqft)

            return f"""
### 💰 Price Estimate
Estimated range: **{lower_bound} – {upper_bound} Lakhs**

### 🔍 Key Factors
- {explanations[0]}
- {explanations[1]}
- {explanations[2]}
"""
        else:
            return "Please specify location and BHK."

    results = search_properties(query)

    if results.empty:
        return "No matching properties found."

    return format_properties(results)


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(page_title="Find Your Space AI", page_icon="🏠")
st.title("🏠 Find Your Space - AI Real Estate Advisor")

query = st.text_input("Ask about properties or price estimates:")

if st.button("Submit") and query:
    response = generate_response(query)
    st.markdown(response)
