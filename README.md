# 🏠 Find Your Space — Intelligent Real Estate Advisor

An end-to-end **data-driven real estate intelligence platform** designed for housing market analysis in Bangalore.
This project integrates data preprocessing, machine learning, statistical modeling, and conversational AI to deliver actionable property insights through an interactive web application.

The system combines:

* 📊 Data preprocessing and feature engineering
* 🤖 Machine learning–based price prediction
* 📈 Statistical uncertainty estimation
* 🏗 Investment analytics (ROI and rental yield)
* 💬 Hybrid AI assistant grounded in structured data
* 🌐 Deployment via Streamlit Cloud

---

## 🚀 Live Application

Access the deployed application here:

### 🌐 Streamlit Deployment

👉 **[Launch Find Your Space](https://real-estate-ai-dmymmrkkqyaa2kxvzyc9cg.streamlit.app/)**


---

## 📂 Repository Structure

```
FindYourSpace/
│
├── app.py
├── FindYourSpace.ipynb
├── clean_bangalore_real_estate.csv
├── price_model.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** Bengaluru House Price Dataset (Kaggle)

The dataset includes structured property information such as:

* Location
* BHK configuration
* Total square footage
* Bathrooms
* Balcony count
* Price (Lakhs)
* Availability
* Area type

---

## 🧹 Data Preprocessing

Implemented in `FindYourSpace.ipynb`.

### Cleaning and Transformation Steps

* Removed high-null columns
* Imputed missing balcony values using median statistics
* Extracted numeric BHK values from text fields
* Standardized mixed-unit square footage values

  * Sq. Meter → Sqft
  * Sq. Yards → Sqft
  * Acres → Sqft
  * Perch → Sqft
  * Range values converted to averages
* Removed statistical outliers
* Grouped low-frequency locations into **“Other”**
* Engineered derived feature:

```
price_per_sqft
```

These steps ensure model stability and consistent feature representation.

---

## 🤖 Machine Learning Model

**Model Used:** Random Forest Regressor
**Prediction Target:** `log(price)`

### Feature Set

* total_sqft
* bath
* balcony
* bhk
* one-hot encoded location

---

### Model Performance

| Metric                      | Value          |
| --------------------------- | -------------- |
| R² Score                    | ~0.77          |
| Residual Standard Deviation | 38.6 Lakhs     |
| Prediction Interval         | 80% Confidence |

The model provides not only point predictions but also uncertainty bounds to support informed decision-making.

---

## 📉 Prediction Intervals

Instead of producing a single estimate:

```
Predicted Price = ₹ 85 Lakhs
```

The system returns:

```
₹ 73 – ₹ 97 Lakhs (80% Confidence Interval)
```

Computed using:

```
Prediction ± Z × residual_std
```

This statistical framing improves interpretability and risk awareness.

---

## 💬 Hybrid AI Assistant

The conversational assistant augments the ML system by combining structured analytics with natural language reasoning.

### Workflow

1. Extract structured data from the dataset
2. Perform ML inference if required
3. Compute supporting analytics (averages, ranges, comparisons)
4. Provide verified outputs to an LLM
5. Generate clear explanations grounded in real results

The assistant operates on:

* Dataset-derived insights
* Model predictions
* Computed statistics

It does **not generate unsupported market claims**.

---

## 📊 Streamlit Application Features

### Assistant Interface

* Property listing by location
* Cheapest listing identification
* Location comparisons
* Price estimation
* Budget-based advisory
* Market explanations

### Market Analytics

* Average price by location
* Distribution visualizations

### Investment Studio

* Price prediction
* Confidence intervals
* Rental yield estimation
* Appreciation projection

### Model Diagnostics

* Feature importance visualization
* Model summary statistics
* Residual analysis

---

## 🧠 Example Queries

* List properties in Whitefield
* Cheapest 3 BHK in JP Nagar
* Compare pricing between two locations
* Estimate price of a specified property
* Budget-based purchase suggestions

---

## 🛠 Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Groq API (LLM reasoning layer)

---


## 📈 Project Significance

This platform extends beyond traditional chatbot or regression models. It delivers:

* Data-grounded reasoning
* Machine learning predictions
* Statistical interpretability
* Investment-aware analytics
* Production-grade deployment

It represents a **hybrid AI decision-support system** combining structured modeling with language-driven explanation.


---
