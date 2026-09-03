import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI Customer Churn Analysis", page_icon=None, layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_FOLDER = PROJECT_ROOT / "08_Reports"
BUSINESS_FILE = REPORTS_FOLDER / "final_churn_business_insights.csv"
MODEL_FILE = REPORTS_FOLDER / "final_model_comparison.csv"
AI_REPORT_FILE = REPORTS_FOLDER / "ai_churn_analyst_report.txt"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def business_value(df, name):
    if "Insight" not in df.columns or "Value" not in df.columns:
        return None
    result = df[df["Insight"].astype(str).str.contains(name, case=False, na=False, regex=False)]
    return result.iloc[0]["Value"] if not result.empty else None

def number(value, default=None):
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default

def fmt(value):
    n = number(value)
    if n is None:
        return str(value)
    return f"{int(n):,}" if n.is_integer() else f"{n:,.2f}"

def best_model(df, metric):
    if "Model" not in df.columns or metric not in df.columns:
        return None, None
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[values.notna(), ["Model"]].copy()
    valid[metric] = values[values.notna()]
    if valid.empty:
        return None, None
    row = valid.loc[valid[metric].idxmax()]
    return str(row["Model"]), float(row[metric])

def ollama_available():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        return any(str(x.get("name", "")) == OLLAMA_MODEL for x in models)
    except Exception:
        return False

def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "No response returned.")

try:
    business_df = load_csv(BUSINESS_FILE)
except Exception as error:
    st.error("Business insights file could not be loaded.")
    st.write(str(error))
    st.stop()

try:
    model_df = load_csv(MODEL_FILE)
except Exception as error:
    st.error("Model comparison file could not be loaded.")
    st.write(str(error))
    st.stop()

total_customers = business_value(business_df, "Total Customers")
churned_customers = business_value(business_df, "Churned Customers")
churn_rate = business_value(business_df, "Overall Churn Rate")

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
best = {metric: best_model(model_df, metric) for metric in metrics if metric in model_df.columns}
best_roc_model, best_roc_auc = best.get("ROC-AUC", (None, None))

st.title("AI Customer Churn Analysis")
st.write("Telecom customer churn analysis using machine learning, business insights and AI.")

page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Business Insights", "Model Comparison", "Customer Risk", "AI Analyst"],
)

if page == "Overview":
    st.header("Customer Churn Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", fmt(total_customers))
    c2.metric("Churned Customers", fmt(churned_customers))
    rate = number(churn_rate)
    c3.metric("Overall Churn Rate", f"{rate:.1f} percent" if rate is not None else str(churn_rate))

    st.subheader("Project Conclusion")
    st.write(f"The project contains {fmt(total_customers)} customers and {fmt(churned_customers)} churned customers.")
    if best_roc_auc is not None:
        st.write(f"The best ROC AUC is {best_roc_auc:.4f} from {best_roc_model}.")
        if best_roc_auc < 0.60:
            st.warning("The current models show weak predictive separation. Individual churn decisions should not rely on this model alone.")

    st.subheader("Project Capabilities")
    capabilities = pd.DataFrame({
        "Capability": [
            "Business churn analysis", "Machine learning comparison", "Customer risk analysis",
            "AI business analysis", "Interactive dashboard", "GitHub and hosted deployment support"
        ],
        "Status": ["Available", "Available", "Available", "Available locally", "Available", "Ready"]
    })
    st.dataframe(capabilities, width="stretch", hide_index=True)

    st.subheader("Deployment Information")
    st.write("The dashboard can run locally with Ollama. Streamlit Cloud can host the dashboard, but it cannot access Ollama running on your personal computer.")

elif page == "Business Insights":
    st.header("Business Churn Insights")
    st.dataframe(business_df, width="stretch", hide_index=True)
    st.subheader("Business Findings")
    for _, row in business_df.iterrows():
        insight = str(row.get("Insight", "")).strip()
        value = str(row.get("Value", "")).strip()
        if insight and value:
            st.write(insight + ": " + value)

elif page == "Model Comparison":
    st.header("Machine Learning Model Comparison")
    st.dataframe(model_df, width="stretch", hide_index=True)
    available = [m for m in metrics if m in model_df.columns]
    if available and "Model" in model_df.columns:
        selected = st.selectbox("Select Model Metric", available)
        chart = model_df.copy()
        chart[selected] = pd.to_numeric(chart[selected], errors="coerce")
        chart = chart.dropna(subset=[selected])
        if not chart.empty:
            fig = px.bar(chart, x="Model", y=selected, title="Model Comparison")
            fig.update_layout(xaxis_title="Model", yaxis_title=selected)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Best Model Results")
    rows = []
    for metric in metrics:
        if metric in best:
            model_name, score = best[metric]
            rows.append({"Metric": metric, "Best Model": model_name, "Score": f"{score:.4f}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if best_roc_auc is not None and best_roc_auc < 0.60:
        st.warning("The ROC AUC is below 0.60. The current model has weak predictive separation.")

elif page == "Customer Risk":
    st.header("Customer Risk Analysis")
    st.write("Enter customer information to compare the customer with observed stayed customer averages.")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1.0, 120.0, 25.0)
        dependents = st.number_input("Number of Dependents", 0.0, 50.0, 2.0)
        salary = st.number_input("Estimated Salary", 0.0, value=55000.0)
        calls = st.number_input("Calls Made", 0.0, value=50.0)
    with c2:
        sms = st.number_input("SMS Sent", 0.0, value=25.0)
        data_used = st.number_input("Data Used", 0.0, value=5000.0)
        registration_year = st.number_input("Registration Year", 2000, 2100, 2025)
        registration_month = st.number_input("Registration Month", 1, 12, 2)

    total_communication = calls + sms
    profile = pd.DataFrame({
        "Feature": ["Age", "Dependents", "Estimated Salary", "Calls Made", "SMS Sent", "Data Used", "Total Communication", "Registration Year", "Registration Month"],
        "Customer Value": [age, dependents, salary, calls, sms, data_used, total_communication, registration_year, registration_month]
    })
    st.subheader("Customer Profile")
    st.dataframe(profile, width="stretch", hide_index=True)

    stayed = {
        "Age": 45.53, "Calls Made": 50.57, "SMS Sent": 25.37,
        "Data Used": 5099.32, "Total Communication": 75.94, "Estimated Salary": 84966.26
    }
    customer = {
        "Age": age, "Calls Made": calls, "SMS Sent": sms,
        "Data Used": data_used, "Total Communication": total_communication, "Estimated Salary": salary
    }
    comparison = pd.DataFrame({
        "Feature": list(stayed.keys()),
        "Customer": [customer[x] for x in stayed],
        "Stayed Customer Average": list(stayed.values())
    })
    st.subheader("Observed Customer Comparison")
    st.dataframe(comparison, width="stretch", hide_index=True)

    risk_score = 0
    risk_factors = []
    for feature in ["Age", "Calls Made", "SMS Sent", "Data Used", "Total Communication", "Estimated Salary"]:
        if customer[feature] < stayed[feature]:
            risk_score += 1
            risk_factors.append(feature + " is below the stayed customer average")

    risk_level = "HIGH" if risk_score >= 4 else "MEDIUM" if risk_score >= 2 else "LOW"
    score = round(risk_score / 6 * 100, 1)
    c1, c2 = st.columns(2)
    c1.metric("Analytical Risk Level", risk_level)
    c2.metric("Analytical Score", f"{score:.1f} percent")

    st.subheader("Risk Factors")
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("No below average risk factors were identified.")
    st.warning("This is an analytical comparison score. It is not a calibrated machine learning probability.")

elif page == "AI Analyst":
    st.header("AI Churn Analyst")
    st.write("Local Ollama works when this application and Ollama are running on the same computer.")

    if ollama_available():
        st.success("Local Ollama connection is available.")
        question = st.text_area("Ask a question about the churn project", value="Why is churn prediction weak?")
        if st.button("Ask Local AI"):
            prompt = f"""
You are a telecom customer churn business analyst.
Use only the project information below.

Business insights:
{business_df.to_string(index=False)}

Model comparison:
{model_df.to_string(index=False)}

Question:
{question}

Answer in simple professional English.
Do not invent statistics.
Clearly distinguish observed facts from recommendations.
"""
            try:
                with st.spinner("AI is preparing the answer"):
                    answer = ask_ollama(prompt)
                st.subheader("AI Answer")
                st.write(answer)
            except Exception as error:
                st.error("Local AI request failed.")
                st.write(str(error))
    else:
        st.info("Local Ollama is not available in this environment.")
        st.write("The dashboard can run online, but local Ollama cannot be reached from Streamlit Cloud.")
        if AI_REPORT_FILE.exists():
            st.subheader("Existing AI Business Report")
            st.text_area("AI Report", AI_REPORT_FILE.read_text(encoding="utf-8"), height=500)
        else:
            st.write("The existing AI report was not found.")

st.divider()
st.write("AI Customer Churn Analysis Application")
st.write("Machine learning and AI results are analytical support and should not be treated as guaranteed individual churn predictions.")
