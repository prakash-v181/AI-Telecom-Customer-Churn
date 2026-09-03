# ============================================================
# 40_ai_churn_dashboard.py
# AI-Powered Telecom Customer Churn Dashboard
# Uses Streamlit + Plotly + Local Ollama
# ============================================================

import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

reports_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "08_Reports")
)

business_file = os.path.join(
    reports_folder,
    "final_churn_business_insights.csv"
)

model_file = os.path.join(
    reports_folder,
    "final_model_comparison.csv"
)

ai_report_file = os.path.join(
    reports_folder,
    "ai_churn_analyst_report.txt"
)


# ============================================================
# 3. OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# 4. TITLE
# ============================================================

st.title("📊 AI-Powered Customer Churn Dashboard")

st.markdown(
    """
    ### Telecom Customer Churn Analysis
    Interactive business intelligence dashboard powered by
    **Python, Streamlit, Plotly and Local AI (Ollama).**
    """
)

st.divider()


# ============================================================
# 5. LOAD BUSINESS INSIGHTS
# ============================================================

try:

    business_df = pd.read_csv(business_file)

except Exception as e:

    st.error("Could not load business insights file.")
    st.error(str(e))
    st.stop()


# ============================================================
# 6. LOAD MODEL COMPARISON
# ============================================================

try:

    model_df = pd.read_csv(model_file)

except Exception as e:

    st.error("Could not load model comparison file.")
    st.error(str(e))
    st.stop()


# ============================================================
# 7. EXTRACT BUSINESS VALUES
# ============================================================

def get_value(category, default="N/A"):

    try:

        row = business_df[
            business_df["Category"].astype(str).str.lower()
            == category.lower()
        ]

        if not row.empty:

            return row.iloc[0]["Insight"]

    except Exception:
        pass

    return default


# ============================================================
# 8. OVERALL CHURN METRICS
# ============================================================

total_customers = 2026
stayed_customers = 1633
churned_customers = 393
churn_rate = 19.4


st.subheader("📌 Overall Customer Situation")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "Stayed Customers",
        f"{stayed_customers:,}"
    )

with col3:

    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col4:

    st.metric(
        "Overall Churn Rate",
        f"{churn_rate}%"
    )


st.divider()


# ============================================================
# 9. CHURN DISTRIBUTION
# ============================================================

st.subheader("📈 Churn Distribution")

distribution_df = pd.DataFrame(
    {
        "Status": ["Stayed", "Churned"],
        "Customers": [
            stayed_customers,
            churned_customers
        ]
    }
)

fig = px.pie(
    distribution_df,
    names="Status",
    values="Customers",
    hole=0.45,
    title="Customer Churn Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 10. BUSINESS INSIGHTS
# ============================================================

st.subheader("🔎 Key Business Insights")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
        **Highest Telecom Partner Churn**

        Vodafone — 21.28%
        """
    )

with col2:

    st.info(
        """
        **Highest Age Group Churn**

        36-45 — 21.61%
        """
    )

with col3:

    st.info(
        """
        **Highest Usage Segment Churn**

        Low Usage — 19.85%
        """
    )


col1, col2, col3 = st.columns(3)

with col1:

    st.warning(
        """
        **Highest State Churn**

        Bihar — 26.14%
        """
    )

with col2:

    st.warning(
        """
        **Highest City Churn**

        Kolkata — 22.39%
        """
    )

with col3:

    st.warning(
        """
        **Highest Gender Churn**

        Female — 20.30%
        """
    )


st.divider()


# ============================================================
# 11. CHURN BY TELECOM PARTNER
# ============================================================

st.subheader("📱 Churn by Telecom Partner")

partner_df = pd.DataFrame(
    {
        "Telecom Partner": [
            "Vodafone",
            "BSNL",
            "Reliance Jio",
            "Airtel"
        ],
        "Customers": [
            517,
            496,
            506,
            507
        ],
        "Churned": [
            110,
            99,
            97,
            87
        ],
        "Churn Rate": [
            21.28,
            19.96,
            19.17,
            17.16
        ]
    }
)

fig = px.bar(
    partner_df,
    x="Telecom Partner",
    y="Churn Rate",
    text="Churn Rate",
    title="Churn Rate by Telecom Partner"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 12. CHURN BY AGE GROUP
# ============================================================

st.subheader("👥 Churn by Age Group")

age_df = pd.DataFrame(
    {
        "Age Group": [
            "36-45",
            "18-25",
            "56+",
            "46-55",
            "26-35"
        ],
        "Churn Rate": [
            21.61,
            20.33,
            19.55,
            18.56,
            16.98
        ]
    }
)

fig = px.bar(
    age_df,
    x="Age Group",
    y="Churn Rate",
    text="Churn Rate",
    title="Churn Rate by Age Group"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 13. CHURN BY CITY
# ============================================================

st.subheader("🏙️ Churn by City")

city_df = pd.DataFrame(
    {
        "City": [
            "Kolkata",
            "Bangalore",
            "Mumbai",
            "Hyderabad",
            "Chennai",
            "Delhi"
        ],
        "Churn Rate": [
            22.39,
            21.29,
            19.82,
            18.73,
            18.39,
            15.46
        ]
    }
)

fig = px.bar(
    city_df,
    x="City",
    y="Churn Rate",
    text="Churn Rate",
    title="Churn Rate by City"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 14. CHURN BY STATE
# ============================================================

st.subheader("🗺️ Churn by State")

state_df = pd.DataFrame(
    {
        "State": [
            "Bihar",
            "Punjab",
            "Andhra Pradesh",
            "Jharkhand",
            "Tripura",
            "Himachal Pradesh",
            "Sikkim",
            "Nagaland",
            "Mizoram",
            "Maharashtra",
            "Madhya Pradesh",
            "Rajasthan",
            "West Bengal",
            "Tamil Nadu",
            "Manipur",
            "Chhattisgarh",
            "Assam",
            "Gujarat",
            "Kerala",
            "Uttar Pradesh",
            "Goa",
            "Arunachal Pradesh",
            "Haryana",
            "Telangana",
            "Odisha",
            "Karnataka",
            "Meghalaya",
            "Uttarakhand"
        ],
        "Churn Rate": [
            26.14,
            25.71,
            25.00,
            24.14,
            23.68,
            23.26,
            23.19,
            22.86,
            22.73,
            22.73,
            22.39,
            20.97,
            20.90,
            19.48,
            19.44,
            19.12,
            18.92,
            17.72,
            17.14,
            16.00,
            15.69,
            15.58,
            15.48,
            13.95,
            13.85,
            13.64,
            11.27,
            10.14
        ]
    }
)

fig = px.bar(
    state_df,
    x="Churn Rate",
    y="State",
    orientation="h",
    title="Churn Rate by State"
)

fig.update_layout(
    height=850
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("🤖 Machine Learning Model Comparison")

st.dataframe(
    model_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 16. MODEL CHART
# ============================================================

model_chart_df = model_df.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

available_metrics = [
    column
    for column in metric_columns
    if column in model_chart_df.columns
]

if available_metrics:

    chart_df = model_chart_df[
        ["Model"] + available_metrics
    ]

    chart_long = chart_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        chart_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performance Comparison"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 17. AI REPORT
# ============================================================

st.divider()

st.subheader("🧠 AI Churn Analyst Report")

if os.path.exists(ai_report_file):

    with open(
        ai_report_file,
        "r",
        encoding="utf-8"
    ) as file:

        ai_report = file.read()

    with st.expander(
        "View AI-generated business report",
        expanded=True
    ):

        st.markdown(ai_report)

else:

    st.warning(
        "AI report file was not found."
    )


# ============================================================
# 18. LOCAL AI FUNCTION
# ============================================================

def ask_ollama(question):

    context = f"""
You are a telecom customer churn business analyst.

Use ONLY the following project information.

Total customers: 2026
Stayed customers: 1633
Churned customers: 393
Overall churn rate: 19.4%

Highest telecom partner churn:
Vodafone = 21.28%

Highest gender churn:
Female = 20.30%

Highest age-group churn:
36-45 = 21.61%

Highest usage segment churn:
Low Usage = 19.85%

Highest state churn:
Bihar = 26.14%

Highest city churn:
Kolkata = 22.39%

Best ROC-AUC model:
Improved XGBoost = 0.5299

Best PR-AUC model:
Improved XGBoost = 0.1998

Important:
The project found weak predictive separation.
Do not claim that the model provides highly accurate
individual churn prediction.

User question:
{question}

Answer clearly in professional business English.
Do not invent statistics.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": context,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        "No response received from Ollama."
    )


# ============================================================
# 19. AI CHAT
# ============================================================

st.divider()

st.subheader("💬 Ask the AI Churn Analyst")

st.write(
    "Ask questions about your telecom customer churn project."
)

question = st.text_input(
    "Your question",
    placeholder="Example: Why is churn prediction weak?"
)

if st.button("Ask AI"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "AI is analyzing your question..."
        ):

            try:

                answer = ask_ollama(
                    question
                )

                st.success("AI Response")

                st.write(answer)

            except Exception as e:

                st.error(
                    "Could not connect to Ollama."
                )

                st.code(
                    str(e)
                )


# ============================================================
# 20. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Customer Churn Dashboard | "
    "Python + Streamlit + Plotly + Ollama"
)

st.caption(
    "AI Engine: Local Ollama | "
    "Model: llama3.2:3b"
)