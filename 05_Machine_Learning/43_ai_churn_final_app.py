"""
AI Telecom Customer Churn - Final Production-Ready Streamlit App
Author: Prakash

Features
- Executive churn overview
- Business insights
- ML model comparison
- Interactive customer analytical risk assessment
- Live Gemini Cloud AI Analyst
- AI business interpretation
- AI customer interpretation
- Secure Streamlit Secrets / environment variable handling
- Streamlit Community Cloud compatible
"""

from pathlib import Path
import os
import json

import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Prakash | AI Telecom Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_FOLDER = PROJECT_ROOT / "08_Reports"

BUSINESS_FILE = REPORTS_FOLDER / "final_churn_business_insights.csv"
MODEL_FILE = REPORTS_FOLDER / "final_model_comparison.csv"
AI_REPORT_FILE = REPORTS_FOLDER / "ai_churn_analyst_report.txt"

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1.02rem;
        opacity: 0.78;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 0.92rem;
        opacity: 0.72;
        margin-bottom: 1.2rem;
    }
    .ai-box {
        padding: 1rem 1.15rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.30);
        margin-top: .6rem;
    }
    .risk-high {
        padding: 14px;
        border-radius: 10px;
        border: 2px solid #ff4b4b;
        background: rgba(255,75,75,.10);
        font-weight: 700;
    }
    .risk-medium {
        padding: 14px;
        border-radius: 10px;
        border: 2px solid #ffa500;
        background: rgba(255,165,0,.10);
        font-weight: 700;
    }
    .risk-low {
        padding: 14px;
        border-radius: 10px;
        border: 2px solid #21c354;
        background: rgba(33,195,84,.10);
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def number(value, default=None):
    try:
        return float(
            str(value).replace("%", "").replace(",", "").strip()
        )
    except Exception:
        return default


def fmt(value):
    n = number(value)
    if n is None:
        return str(value)
    if n.is_integer():
        return f"{int(n):,}"
    return f"{n:,.2f}"


def business_value(df, name):
    if "Insight" not in df.columns or "Value" not in df.columns:
        return None
    mask = df["Insight"].astype(str).str.contains(
        name, case=False, na=False, regex=False
    )
    result = df.loc[mask]
    return None if result.empty else result.iloc[0]["Value"]


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


def get_secret(name, default=None):
    """Secrets first, environment variable second."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_resource
def create_gemini_client(api_key):
    if not api_key or not GEMINI_AVAILABLE:
        return None
    return genai.Client(api_key=api_key)


def gemini_available():
    return GEMINI_AVAILABLE and bool(get_secret("GEMINI_API_KEY"))


def ask_gemini(prompt):
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "google-genai is not installed. Add google-genai to requirements.txt."
        )

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Cloud Secrets."
        )

    model = get_secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    client = create_gemini_client(api_key)

    if client is None:
        raise RuntimeError("Gemini client could not be initialized.")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = getattr(response, "text", None)
    if not text:
        return "Gemini returned an empty response."
    return text


def dataframe_context(df, max_rows=80):
    if df is None or df.empty:
        return "No data available."
    return df.head(max_rows).to_string(index=False)


def safe_error(error):
    """Show useful errors without exposing API keys."""
    st.error(f"AI request failed: {type(error).__name__}: {error}")


def ai_prompt(question, business_df, model_df, customer=None, risk=None):
    customer_block = ""
    if customer is not None:
        customer_block = f"""
CUSTOMER PROFILE
================
{json.dumps(customer, indent=2)}

ANALYTICAL RISK RESULT
======================
{json.dumps(risk, indent=2)}
"""

    return f"""
You are the Live AI Analyst for an Indian telecom customer churn analytics
project created by Prakash.

Use ONLY the evidence supplied in this prompt.

BUSINESS INSIGHTS
=================
{dataframe_context(business_df)}

MODEL COMPARISON
================
{dataframe_context(model_df)}

{customer_block}

USER QUESTION
=============
{question}

RULES
=====
1. Never invent statistics, customer facts, model scores, or business findings.
2. If the supplied evidence is insufficient, clearly say so.
3. Clearly separate OBSERVED EVIDENCE from RECOMMENDATIONS.
4. Never describe the analytical customer score as a calibrated ML probability.
5. Do not claim a causal relationship unless the supplied evidence establishes one.
6. Give practical telecom retention recommendations.
7. Explain technical terms in simple professional English.
8. Use headings and concise bullet points.
9. Focus on customer retention, churn prevention, model interpretation, and business value.
10. Treat the output as decision support, not a guaranteed prediction.
"""


# ============================================================
# LOAD PROJECT DATA
# ============================================================
try:
    business_df = load_csv(BUSINESS_FILE)
except Exception as error:
    st.error(f"Could not load business insights: {BUSINESS_FILE}")
    st.exception(error)
    st.stop()

try:
    model_df = load_csv(MODEL_FILE)
except Exception as error:
    st.error(f"Could not load model comparison: {MODEL_FILE}")
    st.exception(error)
    st.stop()


# ============================================================
# METRICS
# ============================================================
total_customers = business_value(business_df, "Total Customers")
churned_customers = business_value(business_df, "Churned Customers")
churn_rate = business_value(business_df, "Overall Churn Rate")

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC",
]

best = {
    metric: best_model(model_df, metric)
    for metric in METRICS
    if metric in model_df.columns
}

best_roc_model, best_roc_auc = best.get("ROC-AUC", (None, None))


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">📊 AI Telecom Customer Churn</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Machine Learning + Business Intelligence + Live Gemini AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="author">Built by <b>Prakash</b> | AI Customer Retention Decision-Support Platform</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Business Insights",
        "Model Comparison",
        "Customer Risk",
        "AI Analyst",
        "Project Information",
    ],
)

st.sidebar.divider()

if gemini_available():
    st.sidebar.success("🟢 Gemini AI Connected")
else:
    st.sidebar.warning("🟡 Gemini AI Not Configured")

st.sidebar.caption(
    f"AI Model: {get_secret('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)}"
)
st.sidebar.caption("Author: Prakash")


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.header("Executive Churn Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", fmt(total_customers))
    c2.metric("Churned Customers", fmt(churned_customers))

    rate = number(churn_rate)
    c3.metric(
        "Overall Churn Rate",
        f"{rate:.1f}%" if rate is not None else str(churn_rate),
    )

    c4.metric(
        "Best ROC-AUC",
        f"{best_roc_auc:.4f}" if best_roc_auc is not None else "N/A",
    )

    st.divider()

    st.subheader("Project Summary")
    st.write(
        """
        This platform combines telecom churn analytics, machine-learning
        model comparison, customer-level analytical risk assessment, and
        live Gemini AI interpretation.
        """
    )

    if best_roc_auc is not None:
        st.info(
            f"Best ROC-AUC model: **{best_roc_model}** "
            f"with **{best_roc_auc:.4f}**."
        )

        if best_roc_auc < 0.60:
            st.warning(
                "The current ROC-AUC indicates weak predictive separation. "
                "Do not treat the customer analytical score as a calibrated "
                "probability of churn."
            )

    st.subheader("Platform Capabilities")

    capabilities = pd.DataFrame(
        {
            "Capability": [
                "Business churn analysis",
                "ML model comparison",
                "Customer analytical risk",
                "Live Gemini Cloud AI",
                "Interactive Plotly charts",
                "Streamlit Cloud deployment",
                "Secure API key handling",
            ],
            "Status": [
                "Available",
                "Available",
                "Available",
                "Connected" if gemini_available() else "Needs API key",
                "Available",
                "Ready",
                "Enabled",
            ],
        }
    )

    st.dataframe(
        capabilities,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================
elif page == "Business Insights":
    st.header("📈 Business Churn Insights")

    st.dataframe(
        business_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Business Findings")

    for _, row in business_df.iterrows():
        insight = str(row.get("Insight", "")).strip()
        value = str(row.get("Value", "")).strip()

        if insight and value:
            st.write(f"**{insight}:** {value}")

    st.divider()
    st.subheader("🤖 AI Business Interpretation")

    if gemini_available():
        if st.button(
            "Generate Live AI Business Analysis",
            use_container_width=True,
        ):
            prompt = ai_prompt(
                """
                Analyze the complete business churn situation.
                Identify the most important observed findings, model implications,
                business risks, and practical retention priorities.
                """,
                business_df,
                model_df,
            )

            try:
                with st.spinner("Gemini is analyzing the business evidence..."):
                    answer = ask_gemini(prompt)

                st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as error:
                safe_error(error)
    else:
        st.info(
            "Configure GEMINI_API_KEY in Streamlit Cloud Secrets "
            "to enable Live AI."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.header("🧠 Machine Learning Model Comparison")

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
    )

    available = [
        metric for metric in METRICS if metric in model_df.columns
    ]

    if available and "Model" in model_df.columns:
        selected = st.selectbox(
            "Select metric for interactive comparison",
            available,
        )

        chart = model_df.copy()
        chart[selected] = pd.to_numeric(
            chart[selected],
            errors="coerce",
        )
        chart = chart.dropna(subset=[selected])

        if not chart.empty:
            fig = px.bar(
                chart,
                x="Model",
                y=selected,
                title=f"Model Comparison — {selected}",
                text=selected,
            )
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title=selected,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Model by Metric")

    rows = []
    for metric in METRICS:
        if metric in best:
            model_name, score = best[metric]
            rows.append(
                {
                    "Metric": metric,
                    "Best Model": model_name,
                    "Score": f"{score:.4f}",
                }
            )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    if best_roc_auc is not None and best_roc_auc < 0.60:
        st.warning(
            "ROC-AUC is below 0.60. The available model results show "
            "limited predictive separation."
        )

    st.info(
        "Model metrics are evaluation results from the project reports. "
        "They should be interpreted together rather than relying on one "
        "metric alone."
    )


# ============================================================
# CUSTOMER RISK
# ============================================================
elif page == "Customer Risk":
    st.header("🎯 Customer Risk Analysis")

    st.write(
        """
        Enter a customer profile to compare it with observed
        stayed-customer averages. The resulting score is an analytical
        screening score, not a calibrated machine-learning probability.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age",
            min_value=1.0,
            max_value=120.0,
            value=25.0,
        )
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
        )
        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=55000.0,
        )
        calls = st.number_input(
            "Calls Made",
            min_value=0.0,
            value=50.0,
        )

    with c2:
        sms = st.number_input(
            "SMS Sent",
            min_value=0.0,
            value=25.0,
        )
        data_used = st.number_input(
            "Data Used",
            min_value=0.0,
            value=5000.0,
        )
        registration_year = st.number_input(
            "Registration Year",
            min_value=2000,
            max_value=2100,
            value=2025,
        )
        registration_month = st.number_input(
            "Registration Month",
            min_value=1,
            max_value=12,
            value=2,
        )

    total_communication = calls + sms

    customer = {
        "Age": age,
        "Dependents": dependents,
        "Estimated Salary": salary,
        "Calls Made": calls,
        "SMS Sent": sms,
        "Data Used": data_used,
        "Total Communication": total_communication,
        "Registration Year": registration_year,
        "Registration Month": registration_month,
    }

    profile = pd.DataFrame(
        {
            "Feature": list(customer.keys()),
            "Customer Value": list(customer.values()),
        }
    )

    st.subheader("Customer Profile")
    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    # Existing project evidence used by the original application.
    stayed = {
        "Age": 45.53,
        "Calls Made": 50.57,
        "SMS Sent": 25.37,
        "Data Used": 5099.32,
        "Total Communication": 75.94,
        "Estimated Salary": 84966.26,
    }

    comparison = pd.DataFrame(
        {
            "Feature": list(stayed.keys()),
            "Customer": [customer[x] for x in stayed],
            "Stayed Customer Average": list(stayed.values()),
        }
    )

    st.subheader("Observed Customer Comparison")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    # Analytical score: preserve the logic of the existing project.
    risk_score = 0
    risk_factors = []

    for feature in stayed:
        if customer[feature] < stayed[feature]:
            risk_score += 1
            risk_factors.append(
                f"{feature} is below the observed stayed-customer average."
            )

    risk_level = (
        "HIGH"
        if risk_score >= 4
        else "MEDIUM"
        if risk_score >= 2
        else "LOW"
    )

    score = round(
        risk_score / len(stayed) * 100,
        1,
    )

    risk = {
        "risk_level": risk_level,
        "analytical_score_percent": score,
        "risk_factors": risk_factors,
    }

    st.divider()

    c1, c2 = st.columns(2)
    c1.metric("Analytical Risk Level", risk_level)
    c2.metric("Analytical Risk Score", f"{score:.1f}%")

    if risk_level == "HIGH":
        st.markdown(
            '<div class="risk-high">🔴 HIGH ANALYTICAL RISK</div>',
            unsafe_allow_html=True,
        )
    elif risk_level == "MEDIUM":
        st.markdown(
            '<div class="risk-medium">🟠 MEDIUM ANALYTICAL RISK</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-low">🟢 LOW ANALYTICAL RISK</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Risk Factors")

    if risk_factors:
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.write("No below-average risk factors were identified.")

    st.warning(
        "Important: this is an analytical comparison score based on "
        "observed stayed-customer averages. It is NOT a calibrated "
        "machine-learning probability of churn."
    )

    st.divider()
    st.subheader("🤖 AI Customer Interpretation")

    if gemini_available():
        if st.button(
            "Generate Live AI Customer Analysis",
            use_container_width=True,
        ):
            prompt = ai_prompt(
                """
                Explain this customer's observed differences and analytical
                risk factors. Then recommend practical, non-discriminatory
                retention actions. Keep observations and recommendations
                clearly separated.
                """,
                business_df,
                model_df,
                customer=customer,
                risk=risk,
            )

            try:
                with st.spinner("Gemini is analyzing this customer..."):
                    answer = ask_gemini(prompt)

                st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as error:
                safe_error(error)
    else:
        st.info(
            "Configure GEMINI_API_KEY to enable Live AI Customer Analysis."
        )


# ============================================================
# LIVE AI ANALYST
# ============================================================
elif page == "AI Analyst":
    st.header("🤖 Live Gemini AI Churn Analyst")

    st.write(
        """
        Ask questions about churn, business findings, model performance,
        customer risk, or retention strategy. Gemini receives the project's
        available evidence as context.
        """
    )

    if gemini_available():
        st.success(
            f"🟢 Live Gemini AI is connected — model: "
            f"{get_secret('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)}"
        )
    else:
        st.warning(
            "🟡 Live AI is not configured. Add GEMINI_API_KEY to "
            "Streamlit Cloud Secrets."
        )

    # Example questions
    with st.expander("Example questions"):
        st.write("• What are the most important churn findings?")
        st.write("• Which model performed best and why?")
        st.write("• What retention actions should a telecom company prioritize?")
        st.write("• Explain the model results in simple English.")
        st.write("• What are the limitations of this churn analysis?")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about churn, models, customers, or strategy..."
    )

    if question:
        if not gemini_available():
            st.error(
                "Gemini is not configured. Add GEMINI_API_KEY to "
                "Streamlit Secrets and restart/redeploy the app."
            )
        else:
            st.session_state.ai_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            prompt = ai_prompt(
                question,
                business_df,
                model_df,
            )

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Gemini is analyzing the project..."):
                        answer = ask_gemini(prompt)
                    st.markdown(answer)

                st.session_state.ai_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                safe_error(error)

    if st.session_state.ai_messages:
        if st.button("Clear AI Conversation"):
            st.session_state.ai_messages = []
            st.rerun()


# ============================================================
# PROJECT INFORMATION
# ============================================================
elif page == "Project Information":
    st.header("ℹ️ Project Information")

    st.subheader("Developer")
    st.success("Prakash — AI Telecom Customer Churn Project")

    st.subheader("Technology Stack")

    stack = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Plotly",
                "Streamlit",
                "Google Gemini API",
                "Git / GitHub",
                "Streamlit Community Cloud",
            ],
            "Purpose": [
                "Application and analytics",
                "Data processing",
                "Numerical analysis",
                "Interactive charts",
                "Web application",
                "Live cloud AI",
                "Version control",
                "Free cloud hosting",
            ],
        }
    )

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Architecture")

    st.code(
        """
Telecom Customer Data
        │
        ▼
Data Cleaning / EDA
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ├── Model Comparison
        ├── Customer Risk
        └── Business Insights
                │
                ▼
        Streamlit Dashboard
                │
                ▼
        Gemini Cloud API
                │
                ▼
        Live AI Analyst
        """,
        language="text",
    )

    st.subheader("Security")

    st.write(
        """
        API credentials are read from Streamlit Secrets first and
        environment variables second. The API key is never displayed
        in the application and `.streamlit/secrets.toml` must remain
        outside GitHub.
        """
    )

    st.subheader("AI Governance")

    st.write(
        """
        Gemini is used as an interpretation and decision-support layer.
        The prompt instructs the model to use supplied evidence, avoid
        fabricated statistics, distinguish observations from recommendations,
        and avoid presenting the analytical customer score as a calibrated
        churn probability.
        """
    )

    if AI_REPORT_FILE.exists():
        st.subheader("Existing AI Business Report")
        try:
            report = AI_REPORT_FILE.read_text(encoding="utf-8")
            st.text_area(
                "Report",
                report,
                height=400,
            )
        except Exception:
            st.info("The existing AI report could not be read.")


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "© Prakash | AI Telecom Customer Churn | "
    "Machine Learning + Business Intelligence + Gemini AI"
)

st.caption(
    "Decision-support application. Analytical results are not "
    "guaranteed individual churn predictions."
)
