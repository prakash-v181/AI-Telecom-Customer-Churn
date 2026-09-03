# ============================================================
# 41_ai_customer_risk.py
# AI-Powered Individual Customer Churn Risk Analyzer
# Uses Local Ollama - No OpenAI API / No Credit Card
# ============================================================

import os
import json
import requests
import pandas as pd
import numpy as np

# ============================================================
# 1. PROJECT PATHS
# ============================================================

reports_folder = "../08_Reports/"

business_insights_file = (
    reports_folder + "final_churn_business_insights.csv"
)

model_comparison_file = (
    reports_folder + "final_model_comparison.csv"
)

output_file = (
    reports_folder + "individual_customer_risk_report.txt"
)

# ============================================================
# 2. OLLAMA SETTINGS
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("AI INDIVIDUAL CUSTOMER CHURN RISK ANALYZER")
print("=" * 70)

print()
print("AI Engine : Ollama")
print("AI Model  :", OLLAMA_MODEL)

# ============================================================
# 4. CHECK OLLAMA
# ============================================================

print()
print("Checking Ollama connection...")

try:

    test_response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )

    if test_response.status_code != 200:
        print("ERROR: Ollama server is not responding correctly.")
        raise SystemExit(1)

    models = test_response.json().get("models", [])

    model_names = [
        model.get("name", "")
        for model in models
    ]

    if OLLAMA_MODEL not in model_names:

        print()
        print("ERROR: Required Ollama model is not available.")
        print("Required model:", OLLAMA_MODEL)
        print()
        print("Run:")
        print(
            '"C:\\Users\\praka\\AppData\\Local\\Programs\\Ollama\\ollama.exe" '
            'pull llama3.2:3b'
        )

        raise SystemExit(1)

    print("Ollama connection: OK")
    print("Model available: OK")

except requests.exceptions.RequestException as e:

    print()
    print("ERROR: Cannot connect to Ollama.")
    print(e)
    print()
    print("Start Ollama and try again.")

    raise SystemExit(1)

# ============================================================
# 5. LOAD BUSINESS INSIGHTS
# ============================================================

try:

    business_df = pd.read_csv(
        business_insights_file
    )

    print()
    print("Business insights loaded successfully!")
    print("Shape:", business_df.shape)

except Exception as e:

    print()
    print("ERROR loading business insights:")
    print(e)

    raise SystemExit(1)

# ============================================================
# 6. LOAD MODEL COMPARISON
# ============================================================

try:

    model_df = pd.read_csv(
        model_comparison_file
    )

    print()
    print("Model comparison loaded successfully!")
    print("Shape:", model_df.shape)

except Exception as e:

    print()
    print("ERROR loading model comparison:")
    print(e)

    raise SystemExit(1)

# ============================================================
# 7. DISPLAY PROJECT INFORMATION
# ============================================================

print()
print("=" * 70)
print("PROJECT MODEL INFORMATION")
print("=" * 70)

print()

if "ROC-AUC" in model_df.columns:

    roc_values = pd.to_numeric(
        model_df["ROC-AUC"],
        errors="coerce"
    )

    valid_roc = roc_values.dropna()

    if len(valid_roc) > 0:

        best_index = valid_roc.idxmax()

        best_model = model_df.loc[
            best_index,
            "Model"
        ]

        best_roc = valid_roc.loc[
            best_index
        ]

        print("Best ROC-AUC Model :", best_model)
        print("Best ROC-AUC       :", round(best_roc, 4))

# ============================================================
# 8. CUSTOMER INPUT
# ============================================================

print()
print("=" * 70)
print("ENTER CUSTOMER INFORMATION")
print("=" * 70)

print()
print("Enter the customer details below.")
print("You can use any reasonable values from your dataset.")
print()

def get_float(prompt_text):

    while True:

        try:

            value = float(
                input(prompt_text)
            )

            return value

        except ValueError:

            print("Please enter a valid number.")


def get_int(prompt_text):

    while True:

        try:

            value = int(
                input(prompt_text)
            )

            return value

        except ValueError:

            print("Please enter a valid integer.")


age = get_float(
    "Age: "
)

num_dependents = get_float(
    "Number of dependents: "
)

estimated_salary = get_float(
    "Estimated salary: "
)

calls_made = get_float(
    "Calls made: "
)

sms_sent = get_float(
    "SMS sent: "
)

data_used = get_float(
    "Data used: "
)

registration_year = get_int(
    "Registration year: "
)

registration_month = get_int(
    "Registration month (1-12): "
)

registration_quarter = (
    (registration_month - 1) // 3
) + 1

# ============================================================
# 9. ENGINEERED FEATURES
# ============================================================

total_communication = (
    calls_made + sms_sent
)

calls_per_data = (
    calls_made / data_used
    if data_used != 0
    else 0
)

sms_per_call = (
    sms_sent / calls_made
    if calls_made != 0
    else 0
)

data_per_call = (
    data_used / calls_made
    if calls_made != 0
    else 0
)

salary_per_age = (
    estimated_salary / age
    if age != 0
    else 0
)

dependents_per_age = (
    num_dependents / age
    if age != 0
    else 0
)

calls_sms_ratio = (
    calls_made / sms_sent
    if sms_sent != 0
    else 0
)

communication_intensity = (
    total_communication
    / (data_used + 1)
)

avg_communication = (
    total_communication / 2
)

usage_per_age = (
    data_used / age
    if age != 0
    else 0
)

# ============================================================
# 10. CUSTOMER PROFILE
# ============================================================

customer = {

    "age": age,

    "num_dependents": num_dependents,

    "estimated_salary": estimated_salary,

    "calls_made": calls_made,

    "sms_sent": sms_sent,

    "data_used": data_used,

    "registration_year": registration_year,

    "registration_month": registration_month,

    "registration_quarter": registration_quarter,

    "total_communication": total_communication,

    "calls_per_data": calls_per_data,

    "sms_per_call": sms_per_call,

    "data_per_call": data_per_call,

    "salary_per_age": salary_per_age,

    "dependents_per_age": dependents_per_age,

    "calls_sms_ratio": calls_sms_ratio,

    "communication_intensity": communication_intensity,

    "avg_communication": avg_communication,

    "usage_per_age": usage_per_age
}

# ============================================================
# 11. BASIC ANALYTICAL RISK SCORE
# ============================================================

# IMPORTANT:
# This is NOT a trained ML probability.
# It is only a transparent analytical score based
# on the observed segment patterns from this project.

risk_score = 0

risk_reasons = []

# ------------------------------------------------------------
# Age
# ------------------------------------------------------------

if 36 <= age <= 45:

    risk_score += 2

    risk_reasons.append(
        "Customer belongs to the 36-45 age group, "
        "which had the highest observed age-group churn rate."
    )

# ------------------------------------------------------------
# Communication
# ------------------------------------------------------------

if total_communication < 75:

    risk_score += 1

    risk_reasons.append(
        "Customer communication level is below the "
        "overall stayed-customer average."
    )

# ------------------------------------------------------------
# Data usage
# ------------------------------------------------------------

if data_used < 5000:

    risk_score += 1

    risk_reasons.append(
        "Customer data usage is below approximately "
        "the overall customer average."
    )

# ------------------------------------------------------------
# Calls
# ------------------------------------------------------------

if calls_made < 50:

    risk_score += 1

    risk_reasons.append(
        "Customer call activity is below approximately "
        "the overall stayed-customer average."
    )

# ------------------------------------------------------------
# SMS
# ------------------------------------------------------------

if sms_sent < 25:

    risk_score += 1

    risk_reasons.append(
        "Customer SMS activity is below approximately "
        "the overall stayed-customer average."
    )

# ============================================================
# 12. RISK CATEGORY
# ============================================================

if risk_score >= 4:

    risk_level = "HIGH"

elif risk_score >= 2:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"

# Analytical score only
risk_percentage = min(
    95,
    20 + (risk_score * 15)
)

# ============================================================
# 13. DISPLAY CUSTOMER ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CUSTOMER RISK ANALYSIS")
print("=" * 70)

print()
print("Risk Level       :", risk_level)
print(
    "Analytical Score :",
    f"{risk_percentage:.1f}%"
)

print()
print("IMPORTANT:")
print(
    "This is an analytical risk estimate, NOT a calibrated "
    "machine-learning probability."
)

print()
print("Risk Factors:")

if risk_reasons:

    for reason in risk_reasons:

        print("-", reason)

else:

    print(
        "- No major risk pattern identified from the "
        "available project features."
    )

# ============================================================
# 14. PREPARE AI CONTEXT
# ============================================================

business_data = business_df.to_string(
    index=False
)

model_data = model_df.to_string(
    index=False
)

customer_data = json.dumps(
    customer,
    indent=2
)

risk_reason_text = "\n".join(
    "- " + reason
    for reason in risk_reasons
)

if not risk_reason_text:

    risk_reason_text = (
        "- No major risk pattern identified."
    )

# ============================================================
# 15. AI PROMPT
# ============================================================

prompt = f"""
You are an AI telecom customer retention analyst.

Analyze ONE telecom customer using ONLY the project information
provided below.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not claim that the customer will definitely churn.
3. Do not present the analytical score as a true probability.
4. The project's best ROC-AUC is approximately 0.5299.
5. Therefore, individual churn prediction is weak.
6. Clearly distinguish observed information from recommendations.
7. Give practical but cautious retention advice.
8. Use simple professional business English.

============================================================
PROJECT BUSINESS INSIGHTS
============================================================

{business_data}

============================================================
MODEL COMPARISON
============================================================

{model_data}

============================================================
CUSTOMER PROFILE
============================================================

{customer_data}

============================================================
ANALYTICAL RISK LEVEL
============================================================

Risk Level:
{risk_level}

Analytical Score:
{risk_percentage:.1f}%

Risk Factors:
{risk_reason_text}

============================================================
TASK
============================================================

Create a short customer risk assessment.

Use these sections:

1. CUSTOMER RISK SUMMARY

Explain the customer's analytical risk level.

2. WHY THIS CUSTOMER RECEIVED THIS RISK LEVEL

Explain the relevant customer characteristics.

3. BUSINESS INTERPRETATION

Explain what the telecom company should understand
from this assessment.

4. RETENTION ACTION

Give 3 practical retention actions.

5. MODEL LIMITATION

Clearly explain that the current project's predictive
models have weak separation and that this assessment
must not be treated as a highly reliable individual
churn prediction.

Keep the response concise and professional.
"""

# ============================================================
# 16. SEND TO OLLAMA
# ============================================================

print()
print("=" * 70)
print("SENDING CUSTOMER PROFILE TO LOCAL AI")
print("=" * 70)

print()
print("Please wait...")

try:

    response = requests.post(

        OLLAMA_URL,

        json={

            "model": OLLAMA_MODEL,

            "prompt": prompt,

            "stream": False,

            "options": {

                "temperature": 0.2,

                "num_predict": 700

            }

        },

        timeout=180

    )

    response.raise_for_status()

    result = response.json()

    ai_report = result.get(
        "response",
        ""
    ).strip()

    if not ai_report:

        print()
        print("ERROR: Ollama returned an empty response.")

        raise SystemExit(1)

except requests.exceptions.RequestException as e:

    print()
    print("ERROR while calling Ollama:")
    print(e)

    raise SystemExit(1)

# ============================================================
# 17. DISPLAY AI REPORT
# ============================================================

print()
print("=" * 70)
print("AI CUSTOMER RISK REPORT")
print("=" * 70)

print()
print(ai_report)

# ============================================================
# 18. SAVE REPORT
# ============================================================

try:

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI INDIVIDUAL CUSTOMER CHURN RISK REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "RISK LEVEL: "
            + risk_level
            + "\n"
        )

        file.write(
            "ANALYTICAL SCORE: "
            + f"{risk_percentage:.1f}%"
            + "\n\n"
        )

        file.write(
            "CUSTOMER PROFILE\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for key, value in customer.items():

            file.write(
                f"{key}: {value}\n"
            )

        file.write("\n")

        file.write(
            "AI ANALYSIS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        file.write(
            ai_report
        )

    print()
    print("=" * 70)
    print("REPORT SAVED SUCCESSFULLY!")
    print("=" * 70)

    print(
        "File:",
        output_file
    )

except Exception as e:

    print()
    print("ERROR saving report:")
    print(e)

    raise SystemExit(1)

# ============================================================
# 19. COMPLETION
# ============================================================

print()
print("=" * 70)
print("AI CUSTOMER RISK ANALYSIS COMPLETED!")
print("=" * 70)