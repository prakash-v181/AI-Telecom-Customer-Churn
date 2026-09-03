# ============================================================
# 38_ai_churn_analyst.py
# AI-Powered Customer Churn Analyst using Ollama
# ============================================================

import os
import json
import urllib.request
import urllib.error
import pandas as pd


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
    reports_folder + "ai_churn_analyst_report.txt"
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
print("AI CUSTOMER CHURN ANALYST")
print("=" * 70)

print("\nAI Engine: Ollama")
print("Model:", OLLAMA_MODEL)


# ============================================================
# 4. CHECK PROJECT FILES
# ============================================================

try:

    business_df = pd.read_csv(
        business_insights_file
    )

    print("\nBusiness insights loaded successfully!")
    print("Shape:", business_df.shape)

except Exception as e:

    print("\nERROR loading business insights:")
    print(e)

    raise SystemExit(1)


try:

    model_df = pd.read_csv(
        model_comparison_file
    )

    print("\nModel comparison loaded successfully!")
    print("Shape:", model_df.shape)

except Exception as e:

    print("\nERROR loading model comparison:")
    print(e)

    raise SystemExit(1)


# ============================================================
# 5. CONVERT DATA TO TEXT
# ============================================================

business_data = business_df.to_string(
    index=False
)

model_data = model_df.to_string(
    index=False
)


# ============================================================
# 6. AI PROMPT
# ============================================================

prompt = f"""
You are an AI Business Analyst specializing in
telecom customer churn.

Analyze the following customer churn project results.

IMPORTANT RULES:

- Use ONLY the information provided below.
- Do not invent customer behavior or statistics.
- Clearly distinguish observed facts from recommendations.
- Do not claim that the machine-learning model is highly accurate.
- The project found weak predictive separation.
- Explain the results in simple professional business language.
- Keep numerical values consistent with the supplied data.

============================================================
BUSINESS INSIGHTS DATA
============================================================

{business_data}


============================================================
MODEL COMPARISON DATA
============================================================

{model_data}


============================================================
YOUR TASK
============================================================

Create a professional AI-powered Customer Churn Analysis Report.

Include these sections:

1. EXECUTIVE SUMMARY

Explain:
- total customer situation
- overall churn
- major business finding
- machine-learning finding

2. CUSTOMER CHURN ANALYSIS

Identify:
- highest churn telecom partner
- highest churn gender
- highest churn age group
- highest churn usage segment
- highest churn state
- highest churn city

3. CUSTOMER BEHAVIOR

Explain:
- numerical feature differences
- engineered feature differences
- whether these differences appear strong or weak

4. MACHINE LEARNING ANALYSIS

Explain:
- best Accuracy model
- best Precision model
- best Recall model
- best F1 model
- best ROC-AUC model
- best PR-AUC model

Most importantly, explain why the ROC-AUC result
indicates weak predictive separation.

5. KEY BUSINESS RISKS

Identify the main areas that the telecom company
should investigate.

6. RETENTION RECOMMENDATIONS

Give practical recommendations based ONLY on
the available data.

7. DATA IMPROVEMENT RECOMMENDATIONS

Explain what additional features should ideally
be collected, such as:

- customer complaints
- network quality
- recharge frequency
- plan changes
- customer tenure
- customer support interactions

Do not state that these variables currently exist
in the dataset.

8. FINAL AI CONCLUSION

Give a clear conclusion explaining whether the
current dataset is suitable for reliable individual
churn prediction.

9. MANAGEMENT SUMMARY

Give 5 short points that a manager can understand
quickly.

Write the report in clear professional English.
"""


# ============================================================
# 7. SEND DATA TO LOCAL OLLAMA
# ============================================================

print("\nSending churn analysis to local AI...")
print("Please wait...")


payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False
}


try:

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=300
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    ai_report = result.get(
        "response",
        ""
    )

    if not ai_report:

        print("\nERROR:")
        print("Ollama returned an empty response.")

        raise SystemExit(1)


except urllib.error.URLError as e:

    print("\nERROR connecting to Ollama:")
    print(e)

    print("\nMake sure Ollama is running.")

    print(
        "\nYou can test it with:"
    )

    print(
        '"C:\\Users\\praka\\AppData\\Local\\Programs\\Ollama\\ollama.exe" list'
    )

    raise SystemExit(1)


except Exception as e:

    print("\nERROR while calling Ollama:")
    print(e)

    raise SystemExit(1)


# ============================================================
# 8. DISPLAY AI REPORT
# ============================================================

print("\n")
print("=" * 70)
print("AI CHURN ANALYST REPORT")
print("=" * 70)

print(ai_report)


# ============================================================
# 9. SAVE AI REPORT
# ============================================================

try:

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI CUSTOMER CHURN ANALYST REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "AI ENGINE: Ollama\n"
        )

        file.write(
            "MODEL: llama3.2:3b\n\n"
        )

        file.write(
            ai_report
        )

    print("\n")
    print("=" * 70)
    print("AI REPORT SAVED SUCCESSFULLY!")
    print("=" * 70)

    print(
        "File:",
        output_file
    )

except Exception as e:

    print("\nERROR saving AI report:")
    print(e)

    raise SystemExit(1)


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("AI CUSTOMER CHURN ANALYSIS COMPLETED!")
print("=" * 70)