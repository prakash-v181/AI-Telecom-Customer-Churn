# ============================================================
# 39_ai_churn_chatbot.py
# AI-Powered Customer Churn Chatbot
# Uses Ollama + Llama 3.2:3b
# ============================================================

import os
import requests
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

business_report_file = (
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
print("AI CUSTOMER CHURN CHATBOT")
print("=" * 70)

print()
print("AI Engine : Ollama")
print("AI Model  :", OLLAMA_MODEL)
print()


# ============================================================
# 4. CHECK OLLAMA
# ============================================================

try:

    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=5
    )

    if response.status_code != 200:

        print("ERROR: Ollama is not responding correctly.")
        raise SystemExit(1)

    models_data = response.json()

    available_models = [
        model["name"]
        for model in models_data.get("models", [])
    ]

    if OLLAMA_MODEL not in available_models:

        print("ERROR: Required Ollama model was not found.")
        print()
        print("Required model:", OLLAMA_MODEL)
        print()
        print("Available models:")

        for model in available_models:
            print("-", model)

        print()
        print("Run:")
        print(
            '"C:\\Users\\praka\\AppData\\Local\\Programs\\Ollama\\ollama.exe" '
            'pull llama3.2:3b'
        )

        raise SystemExit(1)

    print("Ollama connection: OK")
    print("Model available: OK")

except requests.exceptions.ConnectionError:

    print("ERROR: Cannot connect to Ollama.")
    print()
    print("Make sure Ollama is running.")
    print()
    print("You can test it with:")
    print(
        '"C:\\Users\\praka\\AppData\\Local\\Programs\\Ollama\\ollama.exe" list'
    )

    raise SystemExit(1)

except Exception as e:

    print("ERROR checking Ollama:")
    print(e)

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
# 7. LOAD AI BUSINESS REPORT
# ============================================================

try:

    if os.path.exists(business_report_file):

        with open(
            business_report_file,
            "r",
            encoding="utf-8"
        ) as file:

            ai_business_report = file.read()

        print()
        print("AI business report loaded successfully!")

    else:

        ai_business_report = ""

        print()
        print("AI business report not found.")
        print("Continuing with CSV data.")

except Exception as e:

    print()
    print("WARNING loading AI business report:")
    print(e)

    ai_business_report = ""


# ============================================================
# 8. CONVERT DATA TO TEXT
# ============================================================

business_data = business_df.to_string(
    index=False
)

model_data = model_df.to_string(
    index=False
)


# ============================================================
# 9. CREATE PROJECT KNOWLEDGE
# ============================================================

project_knowledge = f"""

============================================================
TELECOM CUSTOMER CHURN PROJECT DATA
============================================================

BUSINESS INSIGHTS
============================================================

{business_data}


============================================================
MODEL COMPARISON
============================================================

{model_data}


============================================================
AI BUSINESS REPORT
============================================================

{ai_business_report}


============================================================
IMPORTANT PROJECT FACTS
============================================================

Total Customers: 2026

Stayed Customers: 1633

Churned Customers: 393

Overall Churn Rate: 19.4%

Highest Telecom Partner Churn:
Vodafone - 21.28%

Highest Gender Churn:
F - 20.30%

Highest Age Group Churn:
36-45 - 21.61%

Highest Usage Segment Churn:
Low Usage - 19.85%

Highest State Churn:
Bihar - 26.14%

Highest City Churn:
Kolkata - 22.39%

Best Accuracy:
SMOTE Decision Tree - 0.7635

Best Precision:
Improved Random Forest - 0.2429

Best Recall:
Random Forest Threshold Tuning - 1.0000

Best F1:
Random Forest Threshold Tuning - 0.3258

Best ROC-AUC:
Improved XGBoost - 0.5299

Best PR-AUC:
Improved XGBoost - 0.1998

IMPORTANT:
The project found weak predictive separation.

ROC-AUC around 0.53 is close to random classification.

The current dataset should NOT be described as providing
highly accurate individual churn prediction.

The AI must distinguish between:
1. Observed project facts
2. Interpretation
3. Business recommendations

Do not invent statistics.
Do not change numerical results.
"""


# ============================================================
# 10. SYSTEM INSTRUCTION
# ============================================================

system_instruction = """

You are an AI Customer Churn Analyst.

You are answering questions about a telecom customer churn
machine-learning project.

Use ONLY the supplied project information.

Rules:

1. Never invent statistics.
2. Never change the supplied numerical results.
3. If the user asks about a metric, use the exact project value.
4. Clearly distinguish facts from recommendations.
5. Do not claim the model is highly accurate.
6. Explain technical concepts in simple professional English.
7. If information is not available, say:
   "This information is not available in the current dataset."
8. Remember that ROC-AUC = 0.5299 indicates weak predictive
   separation and is close to random classification.
9. Accuracy alone should not be used to claim that a model
   is good for churn prediction.
10. For churn prediction, explain why Recall, Precision,
    F1, ROC-AUC and PR-AUC matter.
11. When discussing business segments, explain that a higher
    churn rate identifies an area for investigation, not
    necessarily the cause of churn.
12. Do not claim that correlation proves causation.
"""


# ============================================================
# 11. FUNCTION TO ASK OLLAMA
# ============================================================

def ask_ollama(question):

    prompt = f"""
{system_instruction}

{project_knowledge}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER
============================================================

Give a clear and useful answer.

If appropriate:
- mention the exact numbers
- explain what they mean
- explain the business implication
- avoid unsupported assumptions

Answer in professional but simple English.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return result.get(
            "response",
            "No response received from AI."
        )

    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Cannot connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            "ERROR: AI response timed out. "
            "Please try again."
        )

    except Exception as e:

        return f"ERROR while communicating with Ollama: {e}"


# ============================================================
# 12. CHATBOT INTRODUCTION
# ============================================================

print()
print("=" * 70)
print("CHATBOT READY")
print("=" * 70)

print()
print("Ask questions about your customer churn project.")
print()
print("Examples:")
print("1. Why is churn prediction weak?")
print("2. Which state has the highest churn?")
print("3. Which model has the best ROC-AUC?")
print("4. Which model has the best recall?")
print("5. What are the main business risks?")
print("6. Explain the project to a manager.")
print("7. What additional data should we collect?")
print()
print("Type 'exit' to stop.")
print()


# ============================================================
# 13. CHAT LOOP
# ============================================================

while True:

    print("-" * 70)

    question = input("You: ").strip()

    if not question:

        print("Please enter a question.")
        continue

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print()
        print("=" * 70)
        print("AI CUSTOMER CHURN CHATBOT CLOSED")
        print("=" * 70)

        break

    print()
    print("AI: Thinking...")
    print()

    answer = ask_ollama(question)

    print(answer)
    print()