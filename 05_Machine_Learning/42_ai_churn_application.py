import os
import requests
import pandas as pd

reports_folder = "../08_Reports/"

business_file = reports_folder + "final_churn_business_insights.csv"
model_file = reports_folder + "final_model_comparison.csv"
output_file = reports_folder + "ai_customer_churn_application_report.txt"

ollama_url = "http://localhost:11434/api/generate"
ollama_model = "llama3.2:3b"


print("=" * 70)
print("AI CUSTOMER CHURN APPLICATION")
print("=" * 70)

print()
print("AI Engine")
print("Ollama")

print("AI Model")
print(ollama_model)


print()
print("Checking Ollama connection")

try:

    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )

    response.raise_for_status()

    models = response.json().get("models", [])

    model_names = []

    for model in models:
        model_names.append(model.get("name", ""))

    if ollama_model not in model_names:

        print("Required Ollama model is not available")
        print("Please run the following command")
        print(
            '"C:\\Users\\praka\\AppData\\Local\\Programs\\Ollama\\ollama.exe" pull llama3.2:3b'
        )

        raise SystemExit(1)

    print("Ollama connection OK")
    print("Model available OK")

except Exception as error:

    print("Ollama connection failed")
    print(error)

    raise SystemExit(1)


print()
print("Loading project data")

try:

    business_df = pd.read_csv(business_file)

    print("Business insights loaded successfully")
    print("Business data shape")
    print(business_df.shape)

except Exception as error:

    print("Error loading business insights")
    print(error)

    raise SystemExit(1)


try:

    model_df = pd.read_csv(model_file)

    print("Model comparison loaded successfully")
    print("Model data shape")
    print(model_df.shape)

except Exception as error:

    print("Error loading model comparison")
    print(error)

    raise SystemExit(1)


print()
print("=" * 70)
print("PROJECT INFORMATION")
print("=" * 70)


total_customers = None
churned_customers = None
churn_rate = None


for index, row in business_df.iterrows():

    category = str(row.iloc[0]).lower()
    value = str(row.iloc[1])

    if "overall churn rate" in category:
        try:
            churn_rate = float(
                value.replace("%", "").strip()
            )
        except:
            pass


print()
print("Project data loaded successfully")


print()
print("=" * 70)
print("ENTER CUSTOMER INFORMATION")
print("=" * 70)

print()
print("Use values from your customer dataset")
print()


def get_number(message):

    while True:

        value = input(message)

        try:

            return float(value)

        except ValueError:

            print("Please enter a valid number")


def get_integer(message):

    while True:

        value = input(message)

        try:

            return int(value)

        except ValueError:

            print("Please enter a valid integer")


age = get_number("Age: ")

num_dependents = get_number(
    "Number of dependents: "
)

estimated_salary = get_number(
    "Estimated salary: "
)

calls_made = get_number(
    "Calls made: "
)

sms_sent = get_number(
    "SMS sent: "
)

data_used = get_number(
    "Data used: "
)

registration_year = get_integer(
    "Registration year: "
)

registration_month = get_integer(
    "Registration month from 1 to 12: "
)


if registration_month < 1 or registration_month > 12:

    print()
    print("Registration month must be between 1 and 12")

    raise SystemExit(1)


registration_quarter = (
    (registration_month - 1) // 3
) + 1


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
    total_communication / (data_used + 1)
)


avg_communication = (
    total_communication / 2
)


usage_per_age = (
    data_used / age
    if age != 0
    else 0
)


print()
print("=" * 70)
print("CUSTOMER PROFILE")
print("=" * 70)

print()
print("Age")
print(age)

print("Dependents")
print(num_dependents)

print("Estimated salary")
print(estimated_salary)

print("Calls made")
print(calls_made)

print("SMS sent")
print(sms_sent)

print("Data used")
print(data_used)

print("Registration year")
print(registration_year)

print("Registration month")
print(registration_month)

print("Registration quarter")
print(registration_quarter)

print("Total communication")
print(total_communication)


print()
print("=" * 70)
print("VERIFIED CUSTOMER ANALYSIS")
print("=" * 70)


risk_score = 0

risk_reasons = []


stayed_age = 45.53
churned_age = 45.32

stayed_calls = 50.57
churned_calls = 49.17

stayed_sms = 25.37
churned_sms = 24.85

stayed_data = 5099.32
churned_data = 5061.47

stayed_communication = 75.94
churned_communication = 74.03

stayed_salary = 84966.26
churned_salary = 84170.37


if 36 <= age <= 45:

    risk_score += 2

    risk_reasons.append(
        "The customer belongs to the 36 to 45 age group"
    )


if calls_made < stayed_calls:

    risk_score += 1

    risk_reasons.append(
        "Calls made are below the stayed customer average"
    )


if sms_sent < stayed_sms:

    risk_score += 1

    risk_reasons.append(
        "SMS activity is below the stayed customer average"
    )


if data_used < stayed_data:

    risk_score += 1

    risk_reasons.append(
        "Data usage is below the stayed customer average"
    )


if total_communication < stayed_communication:

    risk_score += 1

    risk_reasons.append(
        "Total communication is below the stayed customer average"
    )


if estimated_salary < stayed_salary:

    risk_score += 1

    risk_reasons.append(
        "Estimated salary is below the stayed customer average"
    )


if risk_score >= 5:

    risk_level = "HIGH"

elif risk_score >= 3:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


print()
print("Risk level")
print(risk_level)

print()
print("Risk score")
print(risk_score)


print()
print("Verified observations")


if risk_reasons:

    for reason in risk_reasons:

        print(reason)

else:

    print(
        "No major risk pattern was identified from the available features"
    )


print()
print("Customer compared with project averages")

print()
print("Customer age")
print(age)

print("Stayed customer average age")
print(stayed_age)

print("Churned customer average age")
print(churned_age)


print()
print("Customer calls")
print(calls_made)

print("Stayed customer average calls")
print(stayed_calls)

print("Churned customer average calls")
print(churned_calls)


print()
print("Customer SMS")
print(sms_sent)

print("Stayed customer average SMS")
print(stayed_sms)

print("Churned customer average SMS")
print(churned_sms)


print()
print("Customer data usage")
print(data_used)

print("Stayed customer average data usage")
print(stayed_data)

print("Churned customer average data usage")
print(churned_data)


print()
print("Customer total communication")
print(total_communication)

print("Stayed customer average communication")
print(stayed_communication)

print("Churned customer average communication")
print(churned_communication)


print()
print("=" * 70)
print("PREPARING AI ANALYSIS")
print("=" * 70)


customer_information = f"""
Customer information

Age {age}
Number of dependents {num_dependents}
Estimated salary {estimated_salary}
Calls made {calls_made}
SMS sent {sms_sent}
Data used {data_used}
Registration year {registration_year}
Registration month {registration_month}
Registration quarter {registration_quarter}
Total communication {total_communication}

Engineered values

Calls per data {calls_per_data}
SMS per call {sms_per_call}
Data per call {data_per_call}
Salary per age {salary_per_age}
Dependents per age {dependents_per_age}
Calls SMS ratio {calls_sms_ratio}
Communication intensity {communication_intensity}
Average communication {avg_communication}
Usage per age {usage_per_age}
"""


verified_information = f"""
Verified project observations

Stayed customer average age {stayed_age}
Churned customer average age {churned_age}

Stayed customer average calls {stayed_calls}
Churned customer average calls {churned_calls}

Stayed customer average SMS {stayed_sms}
Churned customer average SMS {churned_sms}

Stayed customer average data usage {stayed_data}
Churned customer average data usage {churned_data}

Stayed customer average total communication {stayed_communication}
Churned customer average total communication {churned_communication}

Stayed customer average salary {stayed_salary}
Churned customer average salary {churned_salary}

Analytical risk level {risk_level}

Analytical risk score {risk_score}

Verified risk observations

"""


for reason in risk_reasons:

    verified_information += reason + "\n"


model_information = model_df.to_string(
    index=False
)


prompt = f"""
You are a telecom customer retention analyst.

Analyze the customer using only the verified information below.

Do not invent facts.

Do not say that the customer will definitely churn.

Do not convert the analytical risk score into a real probability.

The best project ROC AUC is approximately 0.5299.

This means the current machine learning system has weak
predictive separation.

The Python program has already calculated the customer
comparisons.

You must not change those facts.

CUSTOMER INFORMATION

{customer_information}

VERIFIED PROJECT INFORMATION

{verified_information}

MODEL INFORMATION

{model_information}

Write the answer using these sections.

CUSTOMER RISK SUMMARY

Explain the risk level.

WHY THIS RISK LEVEL

Explain only the verified observations.

BUSINESS INTERPRETATION

Explain what the company should understand.

RETENTION ACTIONS

Give three practical actions.

MODEL LIMITATION

Explain why this is not a reliable individual churn probability.

Use simple professional English.

Do not claim that high data usage is low data usage.

Do not invent telecom partner information.

Do not invent gender information.

Do not invent state information.

Do not invent city information.
"""


print()
print("Sending verified customer information to local AI")
print("Please wait")


try:

    ai_response = requests.post(

        ollama_url,

        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 700
            }
        },

        timeout=180
    )

    ai_response.raise_for_status()

    result = ai_response.json()

    ai_report = result.get(
        "response",
        ""
    ).strip()

    if not ai_report:

        print()
        print("AI returned an empty response")

        raise SystemExit(1)

except Exception as error:

    print()
    print("Error while calling Ollama")
    print(error)

    raise SystemExit(1)


print()
print("=" * 70)
print("AI CUSTOMER ANALYSIS")
print("=" * 70)

print()
print(ai_report)


try:

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI CUSTOMER CHURN APPLICATION REPORT\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            "Risk level " + risk_level + "\n"
        )

        file.write(
            "Risk score " + str(risk_score) + "\n\n"
        )

        file.write(
            "CUSTOMER INFORMATION\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(customer_information)

        file.write("\n")

        file.write(
            "VERIFIED ANALYSIS\n"
        )

        file.write("=" * 70 + "\n\n")

        for reason in risk_reasons:

            file.write(
                reason + "\n"
            )

        file.write("\n")

        file.write(
            "AI ANALYSIS\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(ai_report)

    print()
    print("=" * 70)
    print("REPORT SAVED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("File")
    print(output_file)

except Exception as error:

    print()
    print("Error saving report")
    print(error)

    raise SystemExit(1)


print()
print("=" * 70)
print("AI CUSTOMER CHURN APPLICATION COMPLETED")
print("=" * 70)