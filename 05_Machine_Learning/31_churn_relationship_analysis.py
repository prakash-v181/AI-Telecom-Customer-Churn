import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# ============================================================
# MySQL Connection
# ============================================================

username = "root"
password = "Sangvi@2026#"
host = "localhost"
database = "telecom_churn_db"

encoded_password = quote_plus(password)

engine = create_engine(
    f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
)


# ============================================================
# Load Data
# ============================================================

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# Basic Churn Distribution
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["churn"].value_counts())

print("\nChurn Percentage:")

print(
    (df["churn"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# Function: Churn Rate by Category
# ============================================================

def analyze_categorical(column):

    print("\n" + "=" * 60)
    print("CHURN RATE BY", column.upper())
    print("=" * 60)

    result = (
        df.groupby(column)["churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .sort_values(
            by="churn_rate",
            ascending=False
        )
    )

    result["churn_rate"] = (
        result["churn_rate"] * 100
    ).round(2)

    print(result)

    return result


# ============================================================
# Categorical Analysis
# ============================================================

categorical_columns = [
    "telecom_partner",
    "gender",
    "age_group",
    "usage_segment",
    "state",
    "city"
]

categorical_results = {}

for column in categorical_columns:

    categorical_results[column] = analyze_categorical(
        column
    )


# ============================================================
# Numerical Feature Analysis
# ============================================================

numerical_columns = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used",
    "total_communication"
]


print("\n" + "=" * 60)
print("AVERAGE NUMERICAL FEATURES BY CHURN")
print("=" * 60)

numerical_analysis = (
    df.groupby("churn")[numerical_columns]
    .mean()
    .round(2)
)

print(numerical_analysis)


# ============================================================
# Numerical Difference Analysis
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL DIFFERENCE: CHURNED vs STAYED")
print("=" * 60)

stayed_average = (
    df[df["churn"] == 0][numerical_columns]
    .mean()
)

churned_average = (
    df[df["churn"] == 1][numerical_columns]
    .mean()
)

difference = (
    churned_average - stayed_average
).round(2)

difference_table = pd.DataFrame({
    "Stayed_Average": stayed_average.round(2),
    "Churned_Average": churned_average.round(2),
    "Difference": difference
})

print(difference_table)


# ============================================================
# Churn Rate by Data Usage
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY DATA USAGE QUARTILE")
print("=" * 60)

df["data_usage_quartile"] = pd.qcut(
    df["data_used"],
    q=4,
    duplicates="drop"
)

data_usage_analysis = (
    df.groupby(
        "data_usage_quartile",
        observed=True
    )["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

data_usage_analysis["churn_rate"] = (
    data_usage_analysis["churn_rate"] * 100
).round(2)

print(data_usage_analysis)


# ============================================================
# Churn Rate by Calls
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY CALLS QUARTILE")
print("=" * 60)

df["calls_quartile"] = pd.qcut(
    df["calls_made"],
    q=4,
    duplicates="drop"
)

calls_analysis = (
    df.groupby(
        "calls_quartile",
        observed=True
    )["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

calls_analysis["churn_rate"] = (
    calls_analysis["churn_rate"] * 100
).round(2)

print(calls_analysis)


# ============================================================
# Churn Rate by SMS
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY SMS QUARTILE")
print("=" * 60)

df["sms_quartile"] = pd.qcut(
    df["sms_sent"],
    q=4,
    duplicates="drop"
)

sms_analysis = (
    df.groupby(
        "sms_quartile",
        observed=True
    )["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

sms_analysis["churn_rate"] = (
    sms_analysis["churn_rate"] * 100
).round(2)

print(sms_analysis)


# ============================================================
# Churn Rate by Salary
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY SALARY QUARTILE")
print("=" * 60)

df["salary_quartile"] = pd.qcut(
    df["estimated_salary"],
    q=4,
    duplicates="drop"
)

salary_analysis = (
    df.groupby(
        "salary_quartile",
        observed=True
    )["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

salary_analysis["churn_rate"] = (
    salary_analysis["churn_rate"] * 100
).round(2)

print(salary_analysis)


# ============================================================
# Churn Rate by Age
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY AGE QUARTILE")
print("=" * 60)

df["age_quartile"] = pd.qcut(
    df["age"],
    q=4,
    duplicates="drop"
)

age_analysis = (
    df.groupby(
        "age_quartile",
        observed=True
    )["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

age_analysis["churn_rate"] = (
    age_analysis["churn_rate"] * 100
).round(2)

print(age_analysis)


# ============================================================
# Overall Correlation
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL CORRELATION WITH CHURN")
print("=" * 60)

correlation = (
    df[numerical_columns + ["churn"]]
    .corr()["churn"]
    .drop("churn")
    .sort_values(
        ascending=False
    )
)

print(correlation.round(4))


# ============================================================
# Strongest Positive and Negative Relationships
# ============================================================

print("\n" + "=" * 60)
print("STRONGEST POSITIVE RELATIONSHIPS")
print("=" * 60)

print(
    correlation.head(5).round(4)
)


print("\n" + "=" * 60)
print("STRONGEST NEGATIVE RELATIONSHIPS")
print("=" * 60)

print(
    correlation.tail(5).sort_values().round(4)
)


# ============================================================
# Find Highest and Lowest Churn Categories
# ============================================================

print("\n" + "=" * 60)
print("CATEGORY WITH HIGHEST CHURN")
print("=" * 60)

for column, result in categorical_results.items():

    highest = result.iloc[0]

    print(
        f"{column}: "
        f"{result.index[0]} "
        f"({highest['churn_rate']}%)"
    )


# ============================================================
# Save Numerical Analysis
# ============================================================

numerical_analysis.to_csv(
    "../08_Reports/churn_numerical_analysis.csv"
)

difference_table.to_csv(
    "../08_Reports/churn_numerical_difference.csv"
)

data_usage_analysis.to_csv(
    "../08_Reports/churn_data_usage_analysis.csv"
)

calls_analysis.to_csv(
    "../08_Reports/churn_calls_analysis.csv"
)

sms_analysis.to_csv(
    "../08_Reports/churn_sms_analysis.csv"
)

salary_analysis.to_csv(
    "../08_Reports/churn_salary_analysis.csv"
)

age_analysis.to_csv(
    "../08_Reports/churn_age_analysis.csv"
)

correlation.to_csv(
    "../08_Reports/churn_correlation.csv"
)


# ============================================================
# Final Conclusion
# ============================================================

print("\n" + "=" * 60)
print("CHURN RELATIONSHIP ANALYSIS CONCLUSION")
print("=" * 60)

print(
    "The analysis compares customer behavior and "
    "categorical segments between Stayed and Churned customers."
)

print(
    "\nUse these results to identify features with "
    "meaningful differences in churn rate."
)

print(
    "\nIf most churn rates are close to the overall "
    "19.4% churn rate, the dataset has weak predictive signal."
)

print(
    "\nIf specific features or segments show large "
    "differences, those features can be used for "
    "further feature engineering and model improvement."
)


# ============================================================
# Close Connection
# ============================================================

engine.dispose()

print(
    "\nChurn relationship analysis completed successfully!"
)