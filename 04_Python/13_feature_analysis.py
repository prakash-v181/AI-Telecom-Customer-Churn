import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# MySQL Connection
username = "root"
password = "Sangvi@2026#"
host = "localhost"
database = "telecom_churn_db"

encoded_password = quote_plus(password)

engine = create_engine(
    f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

query = "SELECT * FROM customers"
df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

# --------------------------------------------------
# 1. Numerical Feature Correlation with Churn
# --------------------------------------------------

print("\nCorrelation with Churn:")

numerical_columns = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used",
    "total_communication",
    "registration_year",
    "registration_month",
    "registration_quarter"
]

correlation = (
    df[numerical_columns + ["churn"]]
    .corr()["churn"]
    .drop("churn")
    .sort_values(ascending=False)
)

print(correlation.round(4))

# --------------------------------------------------
# 2. Strongest Positive Relationships
# --------------------------------------------------

print("\nTop Positive Correlations with Churn:")

print(
    correlation
    .sort_values(ascending=False)
    .head(5)
    .round(4)
)

# --------------------------------------------------
# 3. Strongest Negative Relationships
# --------------------------------------------------

print("\nTop Negative Correlations with Churn:")

print(
    correlation
    .sort_values()
    .head(5)
    .round(4)
)

# --------------------------------------------------
# 4. Average Numerical Values by Churn
# --------------------------------------------------

print("\nAverage Numerical Features by Churn:")

average_by_churn = (
    df.groupby("churn")[numerical_columns]
    .mean()
    .round(2)
)

print(average_by_churn)

# --------------------------------------------------
# 5. Churn Rate by Usage Segment
# --------------------------------------------------

print("\nChurn Rate by Usage Segment:")

usage_analysis = (
    df.groupby("usage_segment")["churn"]
    .agg(["count", "sum", "mean"])
)

usage_analysis["churn_rate"] = (
    usage_analysis["mean"] * 100
).round(2)

usage_analysis = usage_analysis.drop(columns=["mean"])

print(
    usage_analysis
    .sort_values("churn_rate", ascending=False)
)

# --------------------------------------------------
# 6. Churn Rate by Telecom Partner
# --------------------------------------------------

print("\nChurn Rate by Telecom Partner:")

partner_analysis = (
    df.groupby("telecom_partner")["churn"]
    .agg(["count", "sum", "mean"])
)

partner_analysis["churn_rate"] = (
    partner_analysis["mean"] * 100
).round(2)

partner_analysis = partner_analysis.drop(columns=["mean"])

print(
    partner_analysis
    .sort_values("churn_rate", ascending=False)
)

# --------------------------------------------------
# 7. Save Feature Analysis
# --------------------------------------------------

correlation.to_csv(
    "../08_Reports/churn_feature_correlation.csv"
)

average_by_churn.to_csv(
    "../08_Reports/average_features_by_churn.csv"
)

usage_analysis.to_csv(
    "../08_Reports/feature_usage_analysis.csv"
)

partner_analysis.to_csv(
    "../08_Reports/feature_partner_analysis.csv"
)

print("\nFeature analysis completed successfully!")

engine.dispose()