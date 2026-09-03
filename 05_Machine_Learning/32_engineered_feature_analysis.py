import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# ============================================================
# MySQL CONNECTION
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
# LOAD DATA
# ============================================================

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Original Shape:", df.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)

print("\nShape after removing unnecessary columns:")
print(df.shape)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Avoid division by zero
df["calls_per_data"] = (
    df["calls_made"] / df["data_used"].replace(0, 1)
)

df["sms_per_call"] = (
    df["sms_sent"] / df["calls_made"].replace(0, 1)
)

df["data_per_call"] = (
    df["data_used"] / df["calls_made"].replace(0, 1)
)

df["avg_communication"] = (
    df["total_communication"] / 2
)

df["communication_intensity"] = (
    df["calls_made"] +
    df["sms_sent"] +
    df["data_used"] / 100
)

df["usage_per_age"] = (
    df["data_used"] / df["age"].replace(0, 1)
)

df["salary_per_age"] = (
    df["estimated_salary"] / df["age"].replace(0, 1)
)

df["dependents_per_age"] = (
    df["num_dependents"] / df["age"].replace(0, 1)
)

df["calls_sms_ratio"] = (
    df["calls_made"] / df["sms_sent"].replace(0, 1)
)


# ============================================================
# DATA USAGE LEVEL
# ============================================================

df["data_usage_level"] = pd.cut(
    df["data_used"],
    bins=[-float("inf"), 3000, 7000, float("inf")],
    labels=[
        "Low Data Usage",
        "Medium Data Usage",
        "High Data Usage"
    ]
)


# ============================================================
# ENGINEERED FEATURES LIST
# ============================================================

engineered_features = [
    "calls_per_data",
    "sms_per_call",
    "data_per_call",
    "avg_communication",
    "communication_intensity",
    "usage_per_age",
    "salary_per_age",
    "dependents_per_age",
    "calls_sms_ratio"
]


# ============================================================
# NUMERICAL FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("ENGINEERED NUMERICAL FEATURES CORRELATION WITH CHURN")
print("=" * 60)

correlation_results = (
    df[engineered_features + ["churn"]]
    .corr()["churn"]
    .drop("churn")
    .sort_values(ascending=False)
)

print(correlation_results)


# ============================================================
# ABSOLUTE CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("ABSOLUTE CORRELATION WITH CHURN")
print("=" * 60)

absolute_correlation = (
    correlation_results
    .abs()
    .sort_values(ascending=False)
)

print(absolute_correlation)


# ============================================================
# TOP POSITIVE RELATIONSHIPS
# ============================================================

print("\n" + "=" * 60)
print("TOP POSITIVE ENGINEERED FEATURES")
print("=" * 60)

positive_features = (
    correlation_results
    .sort_values(ascending=False)
    .head(5)
)

print(positive_features)


# ============================================================
# TOP NEGATIVE RELATIONSHIPS
# ============================================================

print("\n" + "=" * 60)
print("TOP NEGATIVE ENGINEERED FEATURES")
print("=" * 60)

negative_features = (
    correlation_results
    .sort_values(ascending=True)
    .head(5)
)

print(negative_features)


# ============================================================
# AVERAGE ENGINEERED FEATURES BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE ENGINEERED FEATURES BY CHURN")
print("=" * 60)

average_by_churn = (
    df.groupby("churn", observed=False)[engineered_features]
    .mean()
)

print(average_by_churn)


# ============================================================
# DIFFERENCE BETWEEN STAYED AND CHURNED
# ============================================================

print("\n" + "=" * 60)
print("ENGINEERED FEATURE DIFFERENCES")
print("=" * 60)

stayed_average = (
    df[df["churn"] == 0][engineered_features]
    .mean()
)

churned_average = (
    df[df["churn"] == 1][engineered_features]
    .mean()
)

difference = pd.DataFrame({
    "Stayed_Average": stayed_average,
    "Churned_Average": churned_average,
    "Difference": churned_average - stayed_average
})

print(difference)


# ============================================================
# CHURN RATE BY DATA USAGE LEVEL
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY DATA USAGE LEVEL")
print("=" * 60)

data_usage_churn = (
    df.groupby("data_usage_level", observed=False)["churn"]
    .agg(
        customers="count",
        churned="sum",
        churn_rate="mean"
    )
)

data_usage_churn["churn_rate"] = (
    data_usage_churn["churn_rate"] * 100
).round(2)

print(data_usage_churn)


# ============================================================
# SAVE CORRELATION RESULTS
# ============================================================

correlation_output = pd.DataFrame({
    "Feature": correlation_results.index,
    "Correlation_with_Churn": correlation_results.values,
    "Absolute_Correlation": correlation_results.abs().values
})

correlation_output.to_csv(
    "../08_Reports/engineered_feature_correlations.csv",
    index=False
)

print("\nEngineered feature correlation results saved successfully!")


# ============================================================
# SAVE ENGINEERED FEATURE DATA
# ============================================================

df.to_csv(
    "../08_Reports/engineered_feature_analysis_data.csv",
    index=False
)

print("Engineered feature analysis data saved successfully!")


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("ENGINEERED FEATURE ANALYSIS CONCLUSION")
print("=" * 60)

strongest_feature = absolute_correlation.index[0]
strongest_value = absolute_correlation.iloc[0]

print(
    f"Strongest engineered feature: "
    f"{strongest_feature}"
)

print(
    f"Absolute correlation with churn: "
    f"{strongest_value:.4f}"
)

if strongest_value < 0.05:
    print(
        "Conclusion: Engineered numerical features still "
        "show very weak direct correlation with churn."
    )

elif strongest_value < 0.20:
    print(
        "Conclusion: Engineered features show a weak "
        "relationship with churn."
    )

else:
    print(
        "Conclusion: Some engineered features show "
        "meaningful relationships with churn."
    )


# ============================================================
# CLOSE CONNECTION
# ============================================================

engine.dispose()

print("\nEngineered feature analysis completed successfully!")