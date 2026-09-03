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
print("Original Shape:", df.shape)


# ============================================================
# Remove Unnecessary Columns
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
# Feature Engineering
# ============================================================

# ------------------------------------------------------------
# 1. Calls per Data Used
# ------------------------------------------------------------

df["calls_per_data"] = (
    df["calls_made"] /
    (df["data_used"] + 1)
)


# ------------------------------------------------------------
# 2. SMS per Call
# ------------------------------------------------------------

df["sms_per_call"] = (
    df["sms_sent"] /
    (df["calls_made"] + 1)
)


# ------------------------------------------------------------
# 3. Data Used per Call
# ------------------------------------------------------------

df["data_per_call"] = (
    df["data_used"] /
    (df["calls_made"] + 1)
)


# ------------------------------------------------------------
# 4. Average Communication
# ------------------------------------------------------------

df["avg_communication"] = (
    df["total_communication"] / 3
)


# ------------------------------------------------------------
# 5. Communication Intensity
# ------------------------------------------------------------

df["communication_intensity"] = (
    df["calls_made"]
    + df["sms_sent"]
    + df["data_used"] / 1000
)


# ------------------------------------------------------------
# 6. Usage per Age
# ------------------------------------------------------------

df["usage_per_age"] = (
    df["data_used"] /
    (df["age"] + 1)
)


# ------------------------------------------------------------
# 7. Salary per Age
# ------------------------------------------------------------

df["salary_per_age"] = (
    df["estimated_salary"] /
    (df["age"] + 1)
)


# ------------------------------------------------------------
# 8. Dependents per Age
# ------------------------------------------------------------

df["dependents_per_age"] = (
    df["num_dependents"] /
    (df["age"] + 1)
)


# ------------------------------------------------------------
# 9. Calls + SMS Ratio
# ------------------------------------------------------------

df["calls_sms_ratio"] = (
    df["calls_made"] /
    (df["sms_sent"] + 1)
)


# ------------------------------------------------------------
# 10. Data Usage Category
# ------------------------------------------------------------

df["data_usage_level"] = pd.cut(
    df["data_used"],
    bins=[
        -float("inf"),
        3000,
        6000,
        9000,
        float("inf")
    ],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High"
    ]
)


# ============================================================
# Check Missing Values
# ============================================================

print("\nMissing Values After Feature Engineering:")

print(
    df.isnull().sum()
)


# ============================================================
# Check Infinite Values
# ============================================================

numeric_columns = df.select_dtypes(
    include=["number"]
).columns

df[numeric_columns] = df[numeric_columns].replace(
    [float("inf"), -float("inf")],
    0
)


# ============================================================
# Feature List
# ============================================================

print("\nNew Engineered Features:")

engineered_features = [
    "calls_per_data",
    "sms_per_call",
    "data_per_call",
    "avg_communication",
    "communication_intensity",
    "usage_per_age",
    "salary_per_age",
    "dependents_per_age",
    "calls_sms_ratio",
    "data_usage_level"
]

for feature in engineered_features:
    print("-", feature)


# ============================================================
# Final Shape
# ============================================================

print("\nFinal Shape:")
print(df.shape)


# ============================================================
# Display Sample
# ============================================================

print("\nSample of Engineered Data:")

print(
    df[
        [
            "age",
            "calls_made",
            "sms_sent",
            "data_used",
            "total_communication",
            "calls_per_data",
            "sms_per_call",
            "data_per_call",
            "communication_intensity",
            "usage_per_age",
            "churn"
        ]
    ].head(10)
)


# ============================================================
# Save Engineered Dataset
# ============================================================

output_file = (
    "../08_Reports/"
    "engineered_customer_data.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    "\nEngineered dataset saved successfully!"
)

print(
    "File:",
    output_file
)


# ============================================================
# Close Connection
# ============================================================

engine.dispose()

print(
    "\nFeature engineering completed successfully!"
)