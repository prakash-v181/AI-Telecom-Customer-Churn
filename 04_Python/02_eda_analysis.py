import pandas as pd
import matplotlib.pyplot as plt
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

# Load data
query = "SELECT * FROM customers"
df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

# --------------------------------------------------
# 1. Churn Distribution
# --------------------------------------------------

print("\nChurn Distribution:")
print(df["churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# 2. Churn by Telecom Partner
# --------------------------------------------------

print("\nChurn by Telecom Partner:")

partner_churn = (
    df.groupby("telecom_partner")["churn"]
    .agg(["count", "sum", "mean"])
)

partner_churn["churn_rate"] = (
    partner_churn["mean"] * 100
).round(2)

partner_churn = partner_churn.drop(columns=["mean"])

print(partner_churn)

# --------------------------------------------------
# 3. Churn by Gender
# --------------------------------------------------

print("\nChurn by Gender:")

gender_churn = (
    df.groupby("gender")["churn"]
    .agg(["count", "sum", "mean"])
)

gender_churn["churn_rate"] = (
    gender_churn["mean"] * 100
).round(2)

gender_churn = gender_churn.drop(columns=["mean"])

print(gender_churn)

# --------------------------------------------------
# 4. Churn by Age Group
# --------------------------------------------------

print("\nChurn by Age Group:")

age_churn = (
    df.groupby("age_group")["churn"]
    .agg(["count", "sum", "mean"])
)

age_churn["churn_rate"] = (
    age_churn["mean"] * 100
).round(2)

age_churn = age_churn.drop(columns=["mean"])

print(age_churn)

# --------------------------------------------------
# 5. Churn by Usage Segment
# --------------------------------------------------

print("\nChurn by Usage Segment:")

usage_churn = (
    df.groupby("usage_segment")["churn"]
    .agg(["count", "sum", "mean"])
)

usage_churn["churn_rate"] = (
    usage_churn["mean"] * 100
).round(2)

usage_churn = usage_churn.drop(columns=["mean"])

print(usage_churn)

# --------------------------------------------------
# 6. Churn by City
# --------------------------------------------------

print("\nChurn by City:")

city_churn = (
    df.groupby("city")["churn"]
    .agg(["count", "sum", "mean"])
)

city_churn["churn_rate"] = (
    city_churn["mean"] * 100
).round(2)

city_churn = city_churn.drop(columns=["mean"])

print(city_churn.sort_values("churn_rate", ascending=False))

# --------------------------------------------------
# 7. Churn by State
# --------------------------------------------------

print("\nTop States by Churn Rate:")

state_churn = (
    df.groupby("state")["churn"]
    .agg(["count", "sum", "mean"])
)

state_churn["churn_rate"] = (
    state_churn["mean"] * 100
).round(2)

state_churn = state_churn.drop(columns=["mean"])

print(
    state_churn
    .sort_values("churn_rate", ascending=False)
    .head(10)
)

# --------------------------------------------------
# 8. Churn by Number of Dependents
# --------------------------------------------------

print("\nChurn by Number of Dependents:")

dependent_churn = (
    df.groupby("num_dependents")["churn"]
    .agg(["count", "sum", "mean"])
)

dependent_churn["churn_rate"] = (
    dependent_churn["mean"] * 100
).round(2)

dependent_churn = dependent_churn.drop(columns=["mean"])

print(dependent_churn)

# --------------------------------------------------
# 9. Churn by Registration Year
# --------------------------------------------------

print("\nChurn by Registration Year:")

year_churn = (
    df.groupby("registration_year")["churn"]
    .agg(["count", "sum", "mean"])
)

year_churn["churn_rate"] = (
    year_churn["mean"] * 100
).round(2)

year_churn = year_churn.drop(columns=["mean"])

print(year_churn)

# --------------------------------------------------
# 10. Save EDA Results
# --------------------------------------------------

partner_churn.to_csv(
    "../08_Reports/churn_by_partner.csv"
)

gender_churn.to_csv(
    "../08_Reports/churn_by_gender.csv"
)

age_churn.to_csv(
    "../08_Reports/churn_by_age_group.csv"
)

usage_churn.to_csv(
    "../08_Reports/churn_by_usage_segment.csv"
)

city_churn.to_csv(
    "../08_Reports/churn_by_city.csv"
)

state_churn.to_csv(
    "../08_Reports/churn_by_state.csv"
)

# --------------------------------------------------
# 11. Churn Rate by Telecom Partner
# --------------------------------------------------

partner_plot = partner_churn.sort_values(
    "churn_rate",
    ascending=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    partner_plot.index,
    partner_plot["churn_rate"]
)

plt.title("Churn Rate by Telecom Partner")
plt.xlabel("Telecom Partner")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    "../08_Reports/churn_rate_by_partner.png"
)

plt.show()

print("\nEDA analysis completed successfully!")

engine.dispose()

