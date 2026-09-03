import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus

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
# Correlation Analysis
# --------------------------------------------------

print("\nCorrelation with Churn:")

numeric_columns = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used",
    "registration_year",
    "registration_month",
    "registration_quarter",
    "total_communication",
    "churn"
]

correlation = df[numeric_columns].corr()["churn"].sort_values(
    ascending=False
)

print(correlation)

# Remove churn itself for the chart
churn_correlation = correlation.drop("churn")

# --------------------------------------------------
# Correlation Bar Chart
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    churn_correlation.index,
    churn_correlation.values
)

plt.title("Correlation of Numerical Features with Churn")
plt.xlabel("Features")
plt.ylabel("Correlation with Churn")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig("../08_Reports/churn_correlation.png")

plt.show()

engine.dispose()

print("\nCorrelation analysis completed successfully!")