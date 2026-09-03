import pandas as pd
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

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nChurn Distribution:")
print(df["churn"].value_counts())

print("\nChurn Percentage:")
print(df["churn"].value_counts(normalize=True).mul(100).round(2))

import matplotlib.pyplot as plt

churn_counts = df["churn"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Stayed (0)", "Churned (1)"], churn_counts.values)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("../08_Reports/churn_distribution.png")
plt.show()

print("\nNumerical Statistics:")
print(df.describe())