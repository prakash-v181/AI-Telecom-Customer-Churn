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

query = "SELECT * FROM customers"
df = pd.read_sql(query, engine)

# Churn rate by age group
age_churn = (
    df.groupby("age_group")["churn"]
    .mean()
    .mul(100)
)

# Keep age groups in the correct order
age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]

age_churn = age_churn.reindex(age_order)

plt.figure(figsize=(8, 5))

plt.bar(
    age_churn.index,
    age_churn.values
)

plt.title("Churn Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_age_group.png")

plt.show()

engine.dispose()

print("Age group churn analysis completed successfully!")