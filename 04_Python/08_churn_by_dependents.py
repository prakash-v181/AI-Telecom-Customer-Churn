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

# Churn rate by number of dependents
dependent_churn = (
    df.groupby("num_dependents")["churn"]
    .mean()
    .mul(100)
)

dependent_churn = dependent_churn.sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    dependent_churn.index.astype(str),
    dependent_churn.values
)

plt.title("Churn Rate by Number of Dependents")
plt.xlabel("Number of Dependents")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_dependents.png")

plt.show()

engine.dispose()

print("Dependents churn analysis completed successfully!")