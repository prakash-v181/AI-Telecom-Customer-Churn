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

# Churn rate by usage segment
usage_churn = (
    df.groupby("usage_segment")["churn"]
    .mean()
    .mul(100)
)

# Keep segments in logical order
usage_order = [
    "Low Usage",
    "Medium Usage",
    "High Usage"
]

usage_churn = usage_churn.reindex(usage_order)

plt.figure(figsize=(8, 5))

plt.bar(
    usage_churn.index,
    usage_churn.values
)

plt.title("Churn Rate by Usage Segment")
plt.xlabel("Usage Segment")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_usage_segment.png")

plt.show()

engine.dispose()

print("Usage segment churn analysis completed successfully!")