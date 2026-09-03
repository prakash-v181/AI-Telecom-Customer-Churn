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

# Churn rate by state
state_churn = (
    df.groupby("state")["churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

plt.bar(
    state_churn.index,
    state_churn.values
)

plt.title("Top 10 States by Churn Rate")
plt.xlabel("State")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_state.png")

plt.show()

engine.dispose()

print("State churn analysis completed successfully!")