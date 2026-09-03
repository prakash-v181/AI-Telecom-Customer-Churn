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

# Churn rate by registration year
year_churn = (
    df.groupby("registration_year")["churn"]
    .mean()
    .mul(100)
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    year_churn.index.astype(str),
    year_churn.values
)

plt.title("Churn Rate by Registration Year")
plt.xlabel("Registration Year")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_registration_year.png")

plt.show()

engine.dispose()

print("Registration year churn analysis completed successfully!")