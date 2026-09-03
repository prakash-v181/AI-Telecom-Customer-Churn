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

# Churn rate by city
city_churn = (
    df.groupby("city")["churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    city_churn.index,
    city_churn.values
)

plt.title("Churn Rate by City")
plt.xlabel("City")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_city.png")

plt.show()

engine.dispose()

print("City churn analysis completed successfully!")