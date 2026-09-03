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

# Churn rate by gender
gender_churn = (
    df.groupby("gender")["churn"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    gender_churn.index,
    gender_churn.values
)

plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig("../08_Reports/churn_by_gender.png")

plt.show()

engine.dispose()

print("Gender churn analysis completed successfully!")