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

print("Data loaded successfully!")
print("Shape:", df.shape)

# Estimated Salary by Churn

plt.figure(figsize=(7, 5))

df.boxplot(
    column="estimated_salary",
    by="churn"
)

plt.title("Estimated Salary by Churn")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Estimated Salary")

plt.xticks(
    [1, 2],
    ["Stayed (0)", "Churned (1)"]
)

plt.tight_layout()

plt.savefig(
    "../08_Reports/boxplot_estimated_salary.png"
)

plt.show()

engine.dispose()

print("Estimated salary box plot completed successfully!")