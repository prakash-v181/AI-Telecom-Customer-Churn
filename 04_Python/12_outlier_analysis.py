import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# MySQL Connection
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
# Outlier Analysis using IQR
# --------------------------------------------------

numerical_columns = [
    "age",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used",
    "total_communication"
]

print("\nOutlier Analysis:")

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}:")
    print("Q1:", round(Q1, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower Bound:", round(lower_bound, 2))
    print("Upper Bound:", round(upper_bound, 2))
    print("Number of Outliers:", len(outliers))

print("\nOutlier analysis completed successfully!")

engine.dispose()