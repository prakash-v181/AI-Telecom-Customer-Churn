import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --------------------------------------------------
# MySQL Connection
# --------------------------------------------------

username = "root"
password = "Sangvi@2026#"
host = "localhost"
database = "telecom_churn_db"

encoded_password = quote_plus(password)

engine = create_engine(
    f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

# --------------------------------------------------
# Remove Unnecessary Columns
# --------------------------------------------------

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)

print("\nColumns after removing unnecessary columns:")
print(df.columns.tolist())

# --------------------------------------------------
# Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# --------------------------------------------------
# Identify Categorical and Numerical Columns
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)

# --------------------------------------------------
# One-Hot Encoding
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# --------------------------------------------------
# Fit Preprocessor on Training Data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

print("\nData preprocessing completed successfully!")

engine.dispose()