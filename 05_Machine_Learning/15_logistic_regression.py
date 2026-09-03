import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

# --------------------------------------------------
# Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# --------------------------------------------------
# Identify Columns
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
# Preprocessing
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
            StandardScaler(),
            numerical_columns
        )
    ]
)

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# Transform Data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("Processed Testing Data:")
print(X_test_processed.shape)

# --------------------------------------------------
# Logistic Regression Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)

print("\nLogistic Regression model trained successfully!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_processed)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nModel Performance:")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Stayed",
            "Churned"
        ],
        zero_division=0
    )
)

print("\nLogistic Regression analysis completed successfully!")

# --------------------------------------------------
# Close Connection
# --------------------------------------------------

engine.dispose()