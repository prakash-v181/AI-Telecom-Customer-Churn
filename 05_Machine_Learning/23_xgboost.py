import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. MySQL Connection
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
# 2. Load Data
# --------------------------------------------------

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)


# --------------------------------------------------
# 3. Remove Unnecessary Columns
# --------------------------------------------------

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)


# --------------------------------------------------
# 4. Features and Target
# --------------------------------------------------

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# --------------------------------------------------
# 5. Identify Columns
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
# 6. Train Test Split
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


# --------------------------------------------------
# 7. Class Distribution
# --------------------------------------------------

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 8. Preprocessing
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
# 9. Transform Data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("Processed Testing Data:")
print(X_test_processed.shape)


# --------------------------------------------------
# 10. Calculate Class Imbalance
# --------------------------------------------------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nScale Pos Weight:")
print(round(scale_pos_weight, 4))


# --------------------------------------------------
# 11. XGBoost Model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 12. Train Model
# --------------------------------------------------

model.fit(
    X_train_processed,
    y_train
)

print("\nXGBoost model trained successfully!")


# --------------------------------------------------
# 13. Prediction
# --------------------------------------------------

y_pred = model.predict(
    X_test_processed
)


# --------------------------------------------------
# 14. Model Performance
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

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

print("\nXGBoost Performance:")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# --------------------------------------------------
# 15. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# --------------------------------------------------
# 16. Classification Report
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


# --------------------------------------------------
# 17. Save Results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

results.to_csv(
    "../08_Reports/xgboost_results.csv",
    index=False
)

print("\nXGBoost results saved successfully!")


# --------------------------------------------------
# 18. Close Connection
# --------------------------------------------------

engine.dispose()

print("\nXGBoost analysis completed successfully!")