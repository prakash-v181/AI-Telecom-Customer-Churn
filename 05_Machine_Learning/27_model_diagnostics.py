import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


# ============================================================
# MySQL Connection
# ============================================================

username = "root"
password = "Sangvi@2026#"
host = "localhost"
database = "telecom_churn_db"

encoded_password = quote_plus(password)

engine = create_engine(
    f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
)


# ============================================================
# Load Data
# ============================================================

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# Remove Unnecessary Columns
# ============================================================

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)


# ============================================================
# Features and Target
# ============================================================

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# Churn Distribution
# ============================================================

print("\nOverall Churn Distribution:")
print(y.value_counts())

print("\nOverall Churn Percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# Identify Columns
# ============================================================

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


# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# Preprocessing
# ============================================================

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


# ============================================================
# Transform Data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("Processed Testing Data:")
print(X_test_processed.shape)


# ============================================================
# Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest diagnostic model trained successfully!")


# ============================================================
# Probability Prediction
# ============================================================

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# AUC Diagnostics
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n" + "=" * 55)
print("MODEL DIAGNOSTICS")
print("=" * 55)

print("\nROC-AUC :", round(roc_auc, 4))
print("PR-AUC  :", round(pr_auc, 4))


# ============================================================
# Feature Importance
# ============================================================

feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# Top 20 Features
# ============================================================

print("\n" + "=" * 55)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 55)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# Numerical Feature Importance
# ============================================================

numerical_importance = feature_importance[
    feature_importance["Feature"].str.startswith(
        "numerical__"
    )
].copy()

print("\n" + "=" * 55)
print("NUMERICAL FEATURE IMPORTANCE")
print("=" * 55)

print(
    numerical_importance.to_string(
        index=False
    )
)


# ============================================================
# Total Feature Importance
# ============================================================

total_importance = feature_importance[
    "Importance"
].sum()

print("\nTotal Feature Importance:")
print(round(total_importance, 4))


# ============================================================
# Save Feature Importance
# ============================================================

feature_importance.to_csv(
    "../08_Reports/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")


# ============================================================
# Diagnostic Conclusion
# ============================================================

print("\n" + "=" * 55)
print("DIAGNOSTIC CONCLUSION")
print("=" * 55)

if roc_auc < 0.55:
    print(
        "WARNING: ROC-AUC is close to random prediction."
    )
    print(
        "The current features have weak predictive power "
        "for churn."
    )

elif roc_auc < 0.70:
    print(
        "The model has limited predictive power."
    )

else:
    print(
        "The model shows useful predictive power."
    )


if pr_auc < 0.25:
    print(
        "PR-AUC is low, especially for the churn class."
    )

print(
    "\nNext step should focus on feature quality, "
    "feature engineering, and data relationships."
)


# ============================================================
# Close Connection
# ============================================================

engine.dispose()

print("\nModel diagnostics completed successfully!")