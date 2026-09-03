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
    classification_report,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# MYSQL CONNECTION
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
# LOAD DATA
# ============================================================

query = "SELECT * FROM customers"

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Original Shape:", df.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Avoid division by zero

df["calls_per_data"] = (
    df["calls_made"] /
    df["data_used"].replace(0, 1)
)

df["sms_per_call"] = (
    df["sms_sent"] /
    df["calls_made"].replace(0, 1)
)

df["data_per_call"] = (
    df["data_used"] /
    df["calls_made"].replace(0, 1)
)

df["avg_communication"] = (
    df["total_communication"] / 2
)

df["communication_intensity"] = (
    df["calls_made"] +
    df["sms_sent"] +
    df["data_used"] / 100
)

df["usage_per_age"] = (
    df["data_used"] /
    df["age"].replace(0, 1)
)

df["salary_per_age"] = (
    df["estimated_salary"] /
    df["age"].replace(0, 1)
)

df["dependents_per_age"] = (
    df["num_dependents"] /
    df["age"].replace(0, 1)
)

df["calls_sms_ratio"] = (
    df["calls_made"] /
    df["sms_sent"].replace(0, 1)
)


# ============================================================
# DATA USAGE LEVEL
# ============================================================

df["data_usage_level"] = pd.cut(
    df["data_used"],
    bins=[
        -float("inf"),
        3000,
        7000,
        float("inf")
    ],
    labels=[
        "Low Data Usage",
        "Medium Data Usage",
        "High Data Usage"
    ]
)


print("\nFeature engineering completed!")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "str", "category"]
).columns.tolist()


print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ============================================================
# TRAIN TEST SPLIT
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
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())


# ============================================================
# PREPROCESSING
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
# TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("\nProcessed Testing Data:")
print(X_test_processed.shape)


# ============================================================
# SCALE POSITIVE WEIGHT
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nScale Pos Weight:")
print(round(scale_pos_weight, 4))


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train_processed,
    y_train
)

print("\nImproved XGBoost model trained successfully!")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# PERFORMANCE METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED XGBOOST PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("PR-AUC   :", round(pr_auc, 4))


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Improved XGBoost"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "ROC-AUC": [roc_auc],
    "PR-AUC": [pr_auc]
})

results.to_csv(
    "../08_Reports/improved_xgboost_results.csv",
    index=False
)

print("\nImproved XGBoost results saved successfully!")


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "../08_Reports/improved_xgboost_feature_importance.csv",
    index=False
)

print("Feature importance saved successfully!")


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED XGBOOST CONCLUSION")
print("=" * 60)

if roc_auc >= 0.70:
    print(
        "Strong predictive performance detected."
    )

elif roc_auc >= 0.60:
    print(
        "Moderate predictive performance detected."
    )

elif roc_auc >= 0.50:
    print(
        "Weak predictive performance detected."
    )

else:
    print(
        "ROC-AUC is below random baseline."
    )

print(
    "\nThe model uses engineered features and "
    "class weighting to improve churn prediction."
)


# ============================================================
# CLOSE CONNECTION
# ============================================================

engine.dispose()

print("\nImproved XGBoost analysis completed successfully!")