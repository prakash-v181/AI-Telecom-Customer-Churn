import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Load Engineered Dataset
# ============================================================

input_file = "../08_Reports/engineered_customer_data.csv"

df = pd.read_csv(input_file)

print("Engineered dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# Features and Target
# ============================================================

X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# Identify Columns
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category", "str"]
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


# ============================================================
# Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

print(
    "\nImproved Random Forest model "
    "trained successfully!"
)


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(
    X_test_processed
)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# Metrics
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
# Performance
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED RANDOM FOREST PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("PR-AUC   :", round(pr_auc, 4))


# ============================================================
# Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# Classification Report
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
# Feature Importance
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
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
# Save Results
# ============================================================

results = pd.DataFrame({
    "Model": ["Improved Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "ROC-AUC": [roc_auc],
    "PR-AUC": [pr_auc]
})

results.to_csv(
    "../08_Reports/improved_random_forest_results.csv",
    index=False
)

feature_importance.to_csv(
    "../08_Reports/improved_random_forest_feature_importance.csv",
    index=False
)

print(
    "\nImproved Random Forest results saved successfully!"
)

print(
    "Feature importance saved successfully!"
)


print(
    "\nImproved Random Forest analysis "
    "completed successfully!"
)