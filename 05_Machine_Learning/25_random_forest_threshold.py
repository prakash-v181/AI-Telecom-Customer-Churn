import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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
# 7. Preprocessing
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
# 8. Transform Data
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
# 9. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest model trained successfully!")


# --------------------------------------------------
# 10. Get Churn Probabilities
# --------------------------------------------------

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# --------------------------------------------------
# 11. Threshold Analysis
# --------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

results = []

print("\nRandom Forest Threshold Analysis:")

print(
    "\nThreshold | Accuracy | Precision | Recall | F1 Score"
)

print("-" * 60)


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

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

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(
        f"{threshold:9.2f} | "
        f"{accuracy:8.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:8.4f}"
    )


# --------------------------------------------------
# 12. Results DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 13. Best F1 Threshold
# --------------------------------------------------

best_f1_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_threshold = best_f1_row["Threshold"]

print("\nBest Threshold Based on F1 Score:")

print(
    "Threshold:",
    best_threshold
)

print(
    "Accuracy:",
    round(best_f1_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_f1_row["Precision"], 4)
)

print(
    "Recall:",
    round(best_f1_row["Recall"], 4)
)

print(
    "F1 Score:",
    round(best_f1_row["F1 Score"], 4)
)


# --------------------------------------------------
# 14. Final Prediction
# --------------------------------------------------

y_pred_best = (
    y_probability >= best_threshold
).astype(int)


# --------------------------------------------------
# 15. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix at Best Threshold:")

cm = confusion_matrix(
    y_test,
    y_pred_best
)

print(cm)


# --------------------------------------------------
# 16. Save Results
# --------------------------------------------------

results_df.to_csv(
    "../08_Reports/random_forest_threshold_results.csv",
    index=False
)

print(
    "\nRandom Forest threshold results saved successfully!"
)


# --------------------------------------------------
# 17. Plot F1 Score
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["F1 Score"],
    marker="o"
)

plt.xlabel("Prediction Threshold")
plt.ylabel("F1 Score")
plt.title("Random Forest F1 Score vs Prediction Threshold")

plt.grid(True)

plt.savefig(
    "../08_Reports/random_forest_f1_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 18. Close Connection
# --------------------------------------------------

engine.dispose()

print(
    "\nRandom Forest threshold tuning completed successfully!"
)