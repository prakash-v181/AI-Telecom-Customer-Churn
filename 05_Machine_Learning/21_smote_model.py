import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# MySQL Connection
username = "root"
password = "Sangvi@2026#"
host = "localhost"
database = "telecom_churn_db"

encoded_password = quote_plus(password)

engine = create_engine(
    f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
)


# Load Data
query = "SELECT * FROM customers"
df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)


# Remove unnecessary columns
df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ]
)


# Features and Target
X = df.drop(columns=["churn"])
y = df["churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# Identify columns
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


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Original class distribution
print("\nOriginal Training Churn Distribution:")
print(y_train.value_counts())


# Preprocessing
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


# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("Processed Testing Data:")
print(X_test_processed.shape)


# Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nAfter SMOTE:")
print("X_train:", X_train_smote.shape)
print("y_train:", y_train_smote.shape)

print("\nSMOTE Churn Distribution:")
print(y_train_smote.value_counts())


# Train Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=7,
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("\nSMOTE Decision Tree model trained successfully!")


# Prediction
y_pred = model.predict(X_test_processed)


# Performance
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

print("\nSMOTE Decision Tree Performance:")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# Confusion Matrix
print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# Classification Report
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


# Save Results
results = pd.DataFrame({
    "Model": ["SMOTE Decision Tree"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

results.to_csv(
    "../08_Reports/smote_decision_tree_results.csv",
    index=False
)

print("\nSMOTE Decision Tree results saved successfully!")


# Close connection
engine.dispose()

print("\nSMOTE Decision Tree analysis completed successfully!")