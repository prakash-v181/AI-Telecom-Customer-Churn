import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# Churn Distribution
# ============================================================

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())


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
# Transform Training Data
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)


# ============================================================
# Transform Testing Data
# ============================================================

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# Processed Data Shape
# ============================================================

print("\nProcessed Training Data:")
print(X_train_processed.shape)

print("\nProcessed Testing Data:")
print(X_test_processed.shape)


# ============================================================
# Feature Names
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)

print("\nTotal Processed Features:")
print(len(feature_names))


# ============================================================
# Save Processed Feature Names
# ============================================================

feature_info = pd.DataFrame({
    "Feature": feature_names
})

feature_info.to_csv(
    "../08_Reports/improved_feature_names.csv",
    index=False
)

print(
    "\nImproved feature names saved successfully!"
)


print(
    "\nImproved preprocessing completed successfully!"
)