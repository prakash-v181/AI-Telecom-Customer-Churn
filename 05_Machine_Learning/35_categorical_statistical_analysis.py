import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


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
print("Shape:", df.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "customer_id",
        "pincode",
        "date_of_registration"
    ],
    errors="ignore"
)

print("\nShape after removing unnecessary columns:")
print(df.shape)


# ============================================================
# CHECK TARGET
# ============================================================

if "churn" not in df.columns:
    raise ValueError("Target column 'churn' not found.")


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["churn"].value_counts())

print("\nChurn Percentage:")

print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

print(categorical_columns)


# ============================================================
# CHI-SQUARE ANALYSIS
# ============================================================

results = []

for feature in categorical_columns:

    print("\n" + "=" * 60)
    print("ANALYZING:", feature)
    print("=" * 60)

    # Create contingency table
    contingency_table = pd.crosstab(
        df[feature],
        df["churn"]
    )

    print("\nContingency Table:")
    print(contingency_table)

    # Chi-square test
    chi2, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    # Cramer's V
    n = contingency_table.to_numpy().sum()

    min_dimension = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    )

    if min_dimension > 0 and n > 0:
        cramers_v = np.sqrt(
            chi2 / (n * min_dimension)
        )
    else:
        cramers_v = 0

    results.append({
        "Feature": feature,
        "Chi_Square": chi2,
        "Degrees_of_Freedom": degrees_of_freedom,
        "P_Value": p_value,
        "Cramers_V": cramers_v
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

reject, adjusted_p_values, _, _ = multipletests(
    results_df["P_Value"],
    alpha=0.05,
    method="fdr_bh"
)

results_df["Adjusted_P_Value"] = adjusted_p_values

results_df["Significant"] = np.where(
    reject,
    "Yes",
    "No"
)


# ============================================================
# SORT RESULTS
# ============================================================

results_df = results_df.sort_values(
    by="Adjusted_P_Value"
).reset_index(drop=True)


# ============================================================
# ROUND RESULTS FOR DISPLAY
# ============================================================

display_df = results_df.copy()

display_df["Chi_Square"] = display_df[
    "Chi_Square"
].round(4)

display_df["P_Value"] = display_df[
    "P_Value"
].round(6)

display_df["Adjusted_P_Value"] = display_df[
    "Adjusted_P_Value"
].round(6)

display_df["Cramers_V"] = display_df[
    "Cramers_V"
].round(4)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL STATISTICAL ANALYSIS")
print("=" * 60)

print(
    display_df.to_string(index=False)
)


# ============================================================
# CRAMER'S V INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("CRAMER'S V INTERPRETATION")
print("=" * 60)

print("0.00 - 0.09 : Negligible association")
print("0.10 - 0.19 : Weak association")
print("0.20 - 0.29 : Moderate association")
print("0.30+       : Strong association")


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant_features = results_df[
    results_df["Significant"] == "Yes"
]

print("\n" + "=" * 60)
print("STATISTICALLY SIGNIFICANT CATEGORICAL FEATURES")
print("=" * 60)

if len(significant_features) > 0:

    print(
        significant_features[
            [
                "Feature",
                "Chi_Square",
                "P_Value",
                "Adjusted_P_Value",
                "Cramers_V"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "No statistically significant categorical "
        "features found at alpha = 0.05."
    )


# ============================================================
# STRONGEST ASSOCIATIONS
# ============================================================

print("\n" + "=" * 60)
print("STRONGEST CATEGORICAL ASSOCIATIONS")
print("=" * 60)

association_df = results_df.sort_values(
    by="Cramers_V",
    ascending=False
)

print(
    association_df[
        [
            "Feature",
            "Cramers_V",
            "P_Value",
            "Adjusted_P_Value"
        ]
    ].to_string(index=False)
)


# ============================================================
# CHURN RATE BY CATEGORY
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY CATEGORY")
print("=" * 60)

category_results = []

for feature in categorical_columns:

    churn_rates = (
        df.groupby(feature)["churn"]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean"
        )
        .reset_index()
    )

    churn_rates["churn_rate"] = (
        churn_rates["churn_rate"] * 100
    ).round(2)

    churn_rates["Feature"] = feature

    category_results.append(
        churn_rates
    )

    print("\n" + "-" * 60)
    print("Feature:", feature)
    print("-" * 60)

    print(
        churn_rates.to_string(index=False)
    )


# ============================================================
# COMBINE CATEGORY RESULTS
# ============================================================

category_results_df = pd.concat(
    category_results,
    ignore_index=True
)


# ============================================================
# FIND HIGHEST CHURN CATEGORY
# ============================================================

print("\n" + "=" * 60)
print("HIGHEST CHURN CATEGORY")
print("=" * 60)

for feature in categorical_columns:

    temp = category_results_df[
        category_results_df["Feature"] == feature
    ]

    if len(temp) > 0:

        row = temp.loc[
            temp["churn_rate"].idxmax()
        ]

        category_name = row[feature]

        print(
            f"{feature}: "
            f"{category_name} "
            f"({row['churn_rate']:.2f}%)"
        )


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL STATISTICAL ANALYSIS CONCLUSION")
print("=" * 60)

if len(significant_features) > 0:

    print(
        "Some categorical features show a statistically "
        "significant relationship with churn."
    )

    print(
        "These features may provide useful predictive "
        "information for customer churn modeling."
    )

else:

    print(
        "No categorical features show a statistically "
        "significant relationship with churn after "
        "multiple-comparison correction."
    )

    print(
        "This indicates that categorical features also "
        "have weak statistical separation between "
        "Stayed and Churned customers."
    )


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

statistical_output = (
    "../08_Reports/categorical_statistical_analysis.csv"
)

results_df.to_csv(
    statistical_output,
    index=False
)

print(
    "\nStatistical results saved successfully!"
)

print("File:", statistical_output)


# ============================================================
# SAVE CHURN RATE RESULTS
# ============================================================

category_output = (
    "../08_Reports/categorical_churn_rates.csv"
)

category_results_df.to_csv(
    category_output,
    index=False
)

print(
    "Categorical churn-rate results saved successfully!"
)

print("File:", category_output)


# ============================================================
# CLOSE CONNECTION
# ============================================================

engine.dispose()

print(
    "\nCategorical statistical analysis completed successfully!"
)