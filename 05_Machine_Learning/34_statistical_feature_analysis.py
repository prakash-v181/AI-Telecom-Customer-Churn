import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from scipy.stats import ttest_ind
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
    raise ValueError("Target column 'churn' not found in dataset.")


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["churn"].value_counts())


# ============================================================
# IDENTIFY NUMERICAL FEATURES
# ============================================================

numerical_columns = df.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

# Remove target from feature list
if "churn" in numerical_columns:
    numerical_columns.remove("churn")


print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print(numerical_columns)


# ============================================================
# SPLIT STAYED AND CHURNED
# ============================================================

stayed = df[df["churn"] == 0]
churned = df[df["churn"] == 1]

print("\nStayed Customers:", len(stayed))
print("Churned Customers:", len(churned))


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

results = []

for feature in numerical_columns:

    stayed_values = stayed[feature].dropna()
    churned_values = churned[feature].dropna()

    # Means
    stayed_mean = stayed_values.mean()
    churned_mean = churned_values.mean()

    # Difference
    mean_difference = churned_mean - stayed_mean

    # Independent t-test
    t_statistic, p_value = ttest_ind(
        stayed_values,
        churned_values,
        equal_var=False,
        nan_policy="omit"
    )

    # Standard deviations
    stayed_std = stayed_values.std()
    churned_std = churned_values.std()

    # Pooled standard deviation
    n1 = len(stayed_values)
    n2 = len(churned_values)

    pooled_std = np.sqrt(
        (
            ((n1 - 1) * stayed_std ** 2)
            + ((n2 - 1) * churned_std ** 2)
        )
        / (n1 + n2 - 2)
    )

    # Cohen's d
    if pooled_std != 0:
        cohens_d = mean_difference / pooled_std
    else:
        cohens_d = 0

    results.append({
        "Feature": feature,
        "Stayed_Mean": stayed_mean,
        "Churned_Mean": churned_mean,
        "Mean_Difference": mean_difference,
        "T_Statistic": t_statistic,
        "P_Value": p_value,
        "Cohens_D": cohens_d
    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

reject, corrected_p_values, _, _ = multipletests(
    results_df["P_Value"],
    alpha=0.05,
    method="fdr_bh"
)

results_df["Adjusted_P_Value"] = corrected_p_values

results_df["Significant"] = np.where(
    reject,
    "Yes",
    "No"
)


# ============================================================
# SORT BY P-VALUE
# ============================================================

results_df = results_df.sort_values(
    by="Adjusted_P_Value"
).reset_index(drop=True)


# ============================================================
# ROUND VALUES
# ============================================================

display_df = results_df.copy()

display_df["Stayed_Mean"] = display_df["Stayed_Mean"].round(4)
display_df["Churned_Mean"] = display_df["Churned_Mean"].round(4)
display_df["Mean_Difference"] = display_df["Mean_Difference"].round(4)
display_df["T_Statistic"] = display_df["T_Statistic"].round(4)
display_df["P_Value"] = display_df["P_Value"].round(6)
display_df["Adjusted_P_Value"] = display_df[
    "Adjusted_P_Value"
].round(6)
display_df["Cohens_D"] = display_df["Cohens_D"].round(4)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL FEATURE ANALYSIS")
print("=" * 60)

print(display_df.to_string(index=False))


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant_features = results_df[
    results_df["Significant"] == "Yes"
]

print("\n" + "=" * 60)
print("STATISTICALLY SIGNIFICANT FEATURES")
print("=" * 60)

if len(significant_features) > 0:

    print(
        significant_features[
            [
                "Feature",
                "P_Value",
                "Adjusted_P_Value",
                "Cohens_D"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "No statistically significant numerical "
        "features found at alpha = 0.05."
    )


# ============================================================
# STRONGEST EFFECT SIZES
# ============================================================

print("\n" + "=" * 60)
print("STRONGEST EFFECT SIZES")
print("=" * 60)

effect_df = results_df.copy()

effect_df["Absolute_Cohens_D"] = (
    effect_df["Cohens_D"].abs()
)

effect_df = effect_df.sort_values(
    by="Absolute_Cohens_D",
    ascending=False
)

print(
    effect_df[
        [
            "Feature",
            "Cohens_D",
            "Absolute_Cohens_D"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("EFFECT SIZE INTERPRETATION")
print("=" * 60)

print("Cohen's d interpretation:")
print("0.00 - 0.19 : Very small effect")
print("0.20 - 0.49 : Small effect")
print("0.50 - 0.79 : Medium effect")
print("0.80+       : Large effect")


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS CONCLUSION")
print("=" * 60)

if len(significant_features) > 0:

    print(
        "Some numerical features show statistically "
        "significant differences between Stayed and "
        "Churned customers."
    )

    print(
        "These features may be useful for further "
        "feature engineering and model development."
    )

else:

    print(
        "No numerical features show statistically "
        "significant differences after multiple-"
        "comparison correction."
    )

    print(
        "This indicates that the current numerical "
        "features have weak statistical separation "
        "between Stayed and Churned customers."
    )

print(
    "\nStatistical testing uses Welch's independent "
    "t-test with FDR correction."
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "../08_Reports/statistical_feature_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    "\nStatistical feature analysis results saved successfully!"
)

print("File:", output_file)


# ============================================================
# CLOSE DATABASE CONNECTION
# ============================================================

engine.dispose()

print(
    "\nStatistical feature analysis completed successfully!"
)