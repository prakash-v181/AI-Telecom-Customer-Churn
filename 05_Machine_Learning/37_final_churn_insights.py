import pandas as pd
import os


# ============================================================
# FINAL CHURN BUSINESS INSIGHTS
# ============================================================

print("=" * 70)
print("FINAL CUSTOMER CHURN BUSINESS INSIGHTS")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

reports_folder = "../08_Reports"

os.makedirs(reports_folder, exist_ok=True)


# ------------------------------------------------------------
# Load Customer Data
# ------------------------------------------------------------

file_path = "../08_Reports/engineered_customer_data.csv"

df = pd.read_csv(file_path)

print("\nData loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 1. OVERALL CHURN
# ============================================================

total_customers = len(df)

churned_customers = int(
    df["churn"].sum()
)

stayed_customers = (
    total_customers - churned_customers
)

overall_churn_rate = (
    churned_customers / total_customers * 100
)

print("\n" + "=" * 70)
print("1. OVERALL CHURN")
print("=" * 70)

print("Total Customers :", total_customers)
print("Stayed Customers:", stayed_customers)
print("Churned Customers:", churned_customers)
print("Overall Churn Rate:", round(overall_churn_rate, 2), "%")


# ============================================================
# 2. CATEGORY ANALYSIS FUNCTION
# ============================================================

def category_analysis(column):

    result = (
        df.groupby(column)["churn"]
        .agg(
            customers="count",
            churned="sum"
        )
    )

    result["churn_rate"] = (
        result["churned"] /
        result["customers"] * 100
    )

    result = result.sort_values(
        "churn_rate",
        ascending=False
    )

    return result


# ============================================================
# 3. TELECOM PARTNER
# ============================================================

partner_result = category_analysis(
    "telecom_partner"
)

print("\n" + "=" * 70)
print("2. CHURN BY TELECOM PARTNER")
print("=" * 70)

print(
    partner_result.round(2)
)

highest_partner = partner_result.index[0]
highest_partner_rate = partner_result.iloc[0]["churn_rate"]


# ============================================================
# 4. GENDER
# ============================================================

gender_result = category_analysis(
    "gender"
)

print("\n" + "=" * 70)
print("3. CHURN BY GENDER")
print("=" * 70)

print(
    gender_result.round(2)
)

highest_gender = gender_result.index[0]
highest_gender_rate = gender_result.iloc[0]["churn_rate"]


# ============================================================
# 5. AGE GROUP
# ============================================================

age_result = category_analysis(
    "age_group"
)

print("\n" + "=" * 70)
print("4. CHURN BY AGE GROUP")
print("=" * 70)

print(
    age_result.round(2)
)

highest_age_group = age_result.index[0]
highest_age_rate = age_result.iloc[0]["churn_rate"]


# ============================================================
# 6. USAGE SEGMENT
# ============================================================

usage_result = category_analysis(
    "usage_segment"
)

print("\n" + "=" * 70)
print("5. CHURN BY USAGE SEGMENT")
print("=" * 70)

print(
    usage_result.round(2)
)

highest_usage = usage_result.index[0]
highest_usage_rate = usage_result.iloc[0]["churn_rate"]


# ============================================================
# 7. STATE
# ============================================================

state_result = category_analysis(
    "state"
)

print("\n" + "=" * 70)
print("6. CHURN BY STATE")
print("=" * 70)

print(
    state_result.round(2)
)

highest_state = state_result.index[0]
highest_state_rate = state_result.iloc[0]["churn_rate"]


# ============================================================
# 8. CITY
# ============================================================

city_result = category_analysis(
    "city"
)

print("\n" + "=" * 70)
print("7. CHURN BY CITY")
print("=" * 70)

print(
    city_result.round(2)
)

highest_city = city_result.index[0]
highest_city_rate = city_result.iloc[0]["churn_rate"]


# ============================================================
# 9. NUMERICAL FEATURES
# ============================================================

numerical_columns = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used",
    "total_communication"
]

numerical_summary = (
    df.groupby("churn")[numerical_columns]
    .mean()
)

print("\n" + "=" * 70)
print("8. NUMERICAL FEATURE COMPARISON")
print("=" * 70)

print(
    numerical_summary.round(2)
)


# ============================================================
# 10. ENGINEERED FEATURES
# ============================================================

engineered_columns = [
    "calls_per_data",
    "sms_per_call",
    "data_per_call",
    "avg_communication",
    "communication_intensity",
    "usage_per_age",
    "salary_per_age",
    "dependents_per_age",
    "calls_sms_ratio"
]

engineered_summary = (
    df.groupby("churn")[engineered_columns]
    .mean()
)

print("\n" + "=" * 70)
print("9. ENGINEERED FEATURE COMPARISON")
print("=" * 70)

print(
    engineered_summary.round(4)
)


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

correlation_columns = (
    numerical_columns +
    engineered_columns +
    ["churn"]
)

correlation = (
    df[correlation_columns]
    .corr()["churn"]
    .drop("churn")
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\n" + "=" * 70)
print("10. CORRELATION WITH CHURN")
print("=" * 70)

print(
    correlation.round(4)
)

strongest_feature = correlation.index[0]
strongest_correlation = correlation.iloc[0]


# ============================================================
# 12. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("11. KEY BUSINESS INSIGHTS")
print("=" * 70)

print(
    f"""
1. Overall churn rate is {overall_churn_rate:.2f}%,
   meaning approximately {churned_customers} out of
   {total_customers} customers have churned.

2. {highest_partner} has the highest telecom-partner
   churn rate at {highest_partner_rate:.2f}%.

3. {highest_gender} has the highest gender-based
   churn rate at {highest_gender_rate:.2f}%.

4. {highest_age_group} has the highest age-group
   churn rate at {highest_age_rate:.2f}%.

5. {highest_usage} has the highest usage-segment
   churn rate at {highest_usage_rate:.2f}%.

6. {highest_state} has the highest state-level
   churn rate at {highest_state_rate:.2f}%.

7. {highest_city} has the highest city-level
   churn rate at {highest_city_rate:.2f}%.

8. Numerical feature differences between Stayed and
   Churned customers are generally small.

9. Engineered features also show weak direct
   relationships with churn.

10. Statistical analysis found no statistically
    significant numerical or categorical features
    after multiple-comparison correction.

11. Machine-learning models also showed weak
    predictive separation.

12. Therefore, the current dataset should not be used
    to claim highly accurate individual churn prediction.
"""
)


# ============================================================
# 13. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("12. BUSINESS RECOMMENDATIONS")
print("=" * 70)

recommendations = [
    "Focus retention analysis on higher-churn geographic segments.",
    
    "Investigate customer experience and service quality "
    "differences across telecom partners.",
    
    "Review customers in higher-churn age groups for "
    "possible retention opportunities.",
    
    "Monitor customers in higher-churn cities and states "
    "for service or competitive issues.",
    
    "Collect additional behavioral information such as "
    "complaints, network quality, recharge frequency, "
    "plan changes, tenure, and customer support interactions.",
    
    "Improve the dataset with stronger churn-related "
    "features before deploying a predictive churn model.",
    
    "Use model predictions as analytical support rather "
    "than as a fully reliable automated churn decision."
]

for number, recommendation in enumerate(
    recommendations,
    start=1
):
    print(
        f"{number}. {recommendation}"
    )


# ============================================================
# 14. FINAL ML CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("13. FINAL MACHINE LEARNING CONCLUSION")
print("=" * 70)

print(
    """
The project evaluated multiple machine-learning approaches,
including Decision Tree, Random Forest, SMOTE, class weighting,
XGBoost, threshold tuning, feature engineering, and improved
preprocessing.

The Improved XGBoost model achieved the highest ROC-AUC
among the evaluated models, with ROC-AUC of approximately 0.53.

However, this value is close to random classification.

Therefore, the major project finding is that the available
features provide weak predictive information about customer
churn.

The next improvement should focus on collecting or creating
stronger business-related features rather than simply adding
more machine-learning algorithms.
"""
)


# ============================================================
# 15. FINAL INSIGHTS DATASET
# ============================================================

insights = pd.DataFrame({

    "Insight": [
        "Total Customers",
        "Stayed Customers",
        "Churned Customers",
        "Overall Churn Rate",
        "Highest Churn Telecom Partner",
        "Highest Partner Churn Rate",
        "Highest Churn Gender",
        "Highest Gender Churn Rate",
        "Highest Churn Age Group",
        "Highest Age Group Churn Rate",
        "Highest Churn Usage Segment",
        "Highest Usage Segment Churn Rate",
        "Highest Churn State",
        "Highest State Churn Rate",
        "Highest Churn City",
        "Highest City Churn Rate",
        "Strongest Numerical/Engineered Correlation",
        "Strongest Correlation Value"
    ],

    "Value": [
        total_customers,
        stayed_customers,
        churned_customers,
        round(overall_churn_rate, 2),
        highest_partner,
        round(highest_partner_rate, 2),
        highest_gender,
        round(highest_gender_rate, 2),
        highest_age_group,
        round(highest_age_rate, 2),
        highest_usage,
        round(highest_usage_rate, 2),
        highest_state,
        round(highest_state_rate, 2),
        highest_city,
        round(highest_city_rate, 2),
        strongest_feature,
        round(strongest_correlation, 4)
    ]
})


# ============================================================
# 16. SAVE RESULTS
# ============================================================

insights_file = (
    reports_folder +
    "/final_churn_business_insights.csv"
)

insights.to_csv(
    insights_file,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    f"Business insights saved successfully!\n"
    f"File: {insights_file}"
)


# Save category results

partner_result.to_csv(
    reports_folder +
    "/insights_telecom_partner.csv"
)

gender_result.to_csv(
    reports_folder +
    "/insights_gender.csv"
)

age_result.to_csv(
    reports_folder +
    "/insights_age_group.csv"
)

usage_result.to_csv(
    reports_folder +
    "/insights_usage_segment.csv"
)

state_result.to_csv(
    reports_folder +
    "/insights_state.csv"
)

city_result.to_csv(
    reports_folder +
    "/insights_city.csv"
)


print(
    "Category insight files saved successfully!"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHURN BUSINESS INSIGHTS COMPLETED SUCCESSFULLY!")
print("=" * 70)