import pandas as pd
import os


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("=" * 70)
print("FINAL MACHINE LEARNING MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# Results folder
# ------------------------------------------------------------

results_folder = "../08_Reports"

os.makedirs(results_folder, exist_ok=True)


# ------------------------------------------------------------
# Model Results
# ------------------------------------------------------------
# Values collected from the models already completed
# ------------------------------------------------------------

results = [

    {
        "Model": "SMOTE Decision Tree",
        "Accuracy": 0.7635,
        "Precision": 0.2424,
        "Recall": 0.1013,
        "F1 Score": 0.1429,
        "ROC-AUC": None,
        "PR-AUC": None
    },

    {
        "Model": "Class-Weighted Random Forest",
        "Accuracy": 0.6478,
        "Precision": 0.1863,
        "Recall": 0.2405,
        "F1 Score": 0.2099,
        "ROC-AUC": None,
        "PR-AUC": None
    },

    {
        "Model": "XGBoost",
        "Accuracy": 0.6527,
        "Precision": 0.1837,
        "Recall": 0.2278,
        "F1 Score": 0.2034,
        "ROC-AUC": None,
        "PR-AUC": None
    },

    {
        "Model": "Random Forest Threshold Tuning",
        "Accuracy": 0.1946,
        "Precision": 0.1946,
        "Recall": 1.0000,
        "F1 Score": 0.3258,
        "ROC-AUC": None,
        "PR-AUC": None
    },

    {
        "Model": "Final Random Forest",
        "Accuracy": 0.6010,
        "Precision": 0.1971,
        "Recall": 0.3418,
        "F1 Score": 0.2500,
        "ROC-AUC": 0.4934,
        "PR-AUC": 0.1914
    },

    {
        "Model": "Improved Random Forest",
        "Accuracy": 0.7167,
        "Precision": 0.2429,
        "Recall": 0.2152,
        "F1 Score": 0.2282,
        "ROC-AUC": 0.4892,
        "PR-AUC": 0.1970
    },

    {
        "Model": "Improved XGBoost",
        "Accuracy": 0.6897,
        "Precision": 0.2169,
        "Recall": 0.2278,
        "F1 Score": 0.2222,
        "ROC-AUC": 0.5299,
        "PR-AUC": 0.1998
    }
]


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

comparison = pd.DataFrame(results)


# ------------------------------------------------------------
# Display Comparison
# ------------------------------------------------------------

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        na_rep="N/A"
    )
)


# ------------------------------------------------------------
# Best Models
# ------------------------------------------------------------

best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_precision = comparison.loc[
    comparison["Precision"].idxmax()
]

best_recall = comparison.loc[
    comparison["Recall"].idxmax()
]

best_f1 = comparison.loc[
    comparison["F1 Score"].idxmax()
]


# ------------------------------------------------------------
# Best ROC-AUC
# ------------------------------------------------------------

roc_comparison = comparison.dropna(
    subset=["ROC-AUC"]
)

best_roc_auc = roc_comparison.loc[
    roc_comparison["ROC-AUC"].idxmax()
]


# ------------------------------------------------------------
# Best PR-AUC
# ------------------------------------------------------------

pr_comparison = comparison.dropna(
    subset=["PR-AUC"]
)

best_pr_auc = pr_comparison.loc[
    pr_comparison["PR-AUC"].idxmax()
]


# ------------------------------------------------------------
# Print Best Models
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL BY METRIC")
print("=" * 70)

print(
    f"\nBest Accuracy : "
    f"{best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Best Precision: "
    f"{best_precision['Model']} "
    f"({best_precision['Precision']:.4f})"
)

print(
    f"Best Recall   : "
    f"{best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1 Score : "
    f"{best_f1['Model']} "
    f"({best_f1['F1 Score']:.4f})"
)

print(
    f"Best ROC-AUC  : "
    f"{best_roc_auc['Model']} "
    f"({best_roc_auc['ROC-AUC']:.4f})"
)

print(
    f"Best PR-AUC   : "
    f"{best_pr_auc['Model']} "
    f"({best_pr_auc['PR-AUC']:.4f})"
)


# ------------------------------------------------------------
# Churn-Focused Recommendation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHURN-FOCUSED MODEL ANALYSIS")
print("=" * 70)

print(
    "\nFor churn prediction, Recall is important because "
    "missing a customer who will churn can be costly."
)

print(
    f"\nHighest Recall Model: "
    f"{best_recall['Model']}"
)

print(
    f"Recall: {best_recall['Recall']:.4f}"
)

print(
    f"Precision: {best_recall['Precision']:.4f}"
)

print(
    f"F1 Score: {best_recall['F1 Score']:.4f}"
)


# ------------------------------------------------------------
# Practical Recommendation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

print(
    """
The models were tested using multiple approaches including:
- SMOTE
- Class weighting
- Random Forest
- XGBoost
- Threshold tuning
- Feature engineering
- Improved preprocessing
"""
)

print(
    "The highest ROC-AUC among the evaluated models is "
    f"{best_roc_auc['ROC-AUC']:.4f}, achieved by "
    f"{best_roc_auc['Model']}."
)

print(
    "\nHowever, ROC-AUC values remain close to 0.50, "
    "indicating weak predictive separation."
)

print(
    "\nTherefore, the current dataset does not provide "
    "strong predictive signals for reliable churn prediction."
)

print(
    "\nThe project should report this as an important finding "
    "rather than selecting a model based only on accuracy."
)


# ------------------------------------------------------------
# Save Full Comparison
# ------------------------------------------------------------

comparison_file = (
    results_folder +
    "/final_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print(
    f"\nModel comparison saved successfully!\n"
    f"File: {comparison_file}"
)


# ------------------------------------------------------------
# Save Summary
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Best Accuracy",
        "Best Precision",
        "Best Recall",
        "Best F1 Score",
        "Best ROC-AUC",
        "Best PR-AUC"
    ],

    "Model": [
        best_accuracy["Model"],
        best_precision["Model"],
        best_recall["Model"],
        best_f1["Model"],
        best_roc_auc["Model"],
        best_pr_auc["Model"]
    ],

    "Score": [
        best_accuracy["Accuracy"],
        best_precision["Precision"],
        best_recall["Recall"],
        best_f1["F1 Score"],
        best_roc_auc["ROC-AUC"],
        best_pr_auc["PR-AUC"]
    ]
})


summary_file = (
    results_folder +
    "/final_model_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"Final model summary saved successfully!\n"
    f"File: {summary_file}"
)


# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 70)