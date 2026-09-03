import pandas as pd

# --------------------------------------------------
# Model Performance Results
# --------------------------------------------------

results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.5443,
        0.3818,
        0.6626
    ],
    "Precision": [
        0.1989,
        0.1950,
        0.1979
    ],
    "Recall": [
        0.4430,
        0.6962,
        0.2405
    ],
    "F1 Score": [
        0.2745,
        0.3047,
        0.2171
    ]
}

comparison = pd.DataFrame(results)

# --------------------------------------------------
# Display Comparison
# --------------------------------------------------

print("Model Comparison:")
print(comparison.to_string(index=False))

# --------------------------------------------------
# Find Best Models
# --------------------------------------------------

best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_recall = comparison.loc[
    comparison["Recall"].idxmax()
]

best_f1 = comparison.loc[
    comparison["F1 Score"].idxmax()
]

print("\nBest Accuracy:")
print(
    best_accuracy["Model"],
    "-",
    best_accuracy["Accuracy"]
)

print("\nBest Recall:")
print(
    best_recall["Model"],
    "-",
    best_recall["Recall"]
)

print("\nBest F1 Score:")
print(
    best_f1["Model"],
    "-",
    best_f1["F1 Score"]
)

# --------------------------------------------------
# Save Comparison
# --------------------------------------------------

comparison.to_csv(
    "../08_Reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

print("\nModel comparison completed successfully!")