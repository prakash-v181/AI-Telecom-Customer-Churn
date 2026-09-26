# AI Telecom Customer Churn

Hi, I am Prakash. This is my end-to-end telecom customer churn project.
I built it to understand customer data, find patterns related to churn,
compare machine learning models, and present the results through
dashboards.

The idea was to make a project that is not only about training a model.
I also wanted to understand what the data is telling us and how a
business team could use those findings for discussion.

## Project Links

-   **GitHub:**
    https://github.com/prakash-v181/AI-Telecom-Customer-Churn
-   **Live Streamlit App:**
    https://ai-telecom-customer-churn-g7tdgaazpqlrer3quvf9xb.streamlit.app/

## About the Project

Customer churn means a customer stops using a company's service. For a
telecom company, understanding churn can help the team review customer
experience, service quality, pricing, support, and retention efforts.

This project explores customer information and answers questions such
as:

-   How many customers stayed or churned?
-   What is the overall churn rate?
-   How does churn differ by telecom partner, age group, state, and
    city?
-   What can we learn from customer usage?
-   How do the machine learning models perform?
-   Can an AI assistant explain the findings in simple language?

## Dataset

The project uses an India telecom customer dataset. The cleaned dataset
contains **2,026 customer records**. The target column is `churn`, where
`0` means the customer stayed and `1` means the customer churned.

  Metric                    Value
  ----------------------- -------
  Total customers           2,026
  Customers who stayed      1,633
  Customers who churned       393
  Overall churn rate        19.4%

The dataset includes details such as telecom partner, gender, age,
state, city, registration date, number of dependents, estimated salary,
calls made, SMS sent, data used, and churn status.

Check the `01_Data` folder for the data files available in the
repository. Make sure the required dataset is in the expected location
before running scripts.

## Work Completed

### Data analysis

I checked the dataset structure, data types, missing values, duplicate
records, and churn distribution. I also explored churn across telecom
partners, age groups, gender, locations, and usage segments.

### Business insights

Some patterns found in this dataset:

-   Overall churn rate is about **19.4%**.
-   Vodafone had the highest churn rate among the telecom partners
    checked, at about **21.28%**.
-   The 36--45 age group had the highest churn rate among the age groups
    checked, at about **21.61%**.
-   Bihar had the highest state churn rate in the analysis, at about
    **26.14%**.
-   Kolkata had the highest city churn rate in the analysis, at about
    **22.39%**.

These are observations from this dataset. They do not prove that a
partner, age group, or location causes churn.

### Machine learning

I compared classification approaches including Decision Tree, Random
Forest, and XGBoost variations. I reviewed accuracy, precision, recall,
F1-score, ROC-AUC, and PR-AUC.

In the recorded comparison, the improved XGBoost model reached
approximately **0.5299 ROC-AUC** and **0.1998 PR-AUC**.

The results show that the available features have limited ability to
separate customers who churn from customers who stay. I do not treat the
model as a reliable individual churn predictor. More useful
information---such as network complaints, service quality, recharge
history, plan changes, support interactions, and customer tenure---could
help improve future analysis.

### Streamlit app and AI Analyst

The Streamlit app brings the analysis together with sections for:

-   Overview
-   Business Insights
-   Model Comparison
-   Customer Risk
-   AI Analyst
-   Project Information

The AI Analyst uses Gemini to answer questions about the project and its
results. It is there to help explain the analysis, not to replace
checking the data or model evaluation.

### Power BI dashboard

I also worked on a Power BI dashboard with KPI cards, filters, and
visuals for churn, telecom partners, age groups, states, service usage,
monthly churn, customer risk scores, and model performance.

The Power BI report file may be maintained separately from this
repository, depending on which files are currently uploaded.

## Tools and Technologies

-   Python
-   Pandas and NumPy
-   Scikit-learn
-   XGBoost
-   Streamlit
-   Plotly
-   Gemini API
-   Power BI
-   Git and GitHub

## Project Structure

``` text
AI-Telecom-Customer-Churn/
├── 01_Data/
├── 04_Python/
├── 05_Machine_Learning/
│   └── 43_ai_churn_final_app.py
├── 08_Reports/
├── requirements.txt
└── .gitignore
```

The repository may include additional files as the project continues to
develop.

## Run the App Locally

### 1. Clone the repository

``` bash
git clone https://github.com/prakash-v181/AI-Telecom-Customer-Churn.git
cd AI-Telecom-Customer-Churn
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install libraries

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.streamlit` folder in the project root. Inside it, create
`secrets.toml` and add your own API key:

``` toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3.6-flash"
```

Keep your API key private. Do not commit `secrets.toml` or the key to
GitHub.

### 5. Run Streamlit

From the project root:

``` bash
streamlit run 05_Machine_Learning/43_ai_churn_final_app.py
```

Open the local URL shown in the terminal.

## Model Limitations

A machine learning result is useful only when it has been evaluated
properly. The model results in this project show weak separation between
churn and non-churn customers.

For that reason, a risk score should be treated as an analytical signal
for exploration. It is not a calibrated probability and does not
guarantee that a customer will leave.

For real business use, I would collect better churn-related information,
validate the model on new data, monitor its performance, and review the
results with the business team before using predictions in customer
decisions.

## What I Learned

This project gave me practice with:

-   Cleaning and exploring customer data
-   Understanding churn as a business problem
-   Comparing machine learning models with different metrics
-   Building interactive dashboards
-   Connecting an AI API to a Streamlit app
-   Organising files and sharing work through GitHub
-   Explaining model limitations instead of focusing only on accuracy

I see this as a learning project that can be improved with better data
and more testing.

## Future Improvements

-   Add customer behaviour and service quality features
-   Improve model validation and test more approaches
-   Explain customer risk scores more clearly
-   Improve and publish the Power BI report appropriately
-   Add tests and document the data preparation process
-   Monitor model performance when new data is available

## About Me

I am Prakash, and this project is part of my learning journey in Python,
machine learning, data analytics, and business intelligence.

I built it step by step to understand how data analysis and machine
learning can be applied to a business problem. Feedback and suggestions
are welcome.

**Thanks for checking out my project!**
