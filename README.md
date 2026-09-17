# 📉 Telecom Customer Churn Prediction

A machine learning project that analyzes telecom customer data to understand churn drivers and predict whether a customer is likely to leave, with an interactive Streamlit dashboard for exploration and live predictions.

## 📁 Project Structure

```
.
├── TelecomCustomerChurn.csv          # Raw dataset
├── EDA.ipynb                         # Exploratory Data Analysis & data cleaning notebook
├── workspace.ipynb                   # Model training notebook (preprocessing, SMOTE, Logistic Regression)
├── app.py                            # Streamlit web app (dashboard + prediction)
├── logistic_regression_model.pkl     # Trained model (saved with joblib)
└── README.md
```

> **Note:** `app.py` expects a `cleaned_dataset.csv` file (the output of `EDA.ipynb`) in the same directory. Run the EDA notebook first, or point the app at your own cleaned file, before launching it.

## 📊 Dataset

The dataset (`TelecomCustomerChurn.csv`) contains customer-level records with 21 columns, including:

- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Account info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Target:** `Churn` (Yes/No)

## 🧹 Data Preparation (`EDA.ipynb`)

- Standardized column names and text values to lowercase
- Converted `totalcharges` to numeric and handled missing values (mode/mean imputation)
- Removed duplicate records
- Detected outliers using the IQR method
- Explored churn patterns via count plots, box plots, histograms, and pie charts (contract type, payment method, tech support, monthly charges, etc.)
- Exported the cleaned data to `cleaned_dataset.csv`

## 🤖 Model Training (`workspace.ipynb`)

1. Dropped the `customerid` identifier column
2. Split data into features (`X`) and target (`y = churn`)
3. Train/test split (80/20, `random_state=42`)
4. Label-encoded categorical features
5. Scaled numeric features with `StandardScaler`
6. Balanced the training set with **SMOTE** (the churn classes are imbalanced)
7. Trained a **Logistic Regression** classifier on the resampled data
8. Evaluated with accuracy score on the held-out test set
9. Saved the trained model with `joblib` as `logistic_regression_model.pkl`

## 🖥️ Streamlit App (`app.py`)

The app has two pages, selectable from the sidebar:

- **📊 Analysis** — an interactive dashboard showing total customers, churn rate, and churn breakdowns by payment method, contract type, tech support, streaming services, and monthly charges.
- **📉 Prediction** — a form where you enter a customer's attributes (tenure, contract, charges, services, etc.) and get a live churn prediction with probability, using the saved Logistic Regression model.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies
```bash
pip install streamlit pandas scikit-learn plotly joblib imbalanced-learn matplotlib seaborn numpy
```

### 3. Generate the cleaned dataset (if not already present)
Run all cells in `EDA.ipynb` to produce `cleaned_dataset.csv` in the project folder.

### 4. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn, plotly
- **Machine Learning** — scikit-learn (Logistic Regression, StandardScaler, LabelEncoder), imbalanced-learn (SMOTE)
- **App / Deployment** — Streamlit
- **Model persistence** — joblib

## 📌 Key Insights from EDA

- Customers without a partner or dependents show higher churn rates.
- Month-to-month contracts churn far more often than one/two-year contracts.
- Higher monthly charges are associated with a higher likelihood of churn.
- Lack of tech support and online security correlates with increased churn.

## 📄 License

Add your preferred license here (e.g., MIT).
