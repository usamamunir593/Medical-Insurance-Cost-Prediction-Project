# Medical Insurance Cost Prediction

A beginner-friendly machine learning project that estimates an individual's medical insurance charges from demographic and lifestyle features using a **Multiple Linear Regression** model.

> **Important:** This is an educational estimate, not a medical opinion, an insurer quote, or a guarantee of actual costs. Predictions reflect patterns in the historical dataset and may be inaccurate for individuals.

## Problem statement
Medical insurance charges can vary with age, BMI, smoking status, number of children, sex, and geographic region. This project demonstrates an end-to-end regression workflow that uses these fields to estimate insurance charges.

## Dataset
The project uses the commonly shared Medical Cost Personal Dataset (`insurance.csv`) with these fields:

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Female or male category as recorded in the data |
| `bmi` | Body mass index |
| `children` | Number of children/dependents represented in the dataset |
| `smoker` | Smoking status (`yes` / `no`) |
| `region` | US region category |
| `charges` | Historical medical insurance charges; prediction target |

The cleaned project dataset contains **1,337 rows** after removing duplicate records from the original 2,772-row file. No missing values were found in the cleaned dataset. The source data describes a limited population and is not guaranteed to represent today's insurance market or people outside its original context.

## Workflow
1. **Data preparation:** removed duplicate rows; checked missing values and data types.
2. **Exploratory data analysis:** examined distributions and relationships between features and insurance charges. Smoking status, age, and BMI are important patterns to inspect, but associations in this dataset do not establish causation.
3. **Preprocessing:** numeric and categorical columns are handled within a scikit-learn pipeline. Categorical values are encoded; numeric inputs are passed through the preprocessing pipeline.
4. **Model selection:** Multiple Linear Regression was selected as an interpretable baseline regression model.
5. **Training and test split:** 80% training / 20% test data with `random_state=42` (1,069 training rows and 268 test rows).
6. **Evaluation:** the model was evaluated on the held-out test set using MAE, MSE, RMSE, and R².
7. **Prediction app:** Streamlit form accepts the input features and returns an estimated charge.

## Model evaluation
Reported test-set metrics from Phase 6:

| Metric | Result | Interpretation |
|---|---:|---|
| Mean Absolute Error (MAE) | $4,177.05 | Average absolute prediction error |
| Mean Squared Error (MSE) | 35,478,020.68 | Average squared error, in squared currency units |
| Root Mean Squared Error (RMSE) | $5,956.34 | Error measure that penalizes larger misses more strongly |
| R² | 0.8069 | About 80.69% of target variation was explained on this test split |

These are results for this specific split and dataset, not a promise of performance on new populations.

## Example predictions
Two reproducible example inputs from the app's test cases:

| Case | Age | BMI | Children | Sex | Smoker | Region | Predicted charges |
|---|---:|---:|---:|---|---|---|---:|
| 1 | 25 | 22.5 | 0 | Female | No | Southwest | $1,624.26 |
| 2 | 45 | 32.0 | 2 | Male | Yes | Southeast | $33,478.60 |

These are model estimates, not real quotes. A different input combination may produce a different result.

## Run locally
Requires Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Open it in your browser, fill in the form, and click the prediction button.

## Deploy with Streamlit Community Cloud
1. Create a GitHub repository and upload the contents of this folder to its root.
2. Ensure `app.py`, `requirements.txt`, and `medical_insurance_linear_regression.joblib` are committed.
3. Sign in to Streamlit Community Cloud and connect your GitHub account.
4. Choose the repository and branch, set the main file path to `app.py`, then deploy.
5. Open the public app URL and test at least two input combinations.
6. Replace the placeholder screenshot assets in `screenshots/` with genuine captures of your own GitHub repository, README, deployed app, and a successful prediction. Do not label mock/generated panels as real browser screenshots.

## Repository structure
```text
medical-insurance-cost-prediction/
├── app.py
├── requirements.txt
├── medical_insurance_linear_regression.joblib
├── data_medical_insurance_cleaned.csv
├── test_case_predictions.csv
├── README.md
├── docs/
│   └── PROJECT_DOCUMENTATION.md
└── screenshots/
    ├── proof_01_application_interface.png
    ├── proof_02_test_case_non_smoker.png
    └── proof_03_test_case_smoker.png
```

## Limitations and responsible use
- This is a learning project, not a production-grade insurance underwriting system.
- The dataset is limited and historical; it may contain sampling and geographic biases.
- Linear Regression assumes an additive linear relationship and may not capture all interactions or nonlinear effects.
- An R² score does not mean that every prediction is 80.69% accurate.
- Errors of several thousand dollars can be meaningful. Review MAE and RMSE before interpreting a prediction.
- Sensitive or proxy attributes may create fairness concerns. Do not use this project to make real insurance eligibility, pricing, or medical decisions.
- Currency amounts are shown as the dataset's charges; the dataset is US-focused and values should not be interpreted as Pakistani rupees.

## Screenshots
See the `screenshots/` directory for illustrative Phase 7 application and test-case panels. These panels were generated for documentation and are **not genuine screenshots of a live deployed website**. Add real browser screenshots after GitHub publication and deployment.

## Reproducibility notes
The training/test split uses `random_state=42`. The saved `.joblib` pipeline was created for the Streamlit application. Only load serialized model files from sources you trust.
