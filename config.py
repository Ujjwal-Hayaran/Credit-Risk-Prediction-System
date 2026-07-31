# config.py
# Single source of truth for categorical encodings.
# Both train_model.py and app.py import from here — if a new category
# is ever added to the training data, update it once, in this file only.

EDU_MAP = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
MARITAL_MAP = {"Single": 2, "Married": 1, "Divorced": 0}
RESIDENCE_MAP = {"Owned": 2, "Rented": 1, "Mortgaged": 0}
PURPOSE_MAP = {"Personal": 3, "Car": 1, "Home": 2, "Business": 0, "Education": 4}

FEATURE_COLUMNS = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "education_level",
    "marital_status",
    "num_dependents",
    "existing_loans_count",
    "residence_type",
    "loan_purpose",
]
