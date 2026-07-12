"""
Reload-and-predict demonstration for Part 3.

Loads the serialized best pipeline (best_model.pkl) and runs .predict() /
.predict_proba() on two hand-crafted employee rows. Run this after
advanced_modeling.py has produced best_model.pkl.

Run with: python3 reload_predict.py
"""
import joblib
import pandas as pd

# Column order must match the training feature matrix exactly.
FEATURE_NAMES = [
    "age", "education_level", "remote_work", "years_experience",
    "performance_score", "satisfaction_score", "bonus_pct",
    "department_Finance", "department_HR", "department_Marketing",
    "department_Sales", "department_Support",
    "region_North", "region_South", "region_West",
]

loaded_pipeline = joblib.load("best_model.pkl")

hand_crafted_rows = pd.DataFrame([
    {  # a senior, highly-educated Engineering employee -> expect "high earner"
        "age": 45.0, "education_level": 3, "remote_work": 0,
        "years_experience": 20.0, "performance_score": 8.0,
        "satisfaction_score": 75.0, "bonus_pct": 5.0,
        "department_Finance": 0, "department_HR": 0, "department_Marketing": 0,
        "department_Sales": 0, "department_Support": 0,
        "region_North": 1, "region_South": 0, "region_West": 0,
    },
    {  # a junior Support employee -> expect "not high earner"
        "age": 24.0, "education_level": 0, "remote_work": 1,
        "years_experience": 1.0, "performance_score": 6.5,
        "satisfaction_score": 60.0, "bonus_pct": 1.0,
        "department_Finance": 0, "department_HR": 0, "department_Marketing": 0,
        "department_Sales": 0, "department_Support": 1,
        "region_North": 0, "region_South": 1, "region_West": 0,
    },
])[FEATURE_NAMES]

predictions = loaded_pipeline.predict(hand_crafted_rows)
probabilities = loaded_pipeline.predict_proba(hand_crafted_rows)[:, 1]

print("Predictions (1 = high earner, 0 = not):", predictions)
print("Probabilities P(high_earner):", probabilities)
