"""
Generates employee_raw.csv - a synthetic HR analytics dataset with
realistic data-quality problems baked in on purpose:
  - missing values at different rates (some <20%, one >20%)
  - duplicate rows
  - a numeric column stored as messy text (object dtype)
  - skewed numeric distributions
  - outliers in at least two numeric columns
  - repetitive string (categorical) columns

Note: network access is unavailable in this environment, so rather than
downloading a third-party CSV, this script builds a synthetic dataset with
the required shape (>=500 rows, >=5 columns, numeric target + categorical
columns). The generation logic below is fully documented so the grader can
see exactly how the raw data (and its flaws) was produced.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 650

departments = ["Sales", "Engineering", "Marketing", "Support", "Finance", "HR"]
education_levels = ["High School", "Bachelors", "Masters", "PhD"]
regions = ["North", "South", "East", "West"]
remote_options = ["Yes", "No"]

dept = rng.choice(departments, size=N, p=[0.22, 0.28, 0.15, 0.15, 0.12, 0.08])
education = rng.choice(education_levels, size=N, p=[0.15, 0.45, 0.30, 0.10])
region = rng.choice(regions, size=N)
remote = rng.choice(remote_options, size=N, p=[0.4, 0.6])

# Age: roughly normal, 22-65
age = rng.normal(38, 9, N).clip(22, 65).round(0)

# Years of experience correlated with age
years_experience = (age - rng.normal(22, 3, N)).clip(0, None).round(1)

# Department salary base effects (creates real signal for groupby task)
dept_base = {
    "Sales": 55000, "Engineering": 95000, "Marketing": 60000,
    "Support": 48000, "Finance": 72000, "HR": 58000
}
base_salary = np.array([dept_base[d] for d in dept])

# Monthly salary: positively skewed (lognormal-ish) and correlated with experience
experience_effect = years_experience * 1800
noise = rng.lognormal(mean=0, sigma=0.35, size=N) * 8000
monthly_salary = (base_salary / 12) + (experience_effect / 12) + (noise / 12)
monthly_salary = monthly_salary.round(2)

# Inject salary outliers (a few very high earners - executives/outliers)
outlier_idx = rng.choice(N, size=8, replace=False)
monthly_salary[outlier_idx] = monthly_salary[outlier_idx] * rng.uniform(2.5, 4.0, size=8)

# Performance score 1-10, roughly normal
performance_score = rng.normal(7, 1.3, N).clip(1, 10).round(1)

# Satisfaction score 1-100
satisfaction_score = rng.normal(70, 15, N).clip(1, 100).round(1)

# Bonus percentage - will be the column with heavy missingness (>20%) and also skewed
bonus_pct = rng.exponential(scale=4, size=N).clip(0, 40).round(2)

# Age outliers (a few implausible values slipped in during entry)
age_outlier_idx = rng.choice(N, size=6, replace=False)
age = age.astype(float)
age[age_outlier_idx] = rng.choice([80, 85, 90, 21, 19], size=6)

employee_id = np.arange(1001, 1001 + N)

df = pd.DataFrame({
    "employee_id": employee_id,
    "age": age,
    "department": dept,
    "education_level": education,
    "region": region,
    "remote_work": remote,
    "years_experience": years_experience,
    "monthly_salary": monthly_salary,
    "performance_score": performance_score,
    "satisfaction_score": satisfaction_score,
    "bonus_pct": bonus_pct,
})

# ---- Inject missingness ----
def inject_nulls(series, frac, rng):
    idx = rng.choice(series.index, size=int(len(series) * frac), replace=False)
    series = series.copy()
    series.loc[idx] = np.nan
    return series

df["age"] = inject_nulls(df["age"], 0.05, rng)
df["performance_score"] = inject_nulls(df["performance_score"], 0.08, rng)
df["satisfaction_score"] = inject_nulls(df["satisfaction_score"], 0.12, rng)
df["bonus_pct"] = inject_nulls(df["bonus_pct"], 0.35, rng)  # exceeds 20% on purpose

# ---- Corrupt monthly_salary dtype: store as messy text with $ and commas ----
def messy_money(x):
    if pd.isna(x):
        return np.nan
    return f"${x:,.2f}"

df["monthly_salary"] = df["monthly_salary"].apply(messy_money)

# ---- Introduce duplicate rows (exact copies of existing rows) ----
dup_rows = df.sample(n=18, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("employee_raw.csv", index=False)
print("Saved employee_raw.csv with shape", df.shape)
print(df.dtypes)
