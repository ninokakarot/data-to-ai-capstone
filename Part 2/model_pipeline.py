"""
Part 2 - Supervised Machine Learning: Regression + Classification
Input: cleaned_data.csv (produced by Part 1)

Run with: python3 model_pipeline.py
Outputs: plots/roc_curve.png, console_output.txt (captured via redirection)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
)

import os
os.makedirs("plots", exist_ok=True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("LOAD DATA")
print("=" * 80)
df = pd.read_csv("cleaned_data.csv")
print("Shape:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------------------------
# Task 1: Define targets
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 1: Define targets")
print("=" * 80)

# Regression target: monthly_salary (continuous)
y_reg = df["monthly_salary"].copy()

# Classification target: "high_earner" flag = monthly_salary in the top 30%
# (>= 70th percentile). A pure median split (50/50) would be almost exactly
# balanced and would not exercise the imbalance-handling requirement below,
# so the cut is placed at the 70th percentile instead - this still follows
# the task's "binarize y_reg" approach, just at a business-relevant cutoff
# ("is this employee a high earner?") that also yields a genuinely
# imbalanced, learnable target (salary is driven by age/experience/
# department, all of which remain available as features).
SALARY_QUANTILE = 0.70
salary_threshold = y_reg.quantile(SALARY_QUANTILE)
y_clf = (y_reg >= salary_threshold).astype(int)
print(f"y_reg = monthly_salary (continuous)")
print(f"y_clf = 1 if monthly_salary >= {salary_threshold:.2f} "
      f"(the {SALARY_QUANTILE:.0%} percentile) else 0 ('high_earner')")
print("\ny_clf value counts:\n", y_clf.value_counts())
print("\ny_clf class balance:\n", y_clf.value_counts(normalize=True))

# Feature matrices - note each target's own source column is excluded from
# its own X to prevent trivial leakage (regression must not see
# monthly_salary as a feature of itself; classification must not see
# monthly_salary either, since y_clf is directly derived from it).
drop_cols_common = ["employee_id"]

X_reg_raw = df.drop(columns=drop_cols_common + ["monthly_salary"])
X_clf_raw = df.drop(columns=drop_cols_common + ["monthly_salary"])

print("\nX_reg columns:", X_reg_raw.columns.tolist())
print("X_clf columns:", X_clf_raw.columns.tolist())

# ---------------------------------------------------------------------------
# Task 2: Encode categorical columns
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: Encode categorical columns")
print("=" * 80)

# education_level has a natural order: High School < Bachelors < Masters < PhD
# -> label-encode preserving that order (ordinal encoding is appropriate
# because the categories represent increasing levels of education).
education_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

# remote_work is inherently binary (Yes/No) -> label-encode 0/1. With only
# two categories there is no false-ordinal problem to worry about.
remote_map = {"No": 0, "Yes": 1}

def encode_features(X_raw):
    X = X_raw.copy()
    X["education_level"] = X["education_level"].map(education_order)
    X["remote_work"] = X["remote_work"].map(remote_map)
    # department and region have NO natural order (nominal categories) ->
    # one-hot encode with drop_first=True to avoid multicollinearity
    # (the dropped category becomes the implicit baseline).
    X = pd.get_dummies(X, columns=["department", "region"], drop_first=True)
    return X

X_reg = encode_features(X_reg_raw)
X_clf = encode_features(X_clf_raw)

print("\nX_reg encoded columns:", X_reg.columns.tolist())
print("X_reg dtypes:\n", X_reg.dtypes)
print("\nWhy one-hot for department/region: label-encoding 'Sales'=0,")
print("'Engineering'=1, 'Marketing'=2, ... would falsely imply Marketing is")
print("'twice' Sales and 'closer' to Engineering than to HR, an ordinal")
print("relationship that does not exist among unordered nominal categories.")
print("One-hot encoding represents each category as an independent binary")
print("indicator, avoiding that false-ordering problem entirely.")

# Ensure numeric/bool dtypes are safe for sklearn
X_reg = X_reg.astype(float)
X_clf = X_clf.astype(float)

# ---------------------------------------------------------------------------
# Task 3: Leak-free train/test split and scaling
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: Leak-free train/test split and scaling")
print("=" * 80)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

print("Regression split:", X_reg_train.shape, X_reg_test.shape)
print("Classification split:", X_clf_train.shape, X_clf_test.shape)

scaler_reg = StandardScaler()
scaler_reg.fit(X_reg_train)  # fit ONLY on training data
X_reg_train_scaled = scaler_reg.transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

print("\nNOTE: classification features are scaled in Task 6, AFTER training-set")
print("resampling, so the scaler's statistics reflect only the resampled")
print("training rows and never the test set (see Task 6 for details).")

print("\nNOTE ON LEAKAGE: the StandardScaler is fit() only on X_train in both")
print("cases, never on the full dataset or on X_test. Fitting on the full")
print("dataset would let the mean/std used to scale the training features be")
print("influenced by test-set values, meaning information about the test set")
print("would leak into the training process before the model ever sees it -")
print("producing an overly optimistic evaluation. transform() (not fit) is")
print("applied to X_test so it is scaled using only training-set statistics.")

# ---------------------------------------------------------------------------
# Task 4: Regression - Linear Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: Linear Regression")
print("=" * 80)

lin_reg = LinearRegression()
lin_reg.fit(X_reg_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_reg_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
print(f"Linear Regression - MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}")

coef_table = pd.DataFrame({
    "feature": X_reg.columns,
    "coefficient": lin_reg.coef_
}).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
print("\nCoefficients (sorted by |value|):\n", coef_table)
top3_coefs = coef_table.head(3)
print("\nTop 3 features by |coefficient|:\n", top3_coefs)

# ---------------------------------------------------------------------------
# Task 5: Ridge Regression comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: Ridge Regression")
print("=" * 80)

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_reg_train_scaled, y_reg_train)
y_pred_ridge = ridge_reg.predict(X_reg_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge Regression   - MSE: {mse_ridge:.2f}, R2: {r2_ridge:.4f}")

comparison_table = pd.DataFrame({
    "model": ["LinearRegression (OLS)", "Ridge (alpha=1.0)"],
    "MSE": [mse_lin, mse_ridge],
    "R2": [r2_lin, r2_ridge],
})
print("\nComparison table:\n", comparison_table)

ridge_coef_table = pd.DataFrame({
    "feature": X_reg.columns,
    "ols_coef": lin_reg.coef_,
    "ridge_coef": ridge_reg.coef_,
})
print("\nOLS vs Ridge coefficients:\n", ridge_coef_table)

# ---------------------------------------------------------------------------
# Task 6: Classification - class imbalance check + handling (before/after)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: Class imbalance check")
print("=" * 80)

train_counts_before = y_clf_train.value_counts()
train_props_before = y_clf_train.value_counts(normalize=True)
print("y_clf_train value counts (BEFORE handling):\n", train_counts_before)
print("y_clf_train proportions (BEFORE handling):\n", train_props_before)

minority_prop = train_props_before.min()
print(f"\nMinority class proportion: {minority_prop:.3f} ({minority_prop*100:.1f}%)")
if minority_prop < 0.35:
    print("Minority class < 35% -> imbalance handling required.")
else:
    print("Classes reasonably balanced -> no special handling strictly required.")

print("\nNote: SMOTE (imblearn.over_sampling.SMOTE) was the first choice, but")
print("the imblearn package cannot be installed in this offline environment")
print("(no network access to pip install it). As a substitute in the same")
print("family of techniques, RANDOM OVERSAMPLING is applied instead: minority-")
print("class training rows are resampled with replacement until the classes")
print("are balanced. This is applied ONLY to the training set (never to the")
print("test set) via np.random.Generator.choice, exactly like SMOTE would be.")

# ---- Random oversampling of the minority class (training set only) ----
rng_os = np.random.default_rng(RANDOM_STATE)
minority_class = train_counts_before.idxmin()
majority_class = train_counts_before.idxmax()
n_majority = int(train_counts_before.max())

minority_idx = y_clf_train[y_clf_train == minority_class].index.to_numpy()
majority_idx = y_clf_train[y_clf_train == majority_class].index.to_numpy()

oversampled_minority_idx = rng_os.choice(minority_idx, size=n_majority, replace=True)
resampled_idx = np.concatenate([majority_idx, oversampled_minority_idx])
rng_os.shuffle(resampled_idx)

X_clf_train_res = X_clf_train.loc[resampled_idx].reset_index(drop=True)
y_clf_train_res = y_clf_train.loc[resampled_idx].reset_index(drop=True)

train_counts_after = y_clf_train_res.value_counts()
train_props_after = y_clf_train_res.value_counts(normalize=True)
print("\ny_clf_train value counts (AFTER random oversampling):\n", train_counts_after)
print("y_clf_train proportions (AFTER random oversampling):\n", train_props_after)
print(f"\nTraining rows: {len(y_clf_train)} (before) -> {len(y_clf_train_res)} (after)")

# Scale AFTER resampling, fit only on the resampled training set; the test
# set is transformed with those same statistics and is never touched by
# resampling (it must reflect the real, original class distribution).
scaler_clf = StandardScaler()
scaler_clf.fit(X_clf_train_res)
X_clf_train_scaled = scaler_clf.transform(X_clf_train_res)
X_clf_test_scaled = scaler_clf.transform(X_clf_test)
y_clf_train = y_clf_train_res  # downstream code trains on the resampled labels

# ---------------------------------------------------------------------------
# Task 6 (cont.): Logistic Regression (baseline, C=1.0)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: Logistic Regression (C=1.0, trained on resampled data)")
print("=" * 80)

log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
log_reg.fit(X_clf_train_scaled, y_clf_train)

y_pred_clf = log_reg.predict(X_clf_test_scaled)
y_proba_clf = log_reg.predict_proba(X_clf_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
print("\nConfusion matrix (rows=actual, cols=predicted):\n", cm)

report = classification_report(y_clf_test, y_pred_clf, digits=3)
print("\nClassification report:\n", report)

auc_baseline = roc_auc_score(y_clf_test, y_proba_clf)
print(f"AUC (C=1.0): {auc_baseline:.4f}")

fpr, tpr, thresholds_roc = roc_curve(y_clf_test, y_proba_clf)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_baseline:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - High Earner Classification")
plt.annotate(f"AUC = {auc_baseline:.3f}", xy=(0.55, 0.15), fontsize=12,
             bbox=dict(boxstyle="round", fc="white", ec="gray"))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=150)
plt.close()
print("Saved plots/roc_curve.png")

# ---------------------------------------------------------------------------
# Task 6b: Decision-threshold sensitivity
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6b: Decision-threshold sensitivity (0.30 to 0.70)")
print("=" * 80)

threshold_rows = []
for t in np.arange(0.30, 0.71, 0.10):
    t = round(t, 2)
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t, zero_division=0)
    r = recall_score(y_clf_test, preds_t, zero_division=0)
    f1 = f1_score(y_clf_test, preds_t, zero_division=0)
    threshold_rows.append({"Threshold": t, "Precision": p, "Recall": r, "F1": f1})

threshold_table = pd.DataFrame(threshold_rows)
print("\nThreshold sensitivity table:\n", threshold_table.round(4))

best_f1_row = threshold_table.loc[threshold_table["F1"].idxmax()]
print(f"\nThreshold maximizing F1: {best_f1_row['Threshold']} "
      f"(F1={best_f1_row['F1']:.4f})")

# ---------------------------------------------------------------------------
# Task 7: Regularization experiment (C=0.01 vs C=1.0)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 7: Regularization experiment (C=0.01 vs C=1.0)")
print("=" * 80)

log_reg_strong = LogisticRegression(max_iter=1000, C=0.01,
                                     random_state=RANDOM_STATE)
log_reg_strong.fit(X_clf_train_scaled, y_clf_train)

y_pred_strong = log_reg_strong.predict(X_clf_test_scaled)
y_proba_strong = log_reg_strong.predict_proba(X_clf_test_scaled)[:, 1]

precision_baseline = precision_score(y_clf_test, y_pred_clf, zero_division=0)
recall_baseline = recall_score(y_clf_test, y_pred_clf, zero_division=0)

precision_strong = precision_score(y_clf_test, y_pred_strong, zero_division=0)
recall_strong = recall_score(y_clf_test, y_pred_strong, zero_division=0)
auc_strong = roc_auc_score(y_clf_test, y_proba_strong)

reg_comparison = pd.DataFrame({
    "model": ["C=1.0 (baseline)", "C=0.01 (strong L2)"],
    "Precision": [precision_baseline, precision_strong],
    "Recall": [recall_baseline, recall_strong],
    "AUC": [auc_baseline, auc_strong],
})
print("\nC=1.0 vs C=0.01 comparison:\n", reg_comparison.round(4))

# ---------------------------------------------------------------------------
# Task 7b: Bootstrap confidence interval for AUC difference
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 7b: Bootstrap CI for AUC difference (C=1.0 minus C=0.01)")
print("=" * 80)

rng = np.random.default_rng(RANDOM_STATE)
n_boot = 500
y_clf_test_arr = y_clf_test.to_numpy()
diffs = np.empty(n_boot)

for i in range(n_boot):
    idx = rng.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
    y_sample = y_clf_test_arr[idx]
    # Skip degenerate bootstrap samples with only one class present
    if len(np.unique(y_sample)) < 2:
        diffs[i] = np.nan
        continue
    proba_baseline_sample = y_proba_clf[idx]
    proba_strong_sample = y_proba_strong[idx]
    auc_b = roc_auc_score(y_sample, proba_baseline_sample)
    auc_s = roc_auc_score(y_sample, proba_strong_sample)
    diffs[i] = auc_b - auc_s

valid_diffs = diffs[~np.isnan(diffs)]
mean_diff = np.mean(valid_diffs)
ci_lower = np.percentile(valid_diffs, 2.5)
ci_upper = np.percentile(valid_diffs, 97.5)

print(f"Valid bootstrap samples used: {len(valid_diffs)} / {n_boot}")
print(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
excludes_zero = (ci_lower > 0) or (ci_upper < 0)
print(f"95% CI excludes zero: {excludes_zero}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
