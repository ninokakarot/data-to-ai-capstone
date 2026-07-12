"""
Part 3 - Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline
Input: cleaned_data.csv (from Part 1)

This script first reproduces the exact same target definitions, encoding,
train/test split, and training-set oversampling used in Part 2 (same
random_state=42 throughout), so that "the Logistic Regression from Part 2"
and "X_train_scaled / X_test_scaled / y_clf_train / y_clf_test" referred to
in the Part 3 spec are consistent with Part 2's actual outputs.

Run with: python3 advanced_modeling.py
Outputs: plots/learning_curve.png, best_model.pkl, console_output.txt
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("plots", exist_ok=True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

RANDOM_STATE = 42

# ===========================================================================
# STEP 0: Reproduce Part 2's preprocessing exactly (same targets, encoding,
# split, and training-set oversampling) so Part 3 builds on the same data.
# ===========================================================================
print("=" * 80)
print("STEP 0: Reproducing Part 2 preprocessing (targets, encoding, split, resampling)")
print("=" * 80)

df = pd.read_csv("cleaned_data.csv")

y_reg_full = df["monthly_salary"].copy()
SALARY_QUANTILE = 0.70
salary_threshold = y_reg_full.quantile(SALARY_QUANTILE)
y_clf_full = (y_reg_full >= salary_threshold).astype(int)

drop_cols_common = ["employee_id"]
X_clf_raw = df.drop(columns=drop_cols_common + ["monthly_salary"])

education_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
remote_map = {"No": 0, "Yes": 1}

def encode_features(X_raw):
    X = X_raw.copy()
    X["education_level"] = X["education_level"].map(education_order)
    X["remote_work"] = X["remote_work"].map(remote_map)
    X = pd.get_dummies(X, columns=["department", "region"], drop_first=True)
    return X

X_clf = encode_features(X_clf_raw).astype(float)
feature_names = X_clf.columns.tolist()

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf_full
)

# Random oversampling of the minority training class (same as Part 2)
rng_os = np.random.default_rng(RANDOM_STATE)
train_counts_before = y_clf_train.value_counts()
minority_class = train_counts_before.idxmin()
majority_class = train_counts_before.idxmax()
n_majority = int(train_counts_before.max())

minority_idx = y_clf_train[y_clf_train == minority_class].index.to_numpy()
majority_idx = y_clf_train[y_clf_train == majority_class].index.to_numpy()
oversampled_minority_idx = rng_os.choice(minority_idx, size=n_majority, replace=True)
resampled_idx = np.concatenate([majority_idx, oversampled_minority_idx])
rng_os.shuffle(resampled_idx)

X_clf_train = X_clf_train.loc[resampled_idx].reset_index(drop=True)  # unscaled, resampled
y_clf_train = y_clf_train.loc[resampled_idx].reset_index(drop=True)  # resampled labels
X_clf_test = X_clf_test.reset_index(drop=True)
y_clf_test = y_clf_test.reset_index(drop=True)

print("Resampled training set shape:", X_clf_train.shape)
print("Test set shape:", X_clf_test.shape)
print("y_clf_train balance:\n", y_clf_train.value_counts(normalize=True))

scaler = StandardScaler()
scaler.fit(X_clf_train)  # fit only on (resampled) training data
X_clf_train_scaled = scaler.transform(X_clf_train)
X_clf_test_scaled = scaler.transform(X_clf_test)

# Refit the Part 2 baseline Logistic Regression (C=1.0) for later comparison
log_reg_part2 = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
log_reg_part2.fit(X_clf_train_scaled, y_clf_train)
print("Refit Part 2 Logistic Regression (C=1.0) for comparison purposes.")

# ===========================================================================
# TASK 1: Decision Tree baseline (unconstrained)
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 1: Decision Tree baseline (max_depth=None)")
print("=" * 80)

dt_baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_baseline.fit(X_clf_train_scaled, y_clf_train)

train_acc_baseline = accuracy_score(y_clf_train, dt_baseline.predict(X_clf_train_scaled))
test_acc_baseline = accuracy_score(y_clf_test, dt_baseline.predict(X_clf_test_scaled))
print(f"Unconstrained tree - Train accuracy: {train_acc_baseline:.4f}, "
      f"Test accuracy: {test_acc_baseline:.4f}")
print(f"Train-test gap: {train_acc_baseline - test_acc_baseline:.4f}")

# ===========================================================================
# TASK 2: Controlled Decision Tree
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 2: Controlled Decision Tree (max_depth=5, min_samples_split=20)")
print("=" * 80)

dt_controlled = DecisionTreeClassifier(
    max_depth=5, min_samples_split=20, random_state=RANDOM_STATE
)
dt_controlled.fit(X_clf_train_scaled, y_clf_train)

train_acc_controlled = accuracy_score(y_clf_train, dt_controlled.predict(X_clf_train_scaled))
test_acc_controlled = accuracy_score(y_clf_test, dt_controlled.predict(X_clf_test_scaled))
print(f"Controlled tree - Train accuracy: {train_acc_controlled:.4f}, "
      f"Test accuracy: {test_acc_controlled:.4f}")
print(f"Train-test gap: {train_acc_controlled - test_acc_controlled:.4f}")

# ===========================================================================
# TASK 3: Gini vs Entropy comparison
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 3: Gini vs Entropy comparison (both max_depth=5)")
print("=" * 80)

dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE)
dt_gini.fit(X_clf_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_clf_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=RANDOM_STATE)
dt_entropy.fit(X_clf_train_scaled, y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_clf_test_scaled))

print(f"Gini    (max_depth=5) - Test accuracy: {test_acc_gini:.4f}")
print(f"Entropy (max_depth=5) - Test accuracy: {test_acc_entropy:.4f}")

# ===========================================================================
# TASK 4: Random Forest
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 4: Random Forest (n_estimators=100, max_depth=10)")
print("=" * 80)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_model.fit(X_clf_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_clf_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_clf_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_clf_test_scaled)[:, 1])

print(f"Random Forest - Train accuracy: {rf_train_acc:.4f}, "
      f"Test accuracy: {rf_test_acc:.4f}, Test AUC: {rf_test_auc:.4f}")

rf_importances = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nTop 5 features by importance:\n", rf_importances.head(5))
print("\nAll feature importances:\n", rf_importances)

# ===========================================================================
# TASK 4a: Gradient Boosting
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 4a: Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3)")
print("=" * 80)

gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
)
gb_model.fit(X_clf_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_clf_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_clf_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_clf_test_scaled)[:, 1])

print(f"Gradient Boosting - Train accuracy: {gb_train_acc:.4f}, "
      f"Test accuracy: {gb_test_acc:.4f}, Test AUC: {gb_test_auc:.4f}")

# ===========================================================================
# TASK 4b: Feature ablation study
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 4b: Feature ablation study")
print("=" * 80)

lowest5 = rf_importances.tail(5)["feature"].tolist()
print("5 lowest-importance features (to be removed):", lowest5)

keep_idx = [i for i, f in enumerate(feature_names) if f not in lowest5]
X_train_reduced = X_clf_train_scaled[:, keep_idx]
X_test_reduced = X_clf_test_scaled[:, keep_idx]

rf_full_auc = rf_test_auc  # from Task 4, full feature set
rf_reduced_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_reduced_model.fit(X_train_reduced, y_clf_train)
rf_reduced_auc = roc_auc_score(
    y_clf_test, rf_reduced_model.predict_proba(X_test_reduced)[:, 1]
)

print(f"Full model    (all {len(feature_names)} features) - Test AUC: {rf_full_auc:.4f}")
print(f"Reduced model ({len(keep_idx)} features)        - Test AUC: {rf_reduced_auc:.4f}")
print(f"AUC change: {rf_reduced_auc - rf_full_auc:+.4f}")

# ===========================================================================
# TASK 5: Cross-validated comparison
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 5: Cross-validated comparison (5-fold StratifiedKFold, scoring=roc_auc)")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_models = {
    "Logistic Regression (Part 2)": log_reg_part2,
    "Decision Tree (max_depth=5)": dt_controlled,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}

cv_results = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_clf_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: mean AUC = {scores.mean():.4f}, std = {scores.std():.4f}")

cv_table = pd.DataFrame([
    {"model": name, "cv_mean_auc": m, "cv_std_auc": s}
    for name, (m, s) in cv_results.items()
])
print("\nCross-validation summary:\n", cv_table)

# ===========================================================================
# TASK 6: Hyperparameter tuning with GridSearchCV
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 6: GridSearchCV on Random Forest pipeline")
print("=" * 80)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE),
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
print(f"Total hyperparameter configurations: {n_configs} "
      f"(x 5 folds = {n_configs * 5} total model fits)")

grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
)
# NOTE: fit on the UNSCALED, resampled X_clf_train / y_clf_train - the
# pipeline itself handles imputation and scaling internally, per fold.
grid_search.fit(X_clf_train, y_clf_train)

print("\nBest params:", grid_search.best_params_)
print("Best CV score (mean AUC):", grid_search.best_score_)

best_pipeline = grid_search.best_estimator_

# ===========================================================================
# TASK 7: Manual learning curve
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 7: Manual learning curve (20% - 100% of training data)")
print("=" * 80)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_rows = []

for f in fractions:
    n_rows = int(f * len(X_clf_train))
    X_subset = X_clf_train.iloc[:n_rows]
    y_subset = y_clf_train.iloc[:n_rows]

    # Fresh pipeline instance with the SAME best hyperparameters, refit on subset
    from sklearn.base import clone
    pipeline_f = clone(best_pipeline)
    pipeline_f.fit(X_subset, y_subset)

    train_auc = roc_auc_score(y_subset, pipeline_f.predict_proba(X_subset)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipeline_f.predict_proba(X_clf_test)[:, 1])

    learning_curve_rows.append({
        "Training fraction": f, "n_rows": n_rows,
        "Training AUC": train_auc, "Test AUC": test_auc
    })
    print(f"Fraction={f:.1f} (n={n_rows}): Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")

learning_curve_table = pd.DataFrame(learning_curve_rows)
print("\nLearning curve table:\n", learning_curve_table)

plt.figure(figsize=(8, 5.5))
plt.plot(learning_curve_table["Training fraction"], learning_curve_table["Training AUC"],
          marker="o", label="Training AUC")
plt.plot(learning_curve_table["Training fraction"], learning_curve_table["Test AUC"],
          marker="o", label="Test AUC")
plt.xlabel("Training set fraction")
plt.ylabel("ROC-AUC")
plt.title("Manual Learning Curve - Tuned Random Forest Pipeline")
plt.legend()
plt.tight_layout()
plt.savefig("plots/learning_curve.png", dpi=150)
plt.close()
print("Saved plots/learning_curve.png")

# ===========================================================================
# TASK 8: Serialize the best model
# ===========================================================================
print("\n" + "=" * 80)
print("TASK 8: Serialize best pipeline")
print("=" * 80)

joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best_model.pkl")

# Reload-and-predict demonstration (also saved separately in reload_predict.py)
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
])[feature_names]  # ensure column order matches training data

predictions = loaded_pipeline.predict(hand_crafted_rows)
probabilities = loaded_pipeline.predict_proba(hand_crafted_rows)[:, 1]
print("\nReload-and-predict demonstration:")
print("Predictions:", predictions)
print("Probabilities (P(high_earner)):", probabilities)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
