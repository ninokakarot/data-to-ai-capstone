# Part 2 — Supervised Machine Learning: Regression & Classification

## 1. Setup / how to run

```bash
pip install pandas numpy scikit-learn matplotlib
python3 model_pipeline.py     # runs everything, writes plots/roc_curve.png
```

Input: `cleaned_data.csv` (copied here from Part 1's output — 650 rows × 11
columns, already null-free and duplicate-free).

## 2. Target definitions (Task 1)

- **`y_reg` (regression target): `monthly_salary`** — a continuous numeric
  column, predicted from the other employee attributes.
- **`y_clf` (classification target): `high_earner`**, defined as
  `y_clf = (monthly_salary >= monthly_salary.quantile(0.70)).astype(int)`
  → 1 if an employee's salary is in the **top 30%** of the company, else 0.

  **Why the 70th percentile instead of the median:** the task allows
  binarizing `y_reg` "at its median... or another natural binary column."
  A median split on this dataset is almost exactly 50/50, which sails past
  the imbalance check without ever exercising it. Cutting at the 70th
  percentile keeps the same "binarize the continuous target" approach but
  produces a **genuinely imbalanced (30% positive), business-meaningful
  label** ("is this employee a high earner?"), which lets Task 6's
  imbalance-handling requirement actually apply and be demonstrated
  properly.

  **Leakage note:** since `y_clf` is directly derived from `monthly_salary`,
  `monthly_salary` is **excluded from `X_clf`** (see below) — otherwise the
  classifier could simply threshold that one column and "cheat" with 100%
  accuracy without learning anything.

## 3. Feature matrices

- `X_reg` = all columns **except** `employee_id` (identifier, not
  predictive) and `monthly_salary` (this is `y_reg` itself).
- `X_clf` = all columns **except** `employee_id` and `monthly_salary` (this
  is the source of `y_clf` — must be excluded to avoid trivial leakage).

Both `X_reg` and `X_clf` therefore contain the same 9 raw columns: `age`,
`department`, `education_level`, `region`, `remote_work`,
`years_experience`, `performance_score`, `satisfaction_score`, `bonus_pct`.

## 4. Categorical encoding (Task 2)

| Column | Encoding | Justification |
|---|---|---|
| `education_level` | **Label/ordinal** encoding: `High School=0, Bachelors=1, Masters=2, PhD=3` | These categories represent a genuine, universally-agreed increasing order of educational attainment, so mapping them to increasing integers preserves real information (more education → higher code) instead of discarding it. |
| `remote_work` | **Label** encoding: `No=0, Yes=1` | Binary category — with only two possible values there's no ordinal relationship to falsely imply either way, so a simple 0/1 mapping is equivalent to one-hot encoding here (and simpler). |
| `department` | **One-hot** encoding (`pd.get_dummies(..., drop_first=True)`) | Department names (Sales, Engineering, Marketing, Support, Finance, HR) have **no inherent order**. Label-encoding them (e.g., Sales=0, Engineering=1, Marketing=2, ...) would falsely imply Marketing is "twice" Sales and numerically "closer" to Engineering than to HR — a false ordinal relationship the model could pick up as spurious signal. One-hot encoding instead represents each department as an independent yes/no indicator, so no department is treated as numerically "between" or "closer to" another. |
| `region` | **One-hot** encoding (`drop_first=True`) | Same reasoning as `department` — North/South/East/West are nominal, unordered categories. |

`drop_first=True` drops one dummy column per categorical feature (e.g., no
`department_Engineering` column exists — it's the implied baseline when all
other department dummies are 0), which avoids the dummy-variable trap
(perfect multicollinearity between the one-hot columns).

## 5. Leak-free train/test split and scaling (Task 3)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)                 # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why fitting the scaler on the full dataset (or on `X_test`) would be data
leakage:** `StandardScaler` computes each feature's mean and standard
deviation to standardize it. If those statistics are computed over the
combined train+test data, the mean/std used to scale the *training* features
would already be influenced by values that only exist in the *test* set —
meaning information about the test set (which is supposed to be completely
unseen, held out for a fair final evaluation) leaks into the training
pipeline before the model is ever fit. This produces an overly optimistic
performance estimate that would not generalize to genuinely new data. The
fix is to call `.fit()` only on `X_train`, and only `.transform()` (never
re-fit) on `X_test`, so the test set is scaled using exclusively
training-set statistics.

Classification split uses `stratify=y_clf` to keep the 70/30 class ratio
consistent between train and test sets.

## 6. Regression — Linear Regression (Task 4)

| Metric | Value |
|---|---|
| MSE | 1,361,059.96 |
| R² | 0.7337 |

**Coefficients, top 3 by absolute value** (features are standardized, so
coefficients are directly comparable in scale):

| Feature | Coefficient |
|---|---|
| department_Support | -1562.07 |
| department_Sales | -1534.95 |
| department_Marketing | -1194.85 |

(For reference, `years_experience` has the largest *non-department*
coefficient at +1110.36.)

**Interpreting the coefficients:**
- A **large positive coefficient** (e.g., `years_experience` ≈ +1110) means
  that a one-standard-deviation increase in that (scaled) feature is
  associated with an increase of about that many dollars in predicted
  monthly salary, holding all other features constant.
- A **large negative coefficient** (e.g., `department_Support` ≈ -1562)
  means that, relative to the baseline department (Engineering — the
  dropped dummy), being in the Support department is associated with a
  predicted monthly salary about $1,562 lower, holding everything else
  constant. Since these are one-hot dummy features (0/1, not standardized
  the same way as continuous ones), the coefficient is read directly as a
  dollar adjustment relative to the Engineering baseline.

## 7. Ridge Regression comparison (Task 5)

| Model | MSE | R² |
|---|---|---|
| LinearRegression (OLS) | 1,361,059.96 | 0.7337 |
| Ridge (alpha=1.0) | 1,357,044.58 | 0.7345 |

Ridge performs essentially the same as OLS here (marginally lower MSE,
marginally higher R²), because the alpha=1.0 penalty is quite mild relative
to the coefficient scale in this dataset — the shrinkage barely moves most
coefficients (e.g., `years_experience`: 1110.36 → 1105.85).

**Why Ridge can produce a different coefficient profile than OLS, and what
`alpha` controls:** Ridge Regression adds an L2 penalty term
(`alpha * sum(coef^2)`) to the OLS loss function, which shrinks all
coefficients toward zero in proportion to their size — larger coefficients
get shrunk more in absolute terms. This matters most when features are
highly correlated (multicollinear): OLS can assign large, unstable, and
sometimes counter-intuitive-signed coefficients to correlated features
because it's free to trade off between them; Ridge instead spreads the
"credit" more evenly across correlated features (like `age` and
`years_experience`, r=0.87 from Part 1) and generally makes the coefficient
estimates smaller and more stable. The `alpha` parameter controls the
*strength* of that penalty: `alpha=0` recovers plain OLS exactly; larger
`alpha` shrinks coefficients more aggressively (trading a little bias for
lower variance), which is especially useful as a defense against
overfitting when there are many features or strong collinearity.

## 8. Classification — class imbalance (Task 6)

`y_clf_train` value counts **before** any handling:

| Class | Count | Proportion |
|---|---|---|
| 0 (not high earner) | 364 | 70.0% |
| 1 (high earner) | 156 | 30.0% |

The minority class (30.0%) falls below the 35% threshold, so imbalance
handling is required.

**Chosen strategy: random oversampling of the minority training class**
(a manual substitute for SMOTE). SMOTE
(`imblearn.over_sampling.SMOTE`) was the first choice, but the `imblearn`
package **cannot be installed in this offline environment** (no internet
access to `pip install` it). As a substitute in the same family of
techniques, minority-class rows in the **training set only** are resampled
**with replacement** (via `np.random.Generator.choice`) until both classes
have equal counts — the same "oversample the training data" principle SMOTE
uses, just drawing duplicate real rows instead of synthesizing interpolated
ones. Critically, this is applied **only to `X_clf_train` / `y_clf_train`**,
never to the test set, so the test set still reflects the true, original
30/70 class distribution for an honest evaluation.

`y_clf_train` value counts **after** oversampling:

| Class | Count | Proportion |
|---|---|---|
| 0 (not high earner) | 364 | 50.0% |
| 1 (high earner) | 364 | 50.0% |

**Training rows: 520 (before) → 728 (after)** — the minority class (156
rows) was resampled with replacement up to 364 rows to match the majority
class, so total training rows grew from 520 to 728. The `StandardScaler` is
fit only on this resampled training set (never on the test set), and that
same fitted scaler is used to transform the untouched test set.

## 9. Logistic Regression results (Task 6, `C=1.0`)

**Confusion matrix** (rows = actual, columns = predicted; 0=not high
earner, 1=high earner):

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 85 | 6 |
| **Actual 1** | 0 | 39 |

**Classification report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 1.000 | 0.934 | 0.966 | 91 |
| 1 | 0.867 | 1.000 | 0.929 | 39 |
| **Accuracy** | | | **0.954** | 130 |

**AUC: 0.9946** — plotted in `plots/roc_curve.png`.

**(a) Precision and Recall formulas** (TP/FP/FN with respect to the
positive class, "high earner"):

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**(b) Which metric matters more here?** For a "high earner" flag likely used
to route people toward retention bonuses, promotion review, or targeted
benefits, **missing a genuine high earner (a false negative) is usually more
costly** than mistakenly flagging someone who isn't (a false positive) — a
missed high-value employee might be under-invested-in and leave, whereas a
false positive mainly costs a slightly wasted review. That makes **recall
the more important metric** for this task. This model already achieves a
**perfect recall of 1.000** for class 1 on the test set.

**(c) What AUC = 0.9946 means:** AUC measures the model's ability to rank a
randomly chosen high earner above a randomly chosen non-high-earner, across
all possible thresholds. An AUC of 0.9946 (out of a maximum of 1.0, where
0.5 is random guessing) means the model separates the two classes almost
perfectly — it is very rarely wrong about which of two employees is more
likely to be a high earner, which makes sense given the target is derived
directly from salary and salary itself is strongly predictable from
`years_experience`, `age`, and `department` (as seen in the regression's R²
of 0.73).

## 10. Decision-threshold sensitivity (Task 6b)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.7647 | 1.0000 | 0.8667 |
| 0.40 | 0.8478 | 1.0000 | 0.9176 |
| **0.50** | **0.8667** | **1.0000** | **0.9286** |
| 0.60 | 0.8810 | 0.9487 | 0.9136 |
| 0.70 | 0.9459 | 0.8974 | 0.9211 |

**(a) Formulas:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**(b) Threshold that maximizes F1: 0.50** (F1 = 0.9286).

**(c) Which matters more, precision or recall?** As discussed above,
**recall** is the more important metric for this task — the cost of
overlooking a genuine high earner (a missed retention opportunity) outweighs
the cost of a false positive (an unnecessary review).

**(d) Raise or lower the threshold?** To optimize purely for recall, you
would **lower** the threshold — thresholds 0.30, 0.40, and 0.50 all achieve
perfect recall (1.000) on this test set, so there's no need to go below
0.50 to catch every true high earner. **The cost of lowering the threshold**
below 0.50 (e.g., to 0.30) is a drop in precision (0.87 → 0.76), meaning
more false positives — employees flagged as high earners who aren't, adding
unnecessary review overhead. Since 0.50 already achieves perfect recall
while keeping precision at its highest point among the perfect-recall
thresholds, **0.50 is the best choice here** — pushing the threshold lower
would only sacrifice precision with no further recall benefit.

## 11. Regularization experiment (Task 7)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.8667 | 1.0000 | 0.9946 |
| C=0.01 (strong L2) | 0.8298 | 1.0000 | 0.9904 |

**What `C` controls:** in scikit-learn's `LogisticRegression`, `C` is the
**inverse** of the regularization strength — smaller `C` means a **stronger**
L2 penalty on the coefficients (more shrinkage toward zero), while larger
`C` means a **weaker** penalty (coefficients are freer to fit the training
data closely, closer to unregularized logistic regression). Going from
`C=1.0` to `C=0.01` therefore applies much heavier regularization.

**Did reducing C help or hurt?** On this dataset, reducing `C` to 0.01
**slightly worsened** performance: AUC dropped from 0.9946 to 0.9904, and
precision dropped from 0.867 to 0.830 (recall stayed perfect at 1.000 for
both). This suggests the `C=1.0` model was not overfitting badly in the
first place — the underlying salary-vs-features relationship is strong and
fairly simple (mostly linear/additive across department, experience, and
age), so heavier regularization mostly just adds unnecessary bias rather
than removing overfitting variance.

## 12. Bootstrap confidence interval for AUC difference (Task 7b)

Procedure: 500 bootstrap resamples of the test set (drawn with replacement
via `np.random.choice`), computing `AUC(C=1.0) - AUC(C=0.01)` on each
resample.

| Statistic | Value |
|---|---|
| Mean AUC difference (C=1.0 − C=0.01) | **+0.0040** |
| 95% CI (2.5th percentile) | **-0.0051** |
| 95% CI (97.5th percentile) | **+0.0162** |

**The 95% confidence interval `[-0.0051, +0.0162]` includes zero.** This
indicates the small average AUC advantage of the `C=1.0` model over the
`C=0.01` model (about 0.4 percentage points) is **not reliably distinguishable
from zero** across different resamples of the test data — in other words,
with both models already performing so well (AUC ≈ 0.99–0.995), the
remaining gap between them is small enough that it could plausibly be due
to sampling noise rather than a genuine, consistent advantage of weaker
regularization on this dataset.

## 13. Files in this repository

```
part2/
├── README.md
├── model_pipeline.py       # all preprocessing, model training, and evaluation code
├── cleaned_data.csv         # input data (copied from Part 1's output)
├── console_output.txt       # full run log (all printed tables/metrics)
└── plots/
    └── roc_curve.png
```
