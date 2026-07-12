# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## 1. Setup / how to run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python3 advanced_modeling.py     # runs everything, writes plots/learning_curve.png + best_model.pkl
python3 reload_predict.py        # standalone reload-and-predict demonstration
```

Input: `cleaned_data.csv` (from Part 1). **Note:** this script begins by
reproducing Part 2's exact target definition, encoding, train/test split,
and training-set oversampling (same `random_state=42` throughout), so that
"the Logistic Regression from Part 2" and `X_train_scaled` /
`X_test_scaled` / `y_clf_train` / `y_clf_test` referenced in the task are
consistent with Part 2's actual outputs rather than being redefined
ad hoc. Concretely:

- `y_clf` = `high_earner` = 1 if `monthly_salary` ≥ its 70th percentile, else 0
- Features: same 9 raw columns as Part 2 (`age`, `department`,
  `education_level`, `region`, `remote_work`, `years_experience`,
  `performance_score`, `satisfaction_score`, `bonus_pct`), encoded the same
  way (ordinal for `education_level`, binary for `remote_work`, one-hot for
  `department`/`region`)
- Same 80/20 stratified split, same random-oversampling of the minority
  training class (520 → 728 rows, 50/50 balanced)
- `X_train_scaled` / `X_test_scaled` = the resampled training features and
  the (untouched) test features, both transformed by a `StandardScaler` fit
  only on the resampled training set

## 2. Decision Tree baseline (Task 1)

| | Train accuracy | Test accuracy |
|---|---|---|
| Unconstrained (`max_depth=None`) | **1.0000** | 0.9538 |

**Train-test gap: 0.0462.** A perfect training accuracy of 1.0 combined
with a noticeably lower test accuracy is the textbook signature of
overfitting: the tree has grown deep enough to carve out a leaf for
essentially every training example (memorizing the training set), rather
than learning a generalizable decision boundary.

**Why decision trees are high-variance models:** at each node, a single
decision tree greedily picks the split that best separates the *training*
data at that node, and then never revisits or adjusts that decision based
on how later splits turn out. Because early splits are chosen locally and
irrevocably, a small change in the training data (a few different rows)
can cascade into a very different tree structure overall — the model's
final shape is highly sensitive to exactly which rows it happened to see,
which is precisely what "high variance" means.

## 3. Controlled Decision Tree (Task 2)

| | Train accuracy | Test accuracy | Train-test gap |
|---|---|---|---|
| Unconstrained | 1.0000 | 0.9538 | 0.0462 |
| Controlled (`max_depth=5, min_samples_split=20`) | 0.8942 | 0.8154 | 0.0788 |

**Role of `max_depth`:** caps how many sequential splits any path from root
to leaf can have, directly limiting how finely the tree can carve up the
feature space. A shallower tree cannot memorize as many individual training
examples, which reduces variance — at the cost of some bias, since it may
now be too simple to capture all genuine structure in the data.

**Role of `min_samples_split`:** prevents a node from splitting further if
it has fewer than this many samples, which stops the tree from creating
splits based on very small, noise-prone subsets of the data (a split
decided by only 3–4 rows is unlikely to generalize).

**Comparing the gap:** interestingly, the controlled tree's absolute
train-test gap (0.0788) is *larger* than the unconstrained tree's (0.0462)
even though its train accuracy is much lower (0.894 vs 1.000) — this is
because the unconstrained tree, by memorizing the training set, still
manages to also fit the test set fairly well (0.9538) since apparently much
of the boundary structure it memorized still generalized reasonably; the
constrained tree, forced into a coarser decision boundary, actually
generalizes somewhat worse on this particular dataset/split. This is a
useful reminder that overfitting is best diagnosed by looking at *test*
performance directly (and via cross-validation, Task 5) rather than only at
the size of the train-test gap.

## 4. Gini vs Entropy comparison (Task 3)

| Criterion | Test accuracy (max_depth=5) |
|---|---|
| Gini | 0.8154 |
| Entropy | 0.8462 |

**Gini impurity:**
```
Gini = 1 - Σ pᵢ²
```
where `pᵢ` is the proportion of samples belonging to class `i` in a node.

**Entropy:**
```
Entropy = -Σ pᵢ log₂(pᵢ)
```

**What Gini = 0 means:** a node with Gini impurity of 0 is perfectly
"pure" — every sample in that node belongs to the same single class (one
`pᵢ = 1` and all others `pᵢ = 0`), so `1 - 1² = 0`. A pure node needs no
further splitting, since predicting its majority (only) class is already
100% correct for every training sample that reaches it.

Entropy performed slightly better than Gini here (0.846 vs 0.815 test
accuracy); in practice the two criteria usually produce very similar trees,
since both measure node impurity and are minimized by the same kinds of
splits — the difference here is a modest, dataset-specific effect rather
than a systematic advantage of one criterion.

## 5. Random Forest (Task 4)

| Metric | Value |
|---|---|
| Train accuracy | 0.9959 |
| Test accuracy | 0.9538 |
| Test AUC | **0.9927** |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| years_experience | 0.2682 |
| age | 0.1715 |
| department_Sales | 0.1098 |
| department_Marketing | 0.0798 |
| department_Support | 0.0753 |

**How Random Forest computes feature importance:** for each feature, the
algorithm looks at every split across every tree in the forest that used
that feature, and averages how much each such split reduced Gini impurity
(weighted by how many samples reached that node). Features that are
repeatedly chosen for splits that cleanly separate classes accumulate a
high total importance score; features rarely used, or used only for splits
that barely improve purity, get low scores. This is fundamentally different
from a **linear regression coefficient**, which measures the size and
direction of a strict linear, additive effect on the continuous outcome,
holding all other features fixed — a Random Forest's importance score, by
contrast, is a *non-linear, interaction-aware, non-directional* measure
(it doesn't tell you whether the feature pushes predictions up or down,
only how useful it was for splitting), and it is computed from tree
structure rather than from an analytic model equation.

**Bagging explained:** a Random Forest builds many individual decision
trees, and reduces variance through two separate sources of randomness. (1)
**Bootstrap sampling:** each tree is trained on a random sample of the same
size as the training set, drawn *with replacement*, so each tree sees a
slightly different subset of rows (some duplicated, some left out
entirely). (2) **Random feature subsets:** at each split, instead of
considering every available feature, only a random subset of roughly
√(number of features) candidates is considered, forcing different trees to
rely on different features even when one feature would otherwise dominate
every tree's early splits. Because each tree is trained on different data
and considers different features, individual trees end up quite different
from one another and make somewhat independent errors; averaging their
predictions (or, for classification, averaging predicted probabilities /
majority voting) cancels out much of that independent, tree-specific noise
— which is exactly why a Random Forest's test performance (AUC 0.9927) is
so much more stable than a single deep, unconstrained tree, even though
each individual tree in the forest is itself still a high-variance model.

## 6. Gradient Boosting (Task 4a)

| Metric | Value |
|---|---|
| Train accuracy | 0.9973 |
| Test accuracy | 0.9538 |
| Test AUC | **0.9969** |

Gradient Boosting narrowly edges out the Random Forest on test AUC (0.9969
vs 0.9927) — consistent with boosting's approach of building trees
sequentially, each one specifically correcting the errors of the ensemble
so far, which often squeezes out marginally better performance than
bagging's independent-trees-averaged approach, at the cost of being more
sensitive to hyperparameters like `learning_rate` and `n_estimators`.

## 7. Feature ablation study (Task 4b)

**5 lowest-importance features (removed):** `education_level`,
`remote_work`, `region_North`, `region_West`, `region_South`.

| Model | Features | Test AUC |
|---|---|---|
| Full Random Forest | 15 | 0.9927 |
| Reduced Random Forest | 10 | **0.9930** |

**AUC change: +0.0003** (essentially unchanged, marginally *higher*
without the 5 dropped features).

**Interpretation:** the removed features were **genuinely uninformative**
for this task — removing them didn't hurt performance at all (AUC even
ticked up very slightly, likely just noise from the different random
splits chosen during tree-building rather than a real improvement). This
matches intuition: `region` was never meaningfully tied to salary in this
dataset (Part 1's correlation analysis showed no such relationship), and
`education_level` and `remote_work` had very low direct predictive power on
salary once `years_experience`, `age`, and `department` were already
available.

**Production trade-off:** dropping 5 of 15 features (a 33% reduction) with
no measurable AUC cost is a genuinely good deal for deployment — fewer
features to collect, validate, and monitor for drift; a slightly smaller,
faster-to-score model; and one less source of potential missing-data or
schema-mismatch bugs in a production feature pipeline. This is only a good
trade in general, though, when the AUC degradation (here, effectively zero)
stays below whatever tolerance the business sets — if dropping features had
cost, say, 3-5 points of AUC, the added simplicity would need to be weighed
against the real cost of worse predictions (e.g., missed high earners in
this use case).

## 8. Cross-validated comparison (Task 5)

5-fold `StratifiedKFold(shuffle=True, random_state=42)`, `scoring='roc_auc'`,
evaluated on the resampled training set:

| Model | CV mean AUC | CV std AUC |
|---|---|---|
| Logistic Regression (Part 2) | 0.9847 | 0.0098 |
| Decision Tree (max_depth=5) | 0.9476 | 0.0162 |
| Random Forest | **0.9927** | 0.0039 |
| Gradient Boosting | 0.9923 | 0.0123 |

**Why cross-validation gives a more reliable estimate than a single
train-test split:** a single 80/20 split's test score depends heavily on
exactly which rows happened to land in the test set — an unusually easy or
unusually hard test subset can make a model look better or worse than it
really is, purely by chance. 5-fold cross-validation instead rotates
through 5 different train/test partitions of the same data, computing a
score on each fold, so the final mean is averaged over 5 independent looks
at generalization performance (and the standard deviation directly
quantifies how much that estimate might vary with a different split) —
giving a much more stable, trustworthy estimate than any single split could.

## 9. GridSearchCV hyperparameter tuning (Task 6)

Pipeline: `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))`

Parameter grid:
```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

**Total configurations evaluated:** 3 × 3 × 2 = **18** distinct
hyperparameter combinations, each evaluated across 5 CV folds = **90 total
model fits**.

**Best parameters found:**
```
{'randomforestclassifier__max_depth': None,
 'randomforestclassifier__min_samples_leaf': 1,
 'randomforestclassifier__n_estimators': 200}
```
**Best CV score (mean AUC): 0.9939**

**Grid Search vs Randomized Search trade-off:** `GridSearchCV` exhaustively
evaluates every combination in the grid, which guarantees finding the best
combination *within that grid* but scales multiplicatively with the number
of hyperparameters and values tried (adding one more parameter with 3
options would roughly triple the total fits required) — this becomes
computationally expensive fast for larger grids or slower models.
`RandomizedSearchCV` instead samples a fixed number of random combinations
from the specified (or sampled) parameter space, which is far cheaper for
large search spaces and still tends to find near-optimal regions in
practice (research has shown random search is often nearly as effective as
grid search per unit of compute, since not all hyperparameters matter
equally), at the cost of no longer guaranteeing that the single best grid
point was actually tried.

## 10. Manual learning curve (Task 7)

Using the best pipeline from Task 6, refit from scratch on progressively
larger prefixes of the (resampled) training data:

| Training fraction | n rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 145 | 1.0000 | 0.9689 |
| 0.4 | 291 | 1.0000 | 0.9760 |
| 0.6 | 436 | 1.0000 | 0.9855 |
| 0.8 | 582 | 1.0000 | 0.9944 |
| 1.0 | 728 | 1.0000 | 0.9934 |

See `plots/learning_curve.png`.

**(i) Does training AUC decrease as the training set grows?** No — training
AUC stays at a perfect 1.0000 across every fraction. This is expected for
this particular model: an unconstrained/lightly-constrained Random Forest
(with `max_depth=None` as chosen by GridSearchCV) is flexible enough to fit
essentially any training set perfectly regardless of size, so it never
shows the classic "training score gently declines as more data arrives"
pattern that simpler, more constrained models often show.

**(ii) Does test AUC increase with more training data?** Yes, clearly — test
AUC rises steadily from 0.9689 (20% of the data) up to 0.9944 (80% of the
data), a meaningful improvement. It dips very slightly at 100% (0.9934 vs
0.9944 at 80%), which is within the range of ordinary run-to-run noise
rather than a real reversal of the trend.

**(iii) Conclusion — data-limited or capacity-limited?** The steady rise in
test AUC through 80% of the data, followed by a plateau (not a continued
climb) at 100%, suggests the model is now **close to its practical
ceiling for the current feature set on this task** — it is no longer
strongly **data-limited** (adding the remaining 20% of already-available
training data barely moved test AUC), and there's no evidence it's
**capacity-limited** either, since training AUC is already at the maximum
possible value (1.0) and cannon go any higher. In other words, the flat
region at 80–100% training data implies that further gains at this point
would more likely come from better or additional *features* (or a genuinely
larger, independently-collected dataset with new information) rather than
from simply reusing more of the rows already on hand.

## 11. Model serialization (Task 8)

The tuned pipeline (`best_pipeline` = `GridSearchCV.best_estimator_`, i.e.
`SimpleImputer → StandardScaler → RandomForestClassifier` with the best
found hyperparameters) is saved with:
```python
joblib.dump(best_pipeline, "best_model.pkl")
```
`best_model.pkl` (~2.6 MB) is committed to this repository.

**Reload-and-predict** (`reload_predict.py`):
```python
import joblib
import pandas as pd

loaded_pipeline = joblib.load("best_model.pkl")
hand_crafted_rows = pd.DataFrame([...])[FEATURE_NAMES]  # two rows, see file
predictions = loaded_pipeline.predict(hand_crafted_rows)
probabilities = loaded_pipeline.predict_proba(hand_crafted_rows)[:, 1]
print(predictions, probabilities)
```
Output when run:
```
Predictions (1 = high earner, 0 = not): [1 0]
Probabilities P(high_earner): [0.875 0.005]
```
Row 1 (a senior, 20-year-experience Engineering employee) is confidently
predicted as a high earner (87.5%); row 2 (a junior, 1-year Support
employee) is confidently predicted as not a high earner (0.5%) — both
predictions are directionally sensible given the feature importances found
above.

## 12. Summary comparison table & final recommendation (Task 9)

| Model | CV mean AUC | CV std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression (Part 2, C=1.0) | 0.9847 | 0.0098 | 0.9946 |
| Decision Tree (max_depth=5) | 0.9476 | 0.0162 | — (accuracy only, see Task 2/3) |
| Random Forest (Task 4, max_depth=10) | 0.9927 | 0.0039 | 0.9927 |
| Gradient Boosting (Task 4a) | 0.9923 | 0.0123 | 0.9969 |
| **Tuned Random Forest (GridSearchCV, Task 6)** | **0.9939** | *(best-fold CV score)* | 0.9934 (at 100% data, Task 7) |

**Recommendation: the tuned Random Forest pipeline from GridSearchCV**
(`max_depth=None, min_samples_leaf=1, n_estimators=200`). It achieves the
highest cross-validated mean AUC (0.9939) of all models tested, and Random
Forest in general showed the lowest cross-validation standard deviation
among the tree-based models (0.0039 for the Task-4 forest), indicating the
most consistent performance across different data folds — an important
property for a model going into production, where reliability across
unseen future data matters as much as peak accuracy. While plain Logistic
Regression is simpler, faster, and fully interpretable (and its test-set
AUC of 0.9946 is competitive), the Random Forest additionally captures any
non-linear or interaction effects among department, experience, and age
without needing them to be manually specified, and the feature-ablation
result (Task 4b) shows it can be trimmed to 10 features with no
performance cost, keeping it practical to deploy. Gradient Boosting is a
close contender (highest single test-set AUC, 0.9969) and would be worth
a second round of tuning if squeezing out the last fraction of a percent
matters, but its higher cross-validation variance (0.0123 vs 0.0039) makes
the Random Forest the safer default recommendation today.

## 13. Files in this repository

```
part3/
├── README.md
├── advanced_modeling.py   # all ensemble/tuning/serialization code (Tasks 1-9)
├── reload_predict.py       # standalone reload-and-predict demonstration
├── cleaned_data.csv         # input data (copied from Part 1's output)
├── best_model.pkl           # serialized tuned Random Forest pipeline
├── console_output.txt       # full run log
└── plots/
    └── learning_curve.png
```
