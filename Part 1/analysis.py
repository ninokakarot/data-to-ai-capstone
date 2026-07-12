"""
Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: employee_raw.csv (synthetic HR analytics dataset, see README for
details on why a synthetic dataset was used and how it was generated).

Run with: python3 analysis.py
Outputs: cleaned_data.csv, and PNG plots in ./plots/
All printed output is also captured to console_output.txt for the README.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------------------
# Task 1: Load & inspect
# ---------------------------------------------------------------------------
print("=" * 80)
print("TASK 1: Load and inspect")
print("=" * 80)
df = pd.read_csv("employee_raw.csv")
print("\nFirst 5 rows:\n", df.head())
print("\nDtypes:\n", df.dtypes)
print("\nShape:", df.shape)

# ---------------------------------------------------------------------------
# Task 2: Null value analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: Null value analysis")
print("=" * 80)
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(2)})
print("\nNull count / percentage per column:\n", null_table)

high_null_cols = null_table[null_table["null_pct"] > 20].index.tolist()
print("\nColumns exceeding 20% nulls (NOT median-filled here):", high_null_cols)

# monthly_salary is still text at this point (Task 4 will fix its dtype);
# fill medians only for columns that are already numeric and below 20% nulls.
low_null_numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in high_null_cols and null_table.loc[c, "null_pct"] > 0
]
print("\nNumeric columns below 20% nulls -> filling with median:", low_null_numeric_cols)
for c in low_null_numeric_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med)
    print(f"  {c}: filled {null_counts[c]} nulls with median={med}")

# ---------------------------------------------------------------------------
# Task 3: Duplicate detection and removal
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: Duplicate detection and removal")
print("=" * 80)
n_dupes = df.duplicated().sum()
print("Duplicate rows found:", n_dupes)

null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
print(f"Rows removed: {n_dupes}. New shape: {df.shape}")
null_change = pd.DataFrame({"before_pct": null_pct_before.round(2), "after_pct": null_pct_after.round(2)})
null_change["changed"] = null_change["before_pct"] != null_change["after_pct"]
print("\nNull % before vs after duplicate removal:\n", null_change)

# ---------------------------------------------------------------------------
# Task 4: Data type correction
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: Data type correction")
print("=" * 80)
mem_before = df.memory_usage(deep=True).sum()
print("Memory usage BEFORE conversion (bytes):", mem_before)

# monthly_salary is stored as text like "$4,521.30" -> object dtype (incorrect)
df["monthly_salary"] = (
    df["monthly_salary"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["monthly_salary"] = pd.to_numeric(df["monthly_salary"], errors="coerce")

# Fill any nulls in monthly_salary created after numeric conversion (below 20%)
sal_null_pct = df["monthly_salary"].isnull().mean() * 100
if 0 < sal_null_pct <= 20:
    med_sal = df["monthly_salary"].median()
    df["monthly_salary"] = df["monthly_salary"].fillna(med_sal)
    print(f"monthly_salary converted to numeric; filled remaining nulls with median={med_sal}")

# Convert repetitive string columns to category dtype
for c in ["department", "education_level", "region", "remote_work"]:
    df[c] = df[c].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print("Memory usage AFTER conversion (bytes):", mem_after)
print(f"Memory reduction: {mem_before - mem_after} bytes "
      f"({(1 - mem_after/mem_before)*100:.1f}%)")
print("\nDtypes after correction:\n", df.dtypes)

# ---------------------------------------------------------------------------
# Task 5: Descriptive statistics and skewness
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: Descriptive statistics and skewness")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "employee_id"]
print("\nDescribe (numeric columns):\n", df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("\nSkewness per numeric column (sorted by |skew|):\n", skew_vals)
most_skewed_col = skew_vals.index[0]
print(f"\nMost skewed column: {most_skewed_col} (skew={skew_vals.iloc[0]:.3f})")

# ---------------------------------------------------------------------------
# Task 6: Outlier detection with IQR (>=2 numeric columns)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: Outlier detection with IQR")
print("=" * 80)
iqr_cols = ["monthly_salary", "age"]
iqr_report = {}
for c in iqr_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = df[(df[c] < lower) | (df[c] > upper)].shape[0]
    iqr_report[c] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_out)
    print(f"\n{c}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
          f"bounds=({lower:.2f}, {upper:.2f}), outliers={n_out}")

# ---------------------------------------------------------------------------
# Task 7: Visualizations
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 7: Visualizations")
print("=" * 80)

# 7.1 Line plot
plt.figure(figsize=(10, 5))
plt.plot(df.reset_index()["index"], df["monthly_salary"].values)
plt.title("Monthly Salary by Row Index")
plt.xlabel("Row Index")
plt.ylabel("Monthly Salary ($)")
plt.tight_layout()
plt.savefig("plots/01_line_monthly_salary.png", dpi=150)
plt.close()
print("Saved plots/01_line_monthly_salary.png")

# 7.2 Bar chart: mean monthly_salary by department
plt.figure(figsize=(9, 5))
dept_means = df.groupby("department", observed=True)["monthly_salary"].mean().sort_values(ascending=False)
dept_means.plot.bar(color=sns.color_palette("viridis", len(dept_means)))
plt.title("Mean Monthly Salary by Department")
plt.xlabel("Department")
plt.ylabel("Mean Monthly Salary ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/02_bar_mean_salary_by_department.png", dpi=150)
plt.close()
print("Saved plots/02_bar_mean_salary_by_department.png")

# 7.3 Histogram of most skewed column
plt.figure(figsize=(8, 5))
sns.histplot(df[most_skewed_col], bins=20, kde=True)
plt.title(f"Distribution of {most_skewed_col} (skew={skew_vals.iloc[0]:.2f})")
plt.xlabel(most_skewed_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/03_histogram_most_skewed.png", dpi=150)
plt.close()
print(f"Saved plots/03_histogram_most_skewed.png (column: {most_skewed_col})")

# 7.4 Scatter plot: years_experience vs monthly_salary
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="years_experience", y="monthly_salary", alpha=0.6)
plt.title("Years of Experience vs Monthly Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Monthly Salary ($)")
plt.tight_layout()
plt.savefig("plots/04_scatter_experience_vs_salary.png", dpi=150)
plt.close()
print("Saved plots/04_scatter_experience_vs_salary.png")

# 7.5 Box plot: performance_score split by department
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="department", y="performance_score")
plt.title("Performance Score Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Performance Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/05_boxplot_performance_by_department.png", dpi=150)
plt.close()
print("Saved plots/05_boxplot_performance_by_department.png")

# ---------------------------------------------------------------------------
# Task 8 (heatmap portion): Correlation heat map (Pearson)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8: Correlation heat map (Pearson)")
print("=" * 80)
corr_matrix = df[numeric_cols].corr()
print("\nPearson correlation matrix:\n", corr_matrix.round(3))

# Find highest |correlation| pair (excluding self-correlation)
corr_abs_arr = np.array(corr_matrix.abs().values, dtype=float, copy=True)
np.fill_diagonal(corr_abs_arr, 0)
corr_abs = pd.DataFrame(corr_abs_arr, index=corr_matrix.index, columns=corr_matrix.columns)
max_pair_idx = np.unravel_index(np.argmax(corr_abs.values), corr_abs.shape)
pair_a, pair_b = corr_abs.index[max_pair_idx[0]], corr_abs.columns[max_pair_idx[1]]
print(f"\nHighest |correlation| pair: {pair_a} & {pair_b} "
      f"(r={corr_matrix.loc[pair_a, pair_b]:.3f})")

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heat Map (Pearson) - Numeric Columns")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved plots/06_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# Task 8a: Imputation strategy comparison (mean vs median) for top-2 skew cols
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8a: Imputation strategy comparison (mean vs median)")
print("=" * 80)
top2_skew_cols = skew_vals.index[:2].tolist()
print("Two highest |skew| columns:", top2_skew_cols)

mean_median_table = pd.DataFrame({
    "mean": [df[c].mean() for c in top2_skew_cols],
    "median": [df[c].median() for c in top2_skew_cols],
}, index=top2_skew_cols)
print("\nMean vs Median (before imputing remaining nulls in these columns):\n",
      mean_median_table.round(3))

for c in top2_skew_cols:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

print("\nNulls remaining in these columns after imputation:\n",
      df[top2_skew_cols].isnull().sum())

# ---------------------------------------------------------------------------
# Task 8b: Spearman rank correlation vs Pearson
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8b: Spearman vs Pearson correlation")
print("=" * 80)
spearman_matrix = df[numeric_cols].corr(method="spearman")
print("\nSpearman correlation matrix:\n", spearman_matrix.round(3))
print("\nPearson correlation matrix (repeated for comparison):\n", corr_matrix.round(3))

diff_arr = np.array((spearman_matrix - corr_matrix).abs().values, dtype=float, copy=True)
np.fill_diagonal(diff_arr, 0)
diff_matrix = pd.DataFrame(diff_arr, index=corr_matrix.index, columns=corr_matrix.columns)

# Extract unique pairs sorted by difference, descending
pairs = []
cols = diff_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((cols[i], cols[j], diff_matrix.iloc[i, j],
                      spearman_matrix.iloc[i, j], corr_matrix.iloc[i, j]))
pairs_df = pd.DataFrame(pairs, columns=["col_a", "col_b", "abs_diff", "spearman", "pearson"])
pairs_df = pairs_df.sort_values("abs_diff", ascending=False).reset_index(drop=True)
print("\nTop 3 pairs by |Spearman - Pearson|:\n", pairs_df.head(3))

# ---------------------------------------------------------------------------
# Task 8c: Grouped aggregation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8c: Grouped aggregation")
print("=" * 80)
group_col, agg_col = "department", "monthly_salary"
grouped = df.groupby(group_col, observed=True)[agg_col].agg(["mean", "std", "count"])
print(f"\nGrouped aggregation of {agg_col} by {group_col}:\n", grouped.round(2))

highest_mean_group = grouped["mean"].idxmax()
highest_std_group = grouped["std"].idxmax()
mean_ratio = grouped["mean"].max() / grouped["mean"].min()
print(f"\nGroup with highest mean: {highest_mean_group} ({grouped['mean'].max():.2f})")
print(f"Group with highest std: {highest_std_group} ({grouped['std'].max():.2f})")
print(f"Ratio of highest group mean to lowest group mean: {mean_ratio:.2f}")

# ---------------------------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Saving cleaned_data.csv")
print("=" * 80)
df.to_csv("cleaned_data.csv", index=False)
print("Final cleaned shape:", df.shape)
print("Remaining nulls per column:\n", df.isnull().sum())
