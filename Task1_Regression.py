"""
CS5062 Machine Learning - Assessment 1, Task 1: Regression
Predict and explain `levelCancerAntigen` with OLS, Ridge and Lasso.
Cross-validation (k-fold, repeated, nested) and the bootstrap are implemented by hand.

HOW TO RUN
    1. Put this file in the same folder as Task1_RegressionTask_CancerData.txt
    2. pip install numpy pandas scipy scikit-learn matplotlib seaborn
    3. python Task1_Regression.py          (about 1 minute; figures are saved to ./figures)
       In VS Code you can also run it cell-by-cell: every "# %%" line starts an interactive cell.

SOURCES / AI STATEMENT
    Written for this assignment with AI assistance (permitted by the brief). No open-source or
    tutorial code was copied. Only NumPy, pandas, SciPy (t-distribution), scikit-learn (Ridge and
    Lasso estimators) and matplotlib/seaborn (plots) are used as libraries.
"""


# ==============================================================================
# CS5062 Machine Learning – Assessment 1
# ==============================================================================

# %%
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

SEED = 42                                              # one seed controls every random choice below
DATA_PATH = Path("Task1_RegressionTask_CancerData.txt")  # <- change if the file lives elsewhere

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
def save_and_show(name):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"scikit-learn {sklearn.__version__}")

# ==============================================================================
# 1. Data import (3 marks)
# ==============================================================================

# %%
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot find {DATA_PATH.resolve()}. Put the data file next to this script "
                            "or edit DATA_PATH in the first code cell.")

raw = pd.read_csv(DATA_PATH, sep="\t")      # tab separated
print("Shape (rows, columns):", raw.shape)
print(raw.head())

# %%
print("Data types:\n", raw.dtypes.to_string(), "\n")
print("Missing values per column:\n", raw.isna().sum().to_string(), "\n")
print("Train/test label counts:\n", raw["train"].str.strip().value_counts().to_string())
print(raw.drop(columns="index").describe().T.round(2))

# ==============================================================================
# 2. Data preprocessing (3 marks)
# ==============================================================================

# %%
FEATURES = ["logCancerVol", "logCancerWeight", "age", "logBenighHP", "svi", "logCP", "gleasonScore", "gleasonS45"]
TARGET = "levelCancerAntigen"

df = raw.drop(columns="index").copy()
df["train"] = df["train"].str.strip()

train_df = df[df["train"] == "T"].reset_index(drop=True)
test_df  = df[df["train"] == "F"].reset_index(drop=True)

X_train, y_train = train_df[FEATURES].to_numpy(float), train_df[TARGET].to_numpy(float)
X_test,  y_test  = test_df[FEATURES].to_numpy(float),  test_df[TARGET].to_numpy(float)
print(f"Training set: X {X_train.shape}, y {y_train.shape}")
print(f"Test set:     X {X_test.shape}, y {y_test.shape}")

# %%
# Quick look at the distributions and the discrete / floor-valued variables
print("svi counts:", train_df["svi"].value_counts().to_dict())
print("gleasonScore counts:", train_df["gleasonScore"].value_counts().sort_index().to_dict())
print(f"logCP at its minimum ({train_df['logCP'].min():.3f}): {(train_df['logCP'] == train_df['logCP'].min()).mean():.0%} of training patients")
print(f"gleasonS45 == 0: {(train_df['gleasonS45'] == 0).mean():.0%} of training patients")
print(f"logBenighHP at its minimum: {(train_df['logBenighHP'] == train_df['logBenighHP'].min()).mean():.0%} of training patients")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].hist(y_train, bins=15, edgecolor="white", color="#4C72B0")
axes[0].set(title="Target distribution (training set)", xlabel=TARGET, ylabel="patients")

corr = train_df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool), k=1), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
axes[1].set(title="Correlation matrix (training set)")
axes[1].grid(False)
plt.tight_layout(); save_and_show("fig01")

# %%
# Multicollinearity check: variance inflation factor, VIF_j = 1 / (1 - R^2_j),
# where R^2_j comes from regressing feature j on all the other features (training data only).
Z = (X_train - X_train.mean(0)) / X_train.std(0)
vif = []
for j in range(Z.shape[1]):
    A = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
    beta = np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
    r2 = 1 - (Z[:, j] - A @ beta).var() / Z[:, j].var()
    vif.append(1 / (1 - r2))
print(pd.Series(vif, index=FEATURES, name="VIF").round(2).to_frame())

# ==============================================================================
# 3. Model training and model selection (14 marks)
# ==============================================================================

# %%
class OLS:
    # Ordinary least squares with an intercept, solved by least squares (no penalty).
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


def fit_predict(make_model, X_fit, y_fit, X_new):
    # Standardise using ONLY the fitting data (no leakage), fit, and predict for X_new.
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0)
    model = make_model().fit((X_fit - mu) / sd, y_fit)
    return model.predict((X_new - mu) / sd), model


def kfold_indices(n, k, rng):
    # Own k-fold splitter: random permutation cut into k (nearly) equal folds -> list of (train_idx, val_idx).
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.setdiff1d(perm, f, assume_unique=True), f) for f in folds]


def cv_fold_errors(make_model, X, y, folds):
    # MSE on each held-out fold.
    errs = []
    for tr, va in folds:
        pred, _ = fit_predict(make_model, X[tr], y[tr], X[va])
        errs.append(np.mean((y[va] - pred) ** 2))
    return np.array(errs)


def tune_penalty(model_cls, grid, X, y, k=10, repeats=10, seed=SEED):
    # Repeated k-fold CV over a grid of penalty values (alpha).
    # Returns the mean CV-MSE curve, its approximate standard error, the best alpha and the 1-SE-rule alpha.
    rng = np.random.default_rng(seed)
    fold_sets = [kfold_indices(len(y), k, rng) for _ in range(repeats)]
    errs = np.zeros((repeats, len(grid), k))
    for r, folds in enumerate(fold_sets):
        for g, a in enumerate(grid):
            errs[r, g] = cv_fold_errors(lambda: model_cls(alpha=a), X, y, folds)
    cv_mean = errs.mean(axis=(0, 2))
    cv_se = (errs.std(axis=2, ddof=1) / np.sqrt(k)).mean(axis=0)     # SE of a single k-fold run, averaged over repeats
    best = int(np.argmin(cv_mean))
    within = np.where(cv_mean <= cv_mean[best] + cv_se[best])[0]     # 1-SE rule: simplest model within 1 SE of the best
    one_se = int(within.max())                                       # grid is increasing -> largest alpha = most regularised
    return dict(grid=np.asarray(grid), cv_mean=cv_mean, cv_se=cv_se, best_idx=best, best_alpha=grid[best],
                one_se_idx=one_se, one_se_alpha=grid[one_se])


# Penalty grids. Lasso: from alpha_max (all coefficients exactly zero) down four orders of magnitude.
Zs = (X_train - X_train.mean(0)) / X_train.std(0)
alpha_max = np.max(np.abs(Zs.T @ (y_train - y_train.mean()))) / len(y_train)
RIDGE_GRID = np.logspace(-3, 3, 40)
LASSO_GRID = np.logspace(np.log10(alpha_max) - 4, np.log10(alpha_max), 40)

make_ols   = lambda: OLS()
make_ridge = lambda a: (lambda: Ridge(alpha=a))
make_lasso = lambda a: (lambda: Lasso(alpha=a, max_iter=100_000))
print(f"Ridge grid: {RIDGE_GRID[0]:.3g} … {RIDGE_GRID[-1]:.3g} | Lasso grid: {LASSO_GRID[0]:.3g} … {LASSO_GRID[-1]:.3g} (40 values each, log-spaced)")

# ==============================================================================
# 3.2 Least squares
# ==============================================================================

# %%
rng = np.random.default_rng(SEED)
ols_folds = [kfold_indices(len(y_train), 10, rng) for _ in range(10)]
ols_errs = np.array([cv_fold_errors(make_ols, X_train, y_train, f) for f in ols_folds])   # (repeats, folds)
print(f"OLS   10-fold CV MSE (10 repeats): {ols_errs.mean():.4f}   (RMSE {np.sqrt(ols_errs.mean()):.4f})")

# ==============================================================================
# 3.3 Ridge regression – choosing λ (called `alpha` in scikit-learn) by cross-validation
# ==============================================================================

# %%
ridge_res = tune_penalty(Ridge, RIDGE_GRID, X_train, y_train)
print(f"Ridge: best alpha = {ridge_res['best_alpha']:.4g}  (CV-MSE {ridge_res['cv_mean'][ridge_res['best_idx']]:.4f} ± {ridge_res['cv_se'][ridge_res['best_idx']]:.4f})")
print(f"Ridge: 1-SE-rule alpha = {ridge_res['one_se_alpha']:.4g}")

# ==============================================================================
# 3.4 Lasso regression – choosing λ by cross-validation
# ==============================================================================

# %%
lasso_res = tune_penalty(Lasso, LASSO_GRID, X_train, y_train)
print(f"Lasso: best alpha = {lasso_res['best_alpha']:.4g}  (CV-MSE {lasso_res['cv_mean'][lasso_res['best_idx']]:.4f} ± {lasso_res['cv_se'][lasso_res['best_idx']]:.4f})")
print(f"Lasso: 1-SE-rule alpha = {lasso_res['one_se_alpha']:.4g}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, res, name in zip(axes, [ridge_res, lasso_res], ["Ridge", "Lasso"]):
    g, m, s = res["grid"], res["cv_mean"], res["cv_se"]
    ax.plot(g, m, "o-", ms=3, color="#4C72B0", label="mean CV-MSE (10-fold × 10 repeats)")
    ax.fill_between(g, m - s, m + s, alpha=0.2, color="#4C72B0", label="± 1 SE")
    ax.axvline(res["best_alpha"], color="#C44E52", ls="--", label=f"minimum: α = {res['best_alpha']:.3g}")
    ax.axvline(res["one_se_alpha"], color="#55A868", ls=":", label=f"1-SE rule: α = {res['one_se_alpha']:.3g}")
    ax.axhline(ols_errs.mean(), color="grey", ls="-.", lw=1, label=f"OLS: {ols_errs.mean():.3f}")
    ax.set(xscale="log", xlabel="penalty strength α (log scale)", title=f"{name}: cross-validation curve",
           ylim=((m - s).min() - 0.03, m.min() + 0.45))      # zoom in: the steep rise at large α would hide the minimum
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("CV mean squared error")
plt.tight_layout(); save_and_show("fig02")

# ==============================================================================
# 3.5 Choosing the best of the three models
# ==============================================================================

# %%
def nested_cv(kind, outer_sets, X, y, inner_k=10, seed=SEED):
    # Nested CV. kind in {"ols", "ridge", "lasso"}. Returns MSE per outer fold, shape (repeats, outer_k).
    out = np.zeros((len(outer_sets), len(outer_sets[0])))
    for r, folds in enumerate(outer_sets):
        for f, (tr, va) in enumerate(folds):
            if kind == "ols":
                make = make_ols
            else:
                cls, grid = (Ridge, RIDGE_GRID) if kind == "ridge" else (Lasso, LASSO_GRID)
                inner = tune_penalty(cls, grid, X[tr], y[tr], k=inner_k, repeats=1, seed=seed + 1000 * r + f)
                make = make_ridge(inner["best_alpha"]) if kind == "ridge" else make_lasso(inner["best_alpha"])
            pred, _ = fit_predict(make, X[tr], y[tr], X[va])
            out[r, f] = np.mean((y[va] - pred) ** 2)
    return out

rng = np.random.default_rng(SEED + 7)
outer_sets = [kfold_indices(len(y_train), 10, rng) for _ in range(5)]      # 5 repeats of outer 10-fold CV

nested = {k: nested_cv(k, outer_sets, X_train, y_train) for k in ["ols", "ridge", "lasso"]}

rows = []
for name, e in nested.items():
    d = e - nested["ols"]                                        # paired difference to OLS (same outer folds)
    rows.append({"model": name.upper() if name == "ols" else name.capitalize(),
                 "nested-CV MSE": e.mean(),
                 "SE": (e.std(axis=1, ddof=1) / np.sqrt(e.shape[1])).mean(),
                 "nested-CV RMSE": np.sqrt(e.mean()),
                 "MSE − OLS MSE": d.mean(),
                 "SE of difference": (d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])).mean()})
cv_table = pd.DataFrame(rows).set_index("model")
print("Nested CV (5 repeats × 10 outer folds; penalty re-tuned inside every outer fold):")
print(cv_table.round(4))

# %%
# Fit each model on the FULL training set (standardised with training statistics) using the CV-chosen penalty.
final_models = {
    "OLS":   make_ols,
    "Ridge": make_ridge(ridge_res["best_alpha"]),
    "Lasso": make_lasso(lasso_res["best_alpha"]),
}

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(y_true - y_pred)),
            "R²": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)}

test_rows, fitted, test_pred = [], {}, {}
for name, make in final_models.items():
    pred, model = fit_predict(make, X_train, y_train, X_test)
    fitted[name], test_pred[name] = model, pred
    tr_pred, _ = fit_predict(make, X_train, y_train, X_train)
    m = regression_metrics(y_test, pred)
    test_rows.append({"model": name, "train MSE": np.mean((y_train - tr_pred) ** 2),
                      "nested-CV MSE": cv_table.loc[name.upper() if name == "OLS" else name, "nested-CV MSE"],
                      "test MSE": m["MSE"], "test RMSE": m["RMSE"], "test MAE": m["MAE"], "test R²": m["R²"]})

baseline_mse = np.mean((y_test - y_train.mean()) ** 2)
results = pd.DataFrame(test_rows).set_index("model")
print(f"Reference: predicting the training mean for every test patient gives test MSE = {baseline_mse:.4f}\n")
best_name = results["nested-CV MSE"].idxmin()
print(f"Model selected by nested cross-validation: {best_name}")
print(results.round(4))

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})

# (a) error comparison
ax = axes[0]
names = list(results.index)
x = np.arange(len(names)); w = 0.27
se = cv_table["SE"].to_numpy()
ax.bar(x - w, results["train MSE"], w, label="training MSE", color="#8C8C8C")
ax.bar(x,      results["nested-CV MSE"], w, yerr=se, capsize=3, label="nested-CV MSE (±SE)", color="#4C72B0")
ax.bar(x + w,  results["test MSE"], w, label="test MSE", color="#C44E52")
ax.axhline(baseline_mse, color="k", ls=":", lw=1, label="predict-the-mean (test)")
ax.set(xticks=x, xticklabels=names, ylabel="mean squared error", title="Error of the three models")
ax.set_ylim(0, 1.45); ax.legend(fontsize=7, loc="upper left", ncol=2)

# (b–d) predicted vs observed on the test set
lims = [min(y_test.min(), *[p.min() for p in test_pred.values()]) - 0.2,
        max(y_test.max(), *[p.max() for p in test_pred.values()]) + 0.2]
for ax, name in zip(axes[1:], names):
    ax.scatter(y_test, test_pred[name], s=28, color="#4C72B0", edgecolor="white")
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlim=lims, ylim=lims, xlabel="observed levelCancerAntigen", ylabel="predicted",
           title=f"{name}: test MSE {results.loc[name, 'test MSE']:.3f}")
plt.tight_layout(); save_and_show("fig03")

# ==============================================================================
# 4. Inference – which clinical measures influence the cancer antigen level? (5 marks)
# ==============================================================================

# %%
mu, sd = X_train.mean(0), X_train.std(0)
coef_table = pd.DataFrame({name: fitted[name].coef_ for name in fitted}, index=FEATURES)
coef_table["|best model|"] = coef_table[best_name].abs()
coef_table = coef_table.sort_values("|best model|", ascending=False)
print(f"Standardised coefficients (effect of +1 SD in the feature). Rows sorted by |{best_name} coefficient|:")
print(coef_table.drop(columns="|best model|").round(3))

# %%
# Regularisation paths: coefficients of Ridge and Lasso (fitted on the full training set) over the penalty grid.
def path(cls, grid):
    return np.array([cls(alpha=a, **({"max_iter": 100_000} if cls is Lasso else {})).fit(Zs, y_train).coef_ for a in grid])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
palette = sns.color_palette("tab10", len(FEATURES))
for ax, (cls, grid, res, name) in zip(axes, [(Ridge, RIDGE_GRID, ridge_res, "Ridge"), (Lasso, LASSO_GRID, lasso_res, "Lasso")]):
    P = path(cls, grid)
    for j, f in enumerate(FEATURES):
        ax.plot(grid, P[:, j], label=f, color=palette[j], lw=2)
    ax.axvline(res["best_alpha"], color="k", ls="--", lw=1, label="CV-chosen α")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xscale="log", xlabel="penalty strength α (log scale)", title=f"{name} coefficient path")
axes[0].set_ylabel("standardised coefficient")
axes[1].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.38, 1.0))
plt.tight_layout(); save_and_show("fig04")

# %%
# Bootstrap (own implementation): resample the training patients with replacement, refit the selected model
# with the penalty fixed at its CV-chosen value (standardising inside each resample), and record the coefficients.
B = 1000
rng = np.random.default_rng(SEED + 99)
make_best = final_models[best_name]
boot = np.zeros((B, len(FEATURES)))
for b in range(B):
    idx = rng.integers(0, len(y_train), len(y_train))
    m_b, s_b = X_train[idx].mean(0), X_train[idx].std(0)
    s_b[s_b == 0] = 1.0                                         # guard against a constant column in a resample
    boot[b] = make_best().fit((X_train[idx] - m_b) / s_b, y_train[idx]).coef_

ci = pd.DataFrame({
    "coef": fitted[best_name].coef_,
    "boot 2.5%": np.percentile(boot, 2.5, axis=0),
    "boot 97.5%": np.percentile(boot, 97.5, axis=0),
    "% resamples with same sign": (np.sign(boot) == np.sign(fitted[best_name].coef_)).mean(axis=0) * 100,
}, index=FEATURES)
ci["95% CI excludes 0"] = (ci["boot 2.5%"] > 0) | (ci["boot 97.5%"] < 0)
ci = ci.reindex(coef_table.index)
print(f"{best_name}: bootstrap 95% percentile intervals for the standardised coefficients ({B} resamples)")
print(ci.round(3))

# %%
# Reference: classical OLS inference (t-tests) on the standardised training data, computed from first principles.
A = np.column_stack([np.ones(len(Zs)), Zs])
beta = np.linalg.lstsq(A, y_train, rcond=None)[0]
dof = len(y_train) - A.shape[1]
sigma2 = np.sum((y_train - A @ beta) ** 2) / dof
se_beta = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
t_stat = beta / se_beta
p_val = 2 * stats.t.sf(np.abs(t_stat), dof)
tcrit = stats.t.ppf(0.975, dof)
ols_inf = pd.DataFrame({"coef": beta, "std. error": se_beta, "t": t_stat, "p-value": p_val,
                        "95% CI low": beta - tcrit * se_beta, "95% CI high": beta + tcrit * se_beta},
                       index=["(intercept)"] + FEATURES)
ols_inf = pd.concat([ols_inf.iloc[:1], ols_inf.iloc[1:].reindex(coef_table.index)])
print(f"OLS t-tests on standardised features (residual degrees of freedom = {dof})")
print(ols_inf.round(3))

# %%
# Final ranking figure: coefficients with bootstrap intervals for the selected model.
order = ci.index[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
y_pos = np.arange(len(order))
ax.errorbar(ci.loc[order, "coef"], y_pos,
            xerr=[ci.loc[order, "coef"] - ci.loc[order, "boot 2.5%"], ci.loc[order, "boot 97.5%"] - ci.loc[order, "coef"]],
            fmt="o", color="#4C72B0", ecolor="#8FA8CC", capsize=3, lw=2)
ax.axvline(0, color="k", lw=1)
ax.set(yticks=y_pos, yticklabels=order, xlabel="standardised coefficient (effect of +1 SD)",
       title=f"{best_name}: coefficients with 95% bootstrap intervals")
plt.tight_layout(); save_and_show("fig05")
