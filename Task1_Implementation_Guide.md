# Task 1 (Regression) – Implementation Guide

You have three files for Task 1:

| File | What it is |
|---|---|
| `Task1_Regression.ipynb` | The notebook to submit (outputs already embedded) |
| `Task1_Regression.py` | The **same code as one commented script** (use this to read or run the full source) |
| `Task1_Implementation_Guide.md` | This guide |

---

## 1. Set up in VS Code (one time)

1. **Install Python 3.9+** (check with `python --version` in a terminal).
2. **Install two VS Code extensions:** *Python* and *Jupyter* (both by Microsoft).
3. **Make a project folder** and put these in it:
   ```
   task1/
   ├── Task1_RegressionTask_CancerData.txt     <- from MyAberdeen
   ├── Task1_Regression.ipynb
   └── Task1_Regression.py
   ```
   Open the folder with *File → Open Folder…*.
4. **Create an environment and install the libraries** (VS Code terminal: *Terminal → New Terminal*):
   ```bash
   python -m venv .venv
   # Windows:        .venv\Scripts\activate
   # macOS / Linux:  source .venv/bin/activate
   pip install numpy pandas scipy scikit-learn matplotlib seaborn ipykernel
   ```
5. **Select the interpreter:** press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → *Python: Select Interpreter* → choose the one inside `.venv`.

## 2. Run it

**Notebook:** open `Task1_Regression.ipynb` → top right *Select Kernel* → choose `.venv` → click **Run All**. (≈ 1 minute. The nested cross-validation cell is the slow one.)

**Script:** in the terminal run `python Task1_Regression.py`. Figures open one at a time (close each window to continue) and are also saved to `./figures/`. In VS Code you can alternatively click *Run Cell* above any `# %%` line to run it like a notebook cell.

If the data file is not next to the code, change `DATA_PATH` near the top.

---

## 3. How it is built, step by step

The code follows the subtasks in the brief. Below, each step says **what to write, why, and what you should see.** The complete code is in `Task1_Regression.py`; the key pieces are shown here so you can rebuild or explain them.

### Step 1 – Imports, seed, configuration
Import NumPy, pandas, SciPy, matplotlib, seaborn and **only** `Ridge` and `Lasso` from scikit-learn. Set `SEED = 42` and use `np.random.default_rng(seed)` everywhere randomness is needed, so every run gives identical results (reproducibility is a marking criterion).

### Step 2 – Import the data *(subtask 1)*
```python
raw = pd.read_csv(DATA_PATH, sep="\t")      # the file is TAB-separated
```
Print `shape`, `dtypes`, missing values and `describe()`. **You should see:** 97 rows × 11 columns, no missing values, 67 `T` and 30 `F` in the `train` column.

### Step 3 – Preprocess and split *(subtask 2)*
```python
df = raw.drop(columns="index").copy()          # index is only a row number
df["train"] = df["train"].str.strip()
train_df = df[df["train"] == "T"]              # 67 patients: ALL fitting and model selection
test_df  = df[df["train"] == "F"]              # 30 patients: touched only at the very end
```
Then explain in the report: dropped `index`; used the given split; no missing values; no outlier removal; no extra transforms (the variables are already logs); **standardise features** because Ridge/Lasso penalise coefficient size and features have very different scales. Standardisation is *not* done here but inside every fold (Step 5) to avoid data leakage.

### Step 4 – Look at the data
Plot the target histogram and the correlation heatmap, and compute VIF (variance inflation factors) with plain NumPy:
```python
r2 = 1 - (Z[:, j] - A @ beta).var() / Z[:, j].var()      # regress feature j on the others
vif = 1 / (1 - r2)
```
**You should see:** correlations such as `logCancerVol`–`logCP` 0.69, and VIF ≤ 3.3 (moderate collinearity, none above 5). This motivates trying penalised models.

### Step 5 – Building blocks: OLS and the leakage-free fit
```python
class OLS:                                       # least squares by hand
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self
    def predict(self, X):
        return self.intercept_ + X @ self.coef_

def fit_predict(make_model, X_fit, y_fit, X_new):
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0)       # statistics from the FITTING data only
    model = make_model().fit((X_fit - mu) / sd, y_fit)
    return model.predict((X_new - mu) / sd), model
```
`make_model` is a function that returns a fresh model (`OLS()`, `Ridge(alpha=a)`, `Lasso(alpha=a)`), so the same code evaluates all three.

### Step 6 – Your own k-fold cross-validation *(required: no built-in CV)*
```python
def kfold_indices(n, k, rng):
    perm = rng.permutation(n)                    # shuffle patient indices
    folds = np.array_split(perm, k)              # k nearly equal folds
    return [(np.setdiff1d(perm, f, assume_unique=True), f) for f in folds]   # (train_idx, val_idx)

def cv_fold_errors(make_model, X, y, folds):
    errs = []
    for tr, va in folds:
        pred, _ = fit_predict(make_model, X[tr], y[tr], X[va])
        errs.append(np.mean((y[va] - pred) ** 2))             # MSE on the held-out fold
    return np.array(errs)
```
Do **not** use `KFold`, `cross_val_score`, `GridSearchCV`, `RidgeCV`, `LassoCV` or `train_test_split`; the brief awards zero for those parts.

### Step 7 – Tune the penalty λ over a grid (repeated CV)
```python
def tune_penalty(model_cls, grid, X, y, k=10, repeats=10, seed=SEED):
    rng = np.random.default_rng(seed)
    fold_sets = [kfold_indices(len(y), k, rng) for _ in range(repeats)]
    errs = np.zeros((repeats, len(grid), k))
    for r, folds in enumerate(fold_sets):
        for g, a in enumerate(grid):
            errs[r, g] = cv_fold_errors(lambda: model_cls(alpha=a), X, y, folds)
    cv_mean = errs.mean(axis=(0, 2))                          # CV-MSE for each alpha
    best = int(np.argmin(cv_mean))
    ...                                                       # also returns the SE and the 1-SE-rule alpha
```
Grids: `Ridge` `np.logspace(-3, 3, 40)`; `Lasso` from `alpha_max/10⁴` up to `alpha_max = max|Zᵀ(y−ȳ)|/n` (the penalty at which every coefficient is exactly zero). Repeating the 10-fold split 10 times removes the dependence on one lucky/unlucky split, which matters with only 67 patients.

### Step 8 – Train the three models *(subtask 3)*
- **OLS:** just fit it (no parameter); record its CV error as the baseline.
- **Ridge:** `ridge_res = tune_penalty(Ridge, RIDGE_GRID, X_train, y_train)`
- **Lasso:** `lasso_res = tune_penalty(Lasso, LASSO_GRID, X_train, y_train)`

Plot CV-MSE against α (log axis) with ±1 SE band, the minimum and the 1-SE-rule value. **You should see:** Ridge best α ≈ **3.46**, Lasso best α ≈ **0.0125**; curves are flat for small α and rise steeply for large α.

### Step 9 – Choose the best of the three models by nested CV
Comparing models on the CV error that already picked their λ favours Ridge/Lasso (optimistic). **Nested CV** fixes this: for every *outer* fold, re-run the λ search on the outer-training patients only (inner 10-fold CV), refit, and score on the untouched outer-validation fold. OLS uses the same outer folds.
```python
inner = tune_penalty(cls, grid, X[tr], y[tr], k=10, repeats=1, seed=...)   # inside each outer fold
pred, _ = fit_predict(make_with(inner["best_alpha"]), X[tr], y[tr], X[va])
```
Choose the model with the lowest nested-CV MSE. **You should see:** OLS 0.601, Ridge 0.601, Lasso 0.608, a statistical tie → **Ridge** selected (as a tie-break, which the report should say honestly).

### Step 10 – Final fit and test evaluation
Refit each model on all 67 training patients with its chosen λ, predict the 30 test patients and report MSE, RMSE, MAE and R², plus the "predict the training mean" baseline (test MSE 1.057). Show a table, a bar chart (train / nested-CV / test error) and predicted-vs-observed scatter plots. **Never use the test set to choose a model or λ.**

### Step 11 – Inference with the chosen model *(subtask 4)*
1. **Standardised coefficients** of the three models side by side (effect of +1 SD in a feature).
2. **Regularisation paths**: fit Ridge/Lasso over the whole grid and plot each coefficient against α; important variables stay large the longest.
3. **Bootstrap** (own code): resample patients with replacement 1000 times, refit the chosen model with λ fixed, take the 2.5 % / 97.5 % percentiles of every coefficient:
   ```python
   idx = rng.integers(0, n, n)                                   # sample with replacement
   boot[b] = make_best().fit(standardise(X[idx]), y[idx]).coef_
   ```
4. **OLS t-tests from first principles** as a reference: `σ² = RSS/(n−p−1)`, `Cov(β) = σ²(AᵀA)⁻¹`, `t = β/SE`, p-value from `scipy.stats.t`.

**You should see:** `logCancerVol` ≈ +0.63 (interval 0.37–0.85) far ahead of `svi`, `logCancerWeight` (≈ +0.29 each), `gleasonS45` (+0.22), `logBenighHP` (+0.20); `logCP`, `age`, `gleasonScore` with intervals that include zero.

### Step 12 – Write the interpretation
Explain with ML principles (bias–variance, over-fitting, shrinkage, collinearity): why the models tie, why `logCP` is negative despite a positive raw correlation (conditional vs. marginal effects), and the limits (n = 67, observational data, linearity).

---

## 4. Expected results (SEED = 42)

| Quantity | Value |
|---|---|
| Training / test patients | 67 / 30 |
| Ridge best α / Lasso best α | 3.455 / 0.01252 |
| Nested-CV MSE: OLS / Ridge / Lasso | 0.6014 / 0.6008 / 0.6078 |
| Test MSE: OLS / Ridge / Lasso | 0.5213 / 0.4990 / 0.4946 |
| Test R²: OLS / Ridge / Lasso | 0.503 / 0.525 / 0.529 |
| Predict-the-mean test MSE | 1.0567 |
| Selected model | Ridge |

Tiny last-digit differences can occur with other library versions; if your numbers differ by more than that, check the troubleshooting list.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Cannot find …txt` | Put the data file next to the code, or edit `DATA_PATH`; check the VS Code terminal is in the project folder (`cd task1`). |
| One giant column / `KeyError: 'train'` | The file is tab-separated: use `sep="\t"`. |
| `ModuleNotFoundError` | The venv isn't active/selected. Re-run *Python: Select Interpreter* and `pip install …` inside the activated venv. |
| Notebook "No kernel" | Install `ipykernel`, then pick the `.venv` kernel top-right. |
| Script shows only one plot and seems stuck | `plt.show()` blocks until you close the window; figures are also saved in `./figures/`. |
| Slow run | The nested-CV cell does ≈ 40 000 small model fits; 1–2 minutes is normal. For a quick test only, change `for _ in range(5)` to `range(2)` in the nested-CV cell (results will change slightly; restore it before submitting). |
| Different numbers | Make sure `SEED = 42` is unchanged and you did not edit the grids, folds or repeats. |

## 6. Submission checklist (from the brief)

- [ ] One **zip** named `CS5062_Assessment1_<Surname>_<FirstName>_<StudentID>.zip` containing **exactly three files**: the PDF report (Tasks 1 & 2), the Task 1 notebook, the Task 2 notebook.
- [ ] **No data sets** inside the zip (remove the `.txt` and the images).
- [ ] Notebook saved **with outputs visible** under the code cells (run *Run All* before saving).
- [ ] Report includes code snippets and clearly readable plots (re-use the figures in `./figures/`), and explains every subtask above (~1000 words for Task 1).
- [ ] AI use and any adapted code **declared in both notebook and report** (the notebook's first cell already contains the statement; review it and make sure it is true for your final version).
