# CS5062 Machine Learning – Assessment 1

University of Aberdeen, MSc Artificial Intelligence. This repository holds the work for **Assessment 1**.

| Task | Topic | Status |
|---|---|---|
| **Task 1** | Regression: predicting and explaining the level of cancer antigen (OLS, Ridge, Lasso) | Done |
| **Task 2** | Classification: blood cell images (CNN + one other classifier) | Not started |

---

## Task 1 – Regression

**Goal.** Using 97 patient records, (1) *predict* `levelCancerAntigen` from clinical measurements and (2) *infer* which measurements influence it most.

**Approach**

- Models: least squares (OLS), Ridge and Lasso regression.
- The penalty strength of Ridge and Lasso is chosen with **k-fold cross-validation written from scratch** (repeated 10-fold). No `RidgeCV`, `LassoCV`, `GridSearchCV`, `KFold`, `cross_val_score` or `train_test_split` are used, as the brief requires.
- The three models are compared with **nested cross-validation**, so Ridge and Lasso are not favoured by having their penalty tuned on the same folds used to score them.
- Features are standardised **inside every fold** (using only that fold's training data) to avoid data leakage.
- The provided split is used: 67 training patients (`train == 'T'`) for all fitting and model selection, and 30 test patients (`'F'`) only for the final evaluation.
- Inference uses standardised coefficients, regularisation paths, a hand-written bootstrap (1000 resamples) and OLS t-tests computed from first principles.

**Main results** (`SEED = 42`)

| | OLS | Ridge | Lasso |
|---|---|---|---|
| Best penalty (α) | – | 3.455 | 0.01252 |
| Nested-CV MSE | 0.601 | 0.601 | 0.608 |
| Test MSE | 0.521 | 0.499 | 0.495 |
| Test R² | 0.503 | 0.525 | 0.529 |

Predicting the training mean gives a test MSE of 1.057. The three models are statistically tied on prediction; **Ridge** is selected because it has the lowest nested-CV error. For inference, `logCancerVol` is by far the strongest predictor, followed by `svi`, `logCancerWeight`, `gleasonS45` and `logBenighHP`; `logCP`, `age` and `gleasonScore` add no reliable information once the others are known. Full discussion is in the notebook.

---

## Project structure

```
.
├── README.md                          this file
├── Task1_Regression.ipynb             Task 1 notebook (submit this; outputs included)
├── Task1_Regression.py                same code as one commented script
├── Task1_Implementation_Guide.md      step-by-step explanation of how it is built
├── Task1_RegressionTask_CancerData.txt   data file from MyAberdeen (you add this; do NOT submit it)
└── figures/                           created automatically when you run the script
```

## Requirements

- Python 3.9 or newer
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn (and `ipykernel` for the notebook)

## Setup (Windows PowerShell, from the project folder)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn matplotlib seaborn ipykernel
```

On macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn ipykernel
```

Copy `Task1_RegressionTask_CancerData.txt` into the project folder. If it lives elsewhere, edit `DATA_PATH` near the top of the code.

## Running

**Script**

```powershell
.\.venv\Scripts\python.exe Task1_Regression.py
```

Takes about a minute. Figures open one at a time (close each window to continue) and are also saved to `figures/`.

**Notebook (VS Code)**

1. Open `Task1_Regression.ipynb`.
2. Click **Select Kernel** (top right) and choose the `.venv` environment.
3. Click **Run All**.

A successful run prints `Ridge: best alpha = 3.455`, `Lasso: best alpha = 0.01252` and `Model selected by nested cross-validation: Ridge`. Small last-digit differences are normal with other library versions.

## Reproducibility

Every random choice (fold assignment, bootstrap) uses `numpy.random.default_rng` seeded from `SEED = 42`, so repeated runs give identical results.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Cannot find …CancerData.txt` | Put the data file in the project folder or set `DATA_PATH`. |
| `ModuleNotFoundError` | Install the packages into the same `.venv` you are running, using the commands above. |
| PowerShell cannot run a `/d:/...python.exe` path | Use the relative form `.\.venv\Scripts\python.exe` shown above. |
| Notebook has no kernel | Install `ipykernel` and select the `.venv` kernel. |

## Submission checklist (from the assessment brief)

- One zip named `CS5062_Assessment1_<Surname>_<FirstName>_<StudentID>.zip` containing exactly three files: the PDF report (Tasks 1 and 2), the Task 1 notebook and the Task 2 notebook.
- **No datasets** in the zip.
- Notebooks saved with their outputs visible.
- AI assistance and any adapted code declared in both the notebook and the report.

## Sources and AI statement

Code was written for this assignment with AI assistance (permitted by the brief). No open-source or tutorial code was copied. Libraries used: NumPy, pandas, SciPy (t-distribution), scikit-learn (`Ridge` and `Lasso` estimators only), matplotlib and seaborn.
