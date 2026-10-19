"""
Semantic (clinically informed) feature engineering for CVD prediction
=====================================================================
Full experimental pipeline for the paper:
  "A Semantic Feature Engineering Framework for Cardiovascular Disease
   Prediction: An Empirical Study on Representation and Model Performance"

What it runs
------------
  Exp 0  Data cleaning + a log of exactly how many rows each rule removed
  Exp 1  Main comparison: 6 representations x 4 models x repeated stratified
         5x5 CV (the same folds for every configuration -> paired results)
  Exp 2  Statistical tests: Nadeau-Bengio corrected resampled t-test
         (primary), Wilcoxon (secondary), Holm correction, mean diff + 95% CI
  Exp 3  Ablation of each semantic feature group (from R3)
  Exp 4  Interpretability: grouped SHAP by clinical concept, explanation
         compactness, and ranking stability across folds
  Exp 5  Calibration (Brier score) reported for every configuration

Representations
---------------
  R1  Raw attributes
  R2  Semantic only (+ sex)
  R3  Raw + all semantic features
  R4  Raw + basic semantic features (BP category, BMI category, age group)
  R5  Raw + standard numeric derivations (BMI, pulse pressure, MAP)
      -> answers the reviewer question "is it the semantics or just BMI?"
  R6  Raw + automated degree-2 interactions
      -> "complex, knowledge-free" feature engineering baseline

Usage
-----
  pip install pandas numpy scikit-learn scipy matplotlib xgboost shap
  python cvd_semantic_experiments.py --data cardio_train.csv --out results
  python cvd_semantic_experiments.py --data cardio_train.csv --quick   # fast check
  python cvd_semantic_experiments.py --synthetic --quick                # no data needed

The Kaggle file "cardio_train.csv" is ';'-separated. Both ';' and ',' work.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # fallback so the pipeline still runs
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 42

# ---------------------------------------------------------------------------
# 1. Fixed, literature-based thresholds (report these in the paper's
#    feature-definition table). They are fixed rules, NOT fitted on data,
#    so computing them before splitting causes no leakage.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # Plausibility filters for cleaning (state these in Section 3.1)
    "ap_hi_range": (80, 250),      # systolic mmHg
    "ap_lo_range": (40, 150),      # diastolic mmHg
    "height_range": (120, 220),    # cm
    "weight_range": (30, 200),     # kg
    "bmi_range": (12, 60),         # kg/m^2
    # Whelton et al. 2017 ACC/AHA guideline (Hypertension 71:e13-e115)
    #   normal <120 and <80 | elevated 120-129 and <80
    #   stage 1 130-139 or 80-89 | stage 2 >=140 or >=90
    "bp": {"elevated_sys": 120, "stage1_sys": 130, "stage2_sys": 140,
           "stage1_dia": 80, "stage2_dia": 90},
    # WHO BMI classification: <18.5 | 18.5-24.9 | 25-29.9 | >=30
    "bmi": (18.5, 25.0, 30.0),
    # Age groups: 45 and 55 are the classical age risk-factor cut-offs
    # (men >=45, women >=55; NCEP ATP III 2002). Dataset ages ~29-65.
    "age": (45, 55),
}

RAW = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
       "cholesterol", "gluc", "smoke", "alco", "active"]
SEMANTIC_BASIC = ["bp_cat", "bmi_cat", "age_risk"]
SEMANTIC_AGG = ["metabolic_risk", "lifestyle_risk", "csri"]
SEMANTIC = SEMANTIC_BASIC + SEMANTIC_AGG
NUMERIC_DERIVED = ["bmi", "pulse_pressure", "map"]

# Semantic groups for the ablation (Exp 3)
ABLATION_GROUPS = {
    "BP category": ["bp_cat"],
    "BMI category": ["bmi_cat"],
    "Age group": ["age_risk"],
    "Metabolic risk": ["metabolic_risk"],
    "Lifestyle risk": ["lifestyle_risk"],
    "CSRI": ["csri"],
    "All aggregates (metab+life+CSRI)": SEMANTIC_AGG,
}

# Clinical concept groups for grouped SHAP (Exp 4)
CONCEPTS = {
    "Blood pressure": ["ap_hi", "ap_lo", "bp_cat", "pulse_pressure", "map"],
    "Body composition": ["height", "weight", "bmi", "bmi_cat"],
    "Metabolic": ["cholesterol", "gluc", "metabolic_risk"],
    "Lifestyle": ["smoke", "alco", "active", "lifestyle_risk"],
    "Age": ["age", "age_risk"],
    "Sex": ["gender"],
    "Composite (CSRI)": ["csri"],
}

# Classification threshold for F1/precision/recall/specificity:
# "fixed" = 0.5 (balanced cardio data); "prevalence" = training-fold prevalence
# (used for the imbalanced Framingham data, ~15% positives; no leakage).
THRESHOLD_MODE = "fixed"
DATASET = "cardio"

# ---------------------------------------------------------------------------
# Framingham (external validation dataset) configuration
#   Columns renamed to the common names: male->gender, sysBP->ap_hi,
#   diaBP->ap_lo, BMI->bmi, TenYearCHD->cardio
# ---------------------------------------------------------------------------
FRAM_RAW = ["age", "gender", "ap_hi", "ap_lo", "bmi", "totChol", "glucose",
            "currentSmoker", "cigsPerDay", "BPMeds", "diabetes", "prevalentHyp",
            "prevalentStroke", "heartRate"]
FRAM_THRESHOLDS = {
    "chol": (200, 240),     # NCEP ATP III total cholesterol mg/dL: desirable/borderline/high
    "glucose": (100, 126),  # ADA glucose mg/dL: normal/impaired/diabetic range
    "heavy_smoker_cigs": 20,
}
FRAM_CONCEPTS = {
    "Blood pressure": ["ap_hi", "ap_lo", "bp_cat", "pulse_pressure", "map",
                       "BPMeds", "prevalentHyp"],
    "Body composition": ["bmi", "bmi_cat"],
    "Metabolic": ["totChol", "glucose", "diabetes", "metabolic_risk"],
    "Lifestyle": ["currentSmoker", "cigsPerDay", "lifestyle_risk"],
    "Age": ["age", "age_risk"],
    "Sex": ["gender"],
    "Other clinical": ["heartRate", "prevalentStroke"],
    "Composite (CSRI)": ["csri"],
}


def configure_framingham():
    """Switch the module-level settings to the Framingham dataset."""
    global RAW, NUMERIC_DERIVED, CONCEPTS, THRESHOLD_MODE, DATASET
    RAW = FRAM_RAW
    NUMERIC_DERIVED = ["pulse_pressure", "map"]   # BMI is already a raw column
    CONCEPTS = FRAM_CONCEPTS
    THRESHOLD_MODE = "prevalence"
    DATASET = "framingham"


def clean_framingham(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # load_data lower-cases headers; map back to the names used below
    canon = {c.lower(): c for c in FRAM_RAW}
    canon.update({"male": "gender", "sysbp": "ap_hi", "diabp": "ap_lo",
                  "bmi": "bmi", "tenyearchd": "cardio"})
    df = df.rename(columns={c: canon.get(c.lower(), c) for c in df.columns})
    log = [("Initial records", len(df), 0)]

    def step(name, mask):
        nonlocal df
        before = len(df)
        df = df[mask(df)].copy()
        log.append((name, len(df), before - len(df)))

    needed = FRAM_RAW + ["cardio"]
    step("Incomplete records (complete-case analysis)", lambda d: d[needed].notna().all(axis=1))
    step("Duplicate rows removed", lambda d: ~d.duplicated())
    lo, hi = THRESHOLDS["ap_hi_range"]
    step(f"Systolic outside {lo}-{hi} mmHg", lambda d: d.ap_hi.between(lo, hi))
    lo, hi = THRESHOLDS["ap_lo_range"]
    step(f"Diastolic outside {lo}-{hi} mmHg", lambda d: d.ap_lo.between(lo, hi))
    step("Diastolic >= systolic", lambda d: d.ap_lo < d.ap_hi)
    lo, hi = THRESHOLDS["bmi_range"]
    step(f"BMI outside {lo}-{hi}", lambda d: d.bmi.between(lo, hi))
    df = df[needed].reset_index(drop=True)
    log_df = pd.DataFrame(log, columns=["Step", "Remaining", "Removed"])
    log_df["Positive class %"] = np.nan
    log_df.loc[log_df.index[-1], "Positive class %"] = round(100 * df.cardio.mean(), 2)
    return df, log_df


def add_features_framingham(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    b, t, ft = THRESHOLDS["bp"], THRESHOLDS, FRAM_THRESHOLDS
    d["bp_cat"] = np.select(
        [(d.ap_hi >= b["stage2_sys"]) | (d.ap_lo >= b["stage2_dia"]),
         (d.ap_hi >= b["stage1_sys"]) | (d.ap_lo >= b["stage1_dia"]),
         (d.ap_hi >= b["elevated_sys"])], [3, 2, 1], default=0)
    c1, c2, c3 = t["bmi"]
    d["bmi_cat"] = np.select([d.bmi < c1, d.bmi < c2, d.bmi < c3], [0, 1, 2], default=3)
    a1, a2 = t["age"]
    d["age_risk"] = np.select([d.age < a1, d.age < a2], [0, 1], default=2)
    chol_cat = np.select([d.totChol < ft["chol"][0], d.totChol < ft["chol"][1]], [0, 1], default=2)
    gluc_cat = np.select([d.glucose < ft["glucose"][0], d.glucose < ft["glucose"][1]], [0, 1], default=2)
    d["metabolic_risk"] = chol_cat + gluc_cat                                   # 0..4
    d["lifestyle_risk"] = np.select([d.currentSmoker == 0, d.cigsPerDay < ft["heavy_smoker_cigs"]],
                                    [0, 1], default=2)                          # 0..2
    bmi_risk = np.clip(d.bmi_cat - 1, 0, 2)
    comps = [d.bp_cat / 3, bmi_risk / 2, d.metabolic_risk / 4,
             d.lifestyle_risk / 2, d.age_risk / 2]
    d["csri"] = np.mean(comps, axis=0)
    d["pulse_pressure"] = d.ap_hi - d.ap_lo
    d["map"] = d.ap_lo + (d.ap_hi - d.ap_lo) / 3
    return d


def threshold_for(ytr) -> float:
    return float(np.mean(ytr)) if THRESHOLD_MODE == "prevalence" else 0.5


# ---------------------------------------------------------------------------
# 2. Data loading and cleaning
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def make_synthetic(n: int = 12000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic data with the same schema, ONLY for testing the code."""
    r = np.random.default_rng(seed)
    age_days = r.integers(30 * 365, 65 * 365, n)
    gender = r.integers(1, 3, n)
    height = np.where(gender == 2, r.normal(170, 7, n), r.normal(161, 7, n)).round()
    weight = r.normal(74, 14, n).round(1)
    ap_hi = (r.normal(122, 16, n) + (age_days / 365 - 50) * 0.5).round()
    ap_lo = (ap_hi * 0.65 + r.normal(0, 7, n)).round()
    chol = r.choice([1, 2, 3], n, p=[0.75, 0.14, 0.11])
    gluc = r.choice([1, 2, 3], n, p=[0.85, 0.07, 0.08])
    smoke = r.binomial(1, 0.09, n)
    alco = r.binomial(1, 0.05, n)
    active = r.binomial(1, 0.8, n)
    bmi = weight / (height / 100) ** 2
    logit = (-12 + 0.06 * (age_days / 365) + 0.055 * (ap_hi - 120) / 1.0 * 1.0
             + 0.4 * (chol - 1) + 0.1 * (gluc - 1) + 0.03 * (bmi - 25)
             - 0.2 * active + 0.6 * (ap_hi >= 140) + 0.12 * 120 / 12)
    p = 1 / (1 + np.exp(-(logit - np.median(logit))))
    y = r.binomial(1, p)
    df = pd.DataFrame(dict(id=np.arange(n), age=age_days, gender=gender, height=height,
                           weight=weight, ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=chol,
                           gluc=gluc, smoke=smoke, alco=alco, active=active, cardio=y))
    # inject the kind of errors the real dataset has
    bad = r.choice(n, n // 60, replace=False)
    df.loc[bad[: len(bad) // 2], "ap_hi"] *= 10
    df.loc[bad[len(bad) // 2:], "ap_lo"] = -df.loc[bad[len(bad) // 2:], "ap_lo"]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns cleaned data and a per-rule log (report it as a table/flowchart)."""
    log = [("Initial records", len(df), 0)]
    df = df.drop(columns=[c for c in ["id"] if c in df.columns]).copy()

    def step(name, mask):
        nonlocal df
        before = len(df)
        df = df[mask(df)].copy()
        log.append((name, len(df), before - len(df)))

    step("Duplicate rows removed", lambda d: ~d.duplicated())
    if df["age"].max() > 200:  # Kaggle stores age in days
        df["age"] = df["age"] / 365.25
    lo, hi = THRESHOLDS["ap_hi_range"]
    step(f"Systolic outside {lo}-{hi} mmHg", lambda d: d.ap_hi.between(lo, hi))
    lo, hi = THRESHOLDS["ap_lo_range"]
    step(f"Diastolic outside {lo}-{hi} mmHg", lambda d: d.ap_lo.between(lo, hi))
    step("Diastolic >= systolic", lambda d: d.ap_lo < d.ap_hi)
    lo, hi = THRESHOLDS["height_range"]
    step(f"Height outside {lo}-{hi} cm", lambda d: d.height.between(lo, hi))
    lo, hi = THRESHOLDS["weight_range"]
    step(f"Weight outside {lo}-{hi} kg", lambda d: d.weight.between(lo, hi))
    df["bmi"] = df.weight / (df.height / 100) ** 2
    lo, hi = THRESHOLDS["bmi_range"]
    step(f"BMI outside {lo}-{hi}", lambda d: d.bmi.between(lo, hi))
    df = df.reset_index(drop=True)
    log_df = pd.DataFrame(log, columns=["Step", "Remaining", "Removed"])
    log_df["Positive class %"] = np.nan
    log_df.loc[log_df.index[-1], "Positive class %"] = round(100 * df.cardio.mean(), 2)
    return df, log_df


# ---------------------------------------------------------------------------
# 3. Semantic feature construction (deterministic, rule-based)
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t = THRESHOLDS
    b = t["bp"]
    sys_, dia = d.ap_hi, d.ap_lo
    d["bp_cat"] = np.select(
        [(sys_ >= b["stage2_sys"]) | (dia >= b["stage2_dia"]),
         (sys_ >= b["stage1_sys"]) | (dia >= b["stage1_dia"]),
         (sys_ >= b["elevated_sys"])],
        [3, 2, 1], default=0)                    # 0 normal ... 3 stage 2
    c1, c2, c3 = t["bmi"]
    d["bmi_cat"] = np.select([d.bmi < c1, d.bmi < c2, d.bmi < c3], [0, 1, 2], default=3)
    a1, a2 = t["age"]
    d["age_risk"] = np.select([d.age < a1, d.age < a2], [0, 1], default=2)
    d["metabolic_risk"] = (d.cholesterol - 1) + (d.gluc - 1)              # 0..4
    d["lifestyle_risk"] = d.smoke + d.alco + (1 - d.active)               # 0..3
    # BMI risk for the composite: underweight & normal -> 0, overweight 1, obese 2
    bmi_risk = np.clip(d.bmi_cat - 1, 0, 2)
    comps = [d.bp_cat / 3, bmi_risk / 2, d.metabolic_risk / 4,
             d.lifestyle_risk / 3, d.age_risk / 2]
    d["csri"] = np.mean(comps, axis=0)                                    # 0..1
    d["pulse_pressure"] = d.ap_hi - d.ap_lo
    d["map"] = d.ap_lo + (d.ap_hi - d.ap_lo) / 3
    return d


def representations() -> dict[str, list[str]]:
    return {
        "R1 Raw": RAW,
        "R2 Semantic": ["gender"] + SEMANTIC,
        "R3 Raw+Semantic": RAW + SEMANTIC,
        "R4 Raw+Basic semantic": RAW + SEMANTIC_BASIC,
        "R5 Raw+Numeric derived": RAW + NUMERIC_DERIVED,
        "R6 Raw+Auto interactions": RAW,   # interactions added inside the pipeline
    }


# ---------------------------------------------------------------------------
# 4. Models and search spaces (identical tuning budget for every representation)
# ---------------------------------------------------------------------------
def models(n_jobs: int) -> dict:
    if HAS_XGB:
        gb = XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                           random_state=SEED, n_jobs=n_jobs, verbosity=0)
        gb_space = {"clf__max_depth": [3, 4, 5, 6], "clf__learning_rate": [0.03, 0.05, 0.1],
                    "clf__subsample": [0.7, 0.85, 1.0], "clf__colsample_bytree": [0.7, 0.85, 1.0],
                    "clf__min_child_weight": [1, 5, 10]}
        gb_name = "XGBoost"
    else:
        gb = HistGradientBoostingClassifier(random_state=SEED)
        gb_space = {"clf__max_depth": [3, 4, 5, 6], "clf__learning_rate": [0.03, 0.05, 0.1],
                    "clf__max_iter": [200, 300], "clf__min_samples_leaf": [20, 50, 100]}
        gb_name = "GradBoost (XGB fallback)"
    return {
        "LogReg": (LogisticRegression(max_iter=2000),
                   {"clf__C": [0.01, 0.1, 1, 10]}, True),
        "DecisionTree": (DecisionTreeClassifier(random_state=SEED),
                         {"clf__max_depth": [4, 6, 8, 10, 12],
                          "clf__min_samples_leaf": [20, 50, 100, 200]}, False),
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=n_jobs),
                         {"clf__max_depth": [8, 10, 12, None],
                          "clf__min_samples_leaf": [5, 10, 20, 50],
                          "clf__max_features": ["sqrt", 0.5]}, False),
        gb_name: (gb, gb_space, False),
    }


def build_pipeline(est, scale: bool, rep_name: str) -> Pipeline:
    steps = []
    if rep_name.startswith("R6"):
        steps.append(("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)))
    if scale or rep_name.startswith("R6"):
        steps.append(("scale", StandardScaler()))   # fitted on training fold only
    steps.append(("clf", est))
    return Pipeline(steps)


def metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(F1=f1_score(y, pred), Accuracy=accuracy_score(y, pred),
                Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), Specificity=tn / (tn + fp),
                ROC_AUC=roc_auc_score(y, prob), PR_AUC=average_precision_score(y, prob),
                Brier=brier_score_loss(y, prob))


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------
def corrected_ttest(diff: np.ndarray, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.
    Plain tests over-state significance because CV folds overlap."""
    k = len(diff)
    mean, var = diff.mean(), diff.var(ddof=1)
    se = np.sqrt((1 / k + n_test / n_train) * var)
    if se == 0:
        return mean, 0.0, 1.0, (mean, mean)
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    h = stats.t.ppf(0.975, k - 1) * se
    return mean, t, p, (mean - h, mean + h)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Experiment runners
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    n_splits: int = 5
    n_repeats: int = 5
    tune: bool = True
    n_iter: int = 8
    inner_cv: int = 3
    tune_subsample: int = 10000   # rows used for inner tuning (speed); 0 = all
    n_jobs: int = -1
    resume: bool = True           # skip fits already saved (survives disconnects)


def fit_one(est, space, scale, rep, Xtr, ytr, cfg: Cfg, rng):
    pipe = build_pipeline(clone(est), scale, rep)
    if cfg.tune and space:
        idx = np.arange(len(ytr))
        if cfg.tune_subsample and len(idx) > cfg.tune_subsample:
            idx = rng.choice(idx, cfg.tune_subsample, replace=False)
        n_comb = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(pipe, space, n_iter=min(cfg.n_iter, n_comb),
                                    scoring="roc_auc", random_state=SEED,
                                    cv=StratifiedKFold(cfg.inner_cv, shuffle=True, random_state=SEED),
                                    n_jobs=cfg.n_jobs)
        search.fit(Xtr.iloc[idx], ytr[idx])
        pipe.set_params(**search.best_params_)
        best = search.best_params_
    else:
        best = {}
    pipe.fit(Xtr, ytr)
    return pipe, best


def run_main(df, cfg: Cfg, out: str):
    reps, mods = representations(), models(cfg.n_jobs)
    y = df.cardio.values
    cv = RepeatedStratifiedKFold(n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=SEED)
    folds = list(cv.split(df, y))                           # SAME folds for everything
    rng = np.random.default_rng(SEED)
    res_path = f"{out}/exp1_fold_results.csv"
    par_path = f"{out}/exp1_best_hyperparams.csv"
    done_keys = set()
    if cfg.resume and os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        done_keys = set(zip(prev.fold, prev.representation, prev.model))
        print(f"  resuming: {len(done_keys)} fits already saved")
    else:
        for p in (res_path, par_path):
            if os.path.exists(p):
                os.remove(p)
    total = len(folds) * len(reps) * len(mods)
    done, t0, new = len(done_keys), time.time(), 0
    for f, (tr, te) in enumerate(folds):
        for rep, cols in reps.items():
            X = df[cols]
            for mname, (est, space, scale) in mods.items():
                if (f, rep, mname) in done_keys:
                    continue
                pipe, best = fit_one(est, space, scale, rep, X.iloc[tr], y[tr], cfg, rng)
                prob = pipe.predict_proba(X.iloc[te])[:, 1]
                row = dict(fold=f, repeat=f // cfg.n_splits, model=mname,
                           representation=rep, **metrics(y[te], prob, threshold_for(y[tr])))
                # save every fit immediately, so a disconnect loses nothing
                pd.DataFrame([row]).to_csv(res_path, mode="a", index=False,
                                           header=not os.path.exists(res_path))
                pd.DataFrame([dict(fold=f, model=mname, representation=rep,
                                   params=json.dumps(best, default=str))]).to_csv(
                    par_path, mode="a", index=False, header=not os.path.exists(par_path))
                done += 1
                new += 1
                el = time.time() - t0
                eta = el / new * (total - done) / 60
                print(f"\r  main: {done}/{total} fits  ({el / 60:.1f} min, ~{eta:.0f} min left)",
                      end="", flush=True)
    print()
    res = pd.read_csv(res_path)
    return res, folds


def summarise(res: pd.DataFrame, out: str):
    m = [c for c in ["F1", "ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall",
                     "Specificity", "Brier"]]
    g = res.groupby(["model", "representation"])[m]
    tab = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    tab.to_csv(f"{out}/table_main_results.csv")
    return tab


COMPARISONS = [("R3 Raw+Semantic", "R1 Raw"),          # primary
               ("R2 Semantic", "R1 Raw"),
               ("R4 Raw+Basic semantic", "R1 Raw"),
               ("R3 Raw+Semantic", "R4 Raw+Basic semantic"),
               ("R3 Raw+Semantic", "R5 Raw+Numeric derived"),
               ("R6 Raw+Auto interactions", "R1 Raw")]


def run_stats(res, n, cfg: Cfg, out: str, metrics_=("ROC_AUC", "F1", "Brier")):
    n_test = n // cfg.n_splits
    n_train = n - n_test
    rows = []
    for metric in metrics_:
        for model in res.model.unique():
            sub = res[res.model == model].pivot(index="fold", columns="representation", values=metric)
            for a, b in COMPARISONS:
                diff = (sub[a] - sub[b]).values
                mean, t, p, ci = corrected_ttest(diff, n_train, n_test)
                try:
                    pw = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                except ValueError:
                    pw = 1.0
                rows.append(dict(metric=metric, model=model, comparison=f"{a} vs {b}",
                                 mean_diff=mean, ci95_low=ci[0], ci95_high=ci[1], t=t,
                                 p_corrected_ttest=p, p_wilcoxon=pw, cohens_dz=cohens_dz(diff),
                                 wins=int((diff > 0).sum()), folds=len(diff)))
    st = pd.DataFrame(rows)
    # Holm within each metric family
    st["p_holm"] = st.groupby("metric")["p_corrected_ttest"].transform(lambda s: holm(s.tolist()))
    st["significant_0.05"] = st.p_holm < 0.05
    st.round(6).to_csv(f"{out}/exp2_statistical_tests.csv", index=False)
    return st


def run_ablation(df, folds, cfg: Cfg, out: str, model_names=None):
    mods = models(cfg.n_jobs)
    if model_names:
        mods = {k: v for k, v in mods.items() if k in model_names}
    y = df.cardio.values
    rng = np.random.default_rng(SEED)
    configs = {"Full R3": RAW + SEMANTIC}
    for g, cols in ABLATION_GROUPS.items():
        configs[f"- {g}"] = [c for c in RAW + SEMANTIC if c not in cols]
    n_folds = len(folds)
    ab_path = f"{out}/exp3_ablation_folds.csv"
    done_keys = set()
    if cfg.resume and os.path.exists(ab_path):
        prev = pd.read_csv(ab_path)
        done_keys = set(zip(prev.fold, prev.config, prev.model))
    elif os.path.exists(ab_path):
        os.remove(ab_path)
    for f, (tr, te) in enumerate(folds):
        for cname, cols in configs.items():
            for mname, (est, space, scale) in mods.items():
                if (f, cname, mname) in done_keys:
                    continue
                pipe, _ = fit_one(est, space, scale, "R3", df[cols].iloc[tr], y[tr], cfg, rng)
                prob = pipe.predict_proba(df[cols].iloc[te])[:, 1]
                row = dict(fold=f, model=mname, config=cname,
                           **metrics(y[te], prob, threshold_for(y[tr])))
                pd.DataFrame([row]).to_csv(ab_path, mode="a", index=False,
                                           header=not os.path.exists(ab_path))
        print(f"\r  ablation: fold {f + 1}/{n_folds}", end="", flush=True)
    print()
    ab = pd.read_csv(ab_path)
    ab = ab[ab.model.isin(mods.keys())]
    n_test = len(df) // cfg.n_splits
    summ = []
    for model in ab.model.unique():
        for metric in ["F1", "ROC_AUC"]:
            pv = ab[ab.model == model].pivot(index="fold", columns="config", values=metric)
            for c in configs:
                if c == "Full R3":
                    continue
                diff = (pv["Full R3"] - pv[c]).values      # Delta = full - ablated
                mean, t, p, ci = corrected_ttest(diff, len(df) - n_test, n_test)
                summ.append(dict(model=model, metric=metric, removed=c[2:], delta=mean,
                                 ci95_low=ci[0], ci95_high=ci[1], p_corrected=p))
    s = pd.DataFrame(summ)
    s["p_holm"] = s.groupby(["model", "metric"])["p_corrected"].transform(lambda v: holm(v.tolist()))
    s.round(6).to_csv(f"{out}/exp3_ablation_summary.csv", index=False)
    return s


def _attributions(pipe, Xtr, Xte, cols, n_bg=200):
    """Mean |attribution| per feature. Uses SHAP if installed, else permutation importance."""
    if HAS_SHAP:
        clf = pipe.named_steps["clf"]
        Xt = Xte
        for name, step in pipe.steps[:-1]:
            Xt = step.transform(Xt)
        Xt = pd.DataFrame(np.asarray(Xt), columns=cols)
        if isinstance(clf, LogisticRegression):
            bg = Xtr.sample(min(n_bg, len(Xtr)), random_state=SEED)
            for name, step in pipe.steps[:-1]:
                bg = step.transform(bg)
            exp = shap.LinearExplainer(clf, np.asarray(bg))
            sv = exp.shap_values(Xt.values)
        else:
            exp = shap.TreeExplainer(clf)
            sv = exp.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        return pd.Series(np.abs(sv).mean(0), index=cols), "SHAP"
    from sklearn.inspection import permutation_importance
    yte = Xte.attrs["y"]
    pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    return pd.Series(np.clip(pi.importances_mean, 0, None), index=cols), "Permutation"


def run_explainability(df, folds, cfg: Cfg, out: str, model_name: str, n_explain=2000):
    """Grouped attributions by clinical concept + compactness + rank stability."""
    est, space, scale = models(cfg.n_jobs)[model_name]
    y = df.cardio.values
    rng = np.random.default_rng(SEED)
    reps = {k: v for k, v in representations().items() if k[:2] in ("R1", "R2", "R3")}
    first_repeat = folds[: cfg.n_splits]
    per_fold, method = [], None
    for f, (tr, te) in enumerate(first_repeat):
        te_s = rng.choice(te, min(n_explain, len(te)), replace=False)
        for rep, cols in reps.items():
            pipe, _ = fit_one(est, space, scale, rep, df[cols].iloc[tr], y[tr], cfg, rng)
            Xte = df[cols].iloc[te_s].copy()
            Xte.attrs["y"] = y[te_s]
            imp, method = _attributions(pipe, df[cols].iloc[tr], Xte, cols)
            for feat, v in imp.items():
                per_fold.append(dict(fold=f, representation=rep, feature=feat, importance=v))
    pf = pd.DataFrame(per_fold)
    pf.to_csv(f"{out}/exp4_feature_importance_folds.csv", index=False)

    concept_of = {f: c for c, fs in CONCEPTS.items() for f in fs}
    pf["concept"] = pf.feature.map(concept_of)
    pf["is_semantic"] = pf.feature.isin(SEMANTIC)

    # (a) concept-level share of total attribution
    conc = (pf.groupby(["representation", "fold", "concept"]).importance.sum()
              .groupby(level=[0, 1]).transform(lambda s: s / s.sum())
              .groupby(level=[0, 2]).agg(["mean", "std"]).round(4))
    conc.to_csv(f"{out}/exp4_concept_shares.csv")

    # (b) how much of R3's attribution goes to semantic features (per concept)
    r3 = pf[pf.representation.str.startswith("R3")]
    sem_share = (r3.groupby(["concept", "is_semantic"]).importance.mean().unstack(fill_value=0))
    sem_share["semantic_share_%"] = 100 * sem_share.get(True, 0) / sem_share.sum(axis=1)
    sem_share.round(4).to_csv(f"{out}/exp4_semantic_share_R3.csv")

    # (c) compactness: features needed to reach 80% of total attribution
    # (d) stability: mean pairwise Kendall tau of feature rankings across folds
    rows = []
    for rep, g in pf.groupby("representation"):
        mats = g.pivot(index="fold", columns="feature", values="importance")
        k80 = []
        for _, r in mats.iterrows():
            s = r.sort_values(ascending=False)
            cum = s.cumsum() / max(s.sum(), 1e-12)
            k80.append(int((cum < 0.8).sum() + 1))
        taus = [stats.kendalltau(mats.iloc[i], mats.iloc[j]).statistic
                for i, j in itertools.combinations(range(len(mats)), 2)]
        rows.append(dict(representation=rep, n_features=mats.shape[1],
                         features_for_80pct_mean=np.mean(k80), features_for_80pct_sd=np.std(k80),
                         rank_stability_kendall_tau=np.nanmean(taus), method=method))
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(f"{out}/exp4_compactness_stability.csv", index=False)
    return conc, sem_share, comp, method


def plot_main(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    reps = list(representations().keys())
    mods_ = list(res.model.unique())
    fig, axes = plt.subplots(1, len(mods_), figsize=(4.2 * len(mods_), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, mods_):
        data = [res[(res.model == m) & (res.representation == r)].ROC_AUC.values for r in reps]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(reps) + 1), [r.split()[0] for r in reps])
        ax.set_title(m)
        ax.set_xlabel("Representation")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("ROC-AUC across 25 folds")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_auc_by_representation.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to cardio_train.csv or framingham.csv")
    ap.add_argument("--dataset", choices=["cardio", "framingham"], default="cardio")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved results in --out and start again")
    ap.add_argument("--synthetic", action="store_true", help="test run on synthetic data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="2x2 CV, no tuning, 8k-row sample: checks the code end to end")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=["ablation", "explain"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg(n_jobs=a.n_jobs, tune=not a.no_tune, resume=not a.fresh)
    if a.quick:
        cfg = Cfg(n_splits=2, n_repeats=2, tune=False, n_jobs=a.n_jobs, resume=False)

    if a.dataset == "framingham":
        configure_framingham()
        df, log = clean_framingham(load_data(a.data))
        df = add_features_framingham(df)
    else:
        raw = make_synthetic() if a.synthetic or not a.data else load_data(a.data)
        df, log = clean(raw)
        if a.quick and len(df) > 8000:
            df = df.sample(8000, random_state=SEED).reset_index(drop=True)
        df = add_features(df)
    log.to_csv(f"{a.out}/exp0_cleaning_log.csv", index=False)
    print("\n== Exp 0: cleaning ==\n" + log.to_string(index=False))

    # feature distribution table (Table: semantic feature definitions & prevalence)
    dist = []
    for f in SEMANTIC_BASIC + ["metabolic_risk", "lifestyle_risk"]:
        ct = pd.crosstab(df[f], df.cardio, normalize="index")[1].round(3)
        n = df[f].value_counts().sort_index()
        for lvl in n.index:
            dist.append(dict(feature=f, level=lvl, n=int(n[lvl]),
                             pct=round(100 * n[lvl] / len(df), 1), cvd_rate=ct[lvl]))
    pd.DataFrame(dist).to_csv(f"{a.out}/table_semantic_feature_distribution.csv", index=False)

    meta = dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, pandas=pd.__version__,
                sklearn=__import__("sklearn").__version__,
                xgboost=__import__("xgboost").__version__ if HAS_XGB else "not installed (fallback used)",
                shap=shap.__version__ if HAS_SHAP else "not installed (permutation fallback)",
                dataset=DATASET, threshold_mode=THRESHOLD_MODE,
                seed=SEED, cv=f"{cfg.n_splits}-fold x {cfg.n_repeats} repeats",
                tuning=cfg.tune, n_iter=cfg.n_iter, inner_cv=cfg.inner_cv,
                thresholds=THRESHOLDS,
                framingham_thresholds=FRAM_THRESHOLDS if DATASET == "framingham" else None,
                n_records=len(df), synthetic=bool(a.synthetic or not a.data))
    json.dump(meta, open(f"{a.out}/run_metadata.json", "w"), indent=2, default=str)

    print(f"\n== Exp 1: main comparison ({cfg.n_splits}x{cfg.n_repeats} CV, tuning={cfg.tune}) ==")
    res, folds = run_main(df, cfg, a.out)
    tab = summarise(res, a.out)
    print(tab[["F1", "ROC_AUC", "Brier"]].to_string())
    plot_main(res, a.out)

    print("\n== Exp 2: statistics (corrected t-test, Holm) ==")
    st = run_stats(res, len(df), cfg, a.out)
    show = st[st.metric == "ROC_AUC"][["model", "comparison", "mean_diff", "ci95_low",
                                       "ci95_high", "p_holm", "significant_0.05"]]
    print(show.round(4).to_string(index=False))

    best_model = (res.groupby("model").ROC_AUC.mean().idxmax())
    if "ablation" not in a.skip:
        abl_models = sorted({best_model, "LogReg"})
        print(f"\n== Exp 3: ablation ({', '.join(abl_models)}) ==")
        ab = run_ablation(df, folds, cfg, a.out, model_names=abl_models)
        print(ab[ab.metric == "ROC_AUC"].round(4).to_string(index=False))
    if "explain" not in a.skip:
        print(f"\n== Exp 4: explainability ({best_model}) ==")
        conc, sem, comp, method = run_explainability(df, folds, cfg, a.out, best_model)
        print(f"attribution method: {method}")
        print(comp.to_string(index=False))
        print(sem.round(4).to_string())
    print(f"\nAll outputs written to: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
