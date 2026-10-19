"""Builds Supplementary_Material.md (converted to .docx with pandoc)."""
import json, os, re
def code(s):
    return s.split()[0]
def comp(s):
    return " vs ".join(code(x) for x in s.split(" vs "))

import pandas as pd

def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

L = []
L.append("# Supplementary Material\n")
REPNOTE = "Representations: R1 raw; R2 semantic only; R3 raw + semantic; R4 raw + basic semantic; R5 raw + numeric derived; R6 raw + automatic interactions.\n"
L.append("**Semantic feature engineering for cardiovascular disease prediction: a controlled study of representation, performance and explainability**\n")

# ---- S1 SNOMED mapping
L.append("## Table S1. Semantic feature levels and corresponding SNOMED CT concepts\n")
L.append("Concept identifiers were taken from published SNOMED CT value sets and code lists (NHS England primary-care reference sets on OpenCodelists, HL7 FHIR value sets, and the SNOMED CT browser). Levels without a single matching pre-coordinated concept are mapped to the closest concept, as noted.\n")
s1 = [
    ("Blood-pressure category", "0 Normal (SBP < 120 and DBP < 80)", "2004005 | Normal blood pressure (finding)", ""),
    ("", "1 Elevated (SBP 120–129 and DBP < 80)", "No guideline-specific concept; parent 392570002 | Blood pressure finding", "Closest concept"),
    ("", "2 Stage 1 hypertension (SBP 130–139 or DBP 80–89)", "38341003 | Hypertensive disorder, systemic arterial (disorder)", "Stage not encoded in the concept"),
    ("", "3 Stage 2 hypertension (SBP ≥ 140 or DBP ≥ 90)", "38341003 | Hypertensive disorder, systemic arterial (disorder)", "Stage not encoded in the concept"),
    ("BMI category", "0 Underweight (< 18.5)", "248342006 | Underweight", ""),
    ("", "1 Normal (18.5–24.9)", "35425004 | Normal body mass index", ""),
    ("", "2 Overweight (25–29.9)", "238131007 | Overweight; 162863004 | Body mass index 25-29 - overweight", ""),
    ("", "3 Obese (≥ 30)", "414916001 | Obesity; 162864005 | Body mass index 30+ - obesity", ""),
    ("Metabolic risk", "Glucose above normal / impaired (Framingham 100–125 mg/dL)", "390951007 | Impaired fasting glycemia (disorder)", ""),
    ("", "Glucose well above normal / diabetic range (≥ 126 mg/dL)", "73211009 | Diabetes mellitus (disorder)", "Single reading; not a diagnosis"),
    ("", "Cholesterol above normal", "370992007 | Disorder of lipoprotein metabolism (disorder)", "Closest verified concept"),
    ("Lifestyle risk", "Smoking", "77176002 | Smoker (finding)", ""),
    ("", "Moderate/heavy smoking (Framingham ≥ 20 cigarettes/day)", "230062009 | Moderate cigarette smoker", "Closest verified concept"),
    ("", "Alcohol intake", "219006 | Current drinker of alcohol", ""),
    ("", "Physical inactivity", "68130003 | Physical activity (observable entity), value absent", "Post-coordinated"),
    ("Age group", "< 45, 45–54, ≥ 55 years", "No pre-coordinated concept; represented as the age observation", ""),
    ("CSRI", "Continuous 0–1", "None (study-specific index)", ""),
]
L.append("| Feature | Level | SNOMED CT concept | Note |\n| --- | --- | --- | --- |")
for r in s1:
    L.append("| " + " | ".join(r) + " |")
L.append("")

# ---- S2 full metrics
for d, lab in [("results_cardio", "Kaggle CVD data"), ("results_framingham", "Framingham data")]:
    r = pd.read_csv(f"{d}/exp1_fold_results.csv")
    L.append(f"## Table S2{'a' if 'cardio' in d else 'b'}. All metrics, {lab} (mean ± SD over 25 folds)\n")
    L.append(REPNOTE)
    mets = ["ROC_AUC", "PR_AUC", "F1", "Accuracy", "Precision", "Recall", "Specificity", "Brier"]
    L.append("| Model | Representation | " + " | ".join(m.replace("_", "-") for m in mets) + " |")
    L.append("| --- | --- | " + " | ".join("---" for _ in mets) + " |")
    g = r.groupby(["model", "representation"])
    for (mo, rep), sub in g:
        cells = [f"{sub[m].mean():.3f} ± {sub[m].std():.3f}" for m in mets]
        L.append(f"| {mo} | {code(rep)} | " + " | ".join(cells) + " |")
    L.append("")

# ---- S3 statistics
for d, lab in [("results_cardio", "Kaggle CVD data"), ("results_framingham", "Framingham data")]:
    s = pd.read_csv(f"{d}/exp2_statistical_tests.csv")
    L.append(f"## Table S3{'a' if 'cardio' in d else 'b'}. Pairwise comparisons, {lab}\n")
    L.append("Mean difference (first minus second representation) with 95% CI from the Nadeau–Bengio corrected resampled t-test; p-values Holm-adjusted within each metric; Wilcoxon signed-rank p unadjusted; d_z = mean difference / SD of differences; wins = folds (of 25) in which the first representation scored higher. For Brier, negative differences favour the first representation.\n")
    L.append("| Metric | Model | Comparison | Mean diff | 95% CI | p (Holm) | p (Wilcoxon) | d_z | Wins |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, x in s.iterrows():
        L.append(f"| {x.metric.replace('_','-')} | {x.model} | {comp(x.comparison)} | {x.mean_diff:+.4f} | [{x.ci95_low:+.4f}, {x.ci95_high:+.4f}] | {fp(x.p_holm)} | {fp(x.p_wilcoxon)} | {x.cohens_dz:+.2f} | {int(x.wins)}/{int(x.folds)} |")
    L.append("")

# ---- S4 hyperparameters
L.append("## Table S4. Most frequently selected hyperparameters (Kaggle CVD data)\n")
L.append("Configuration selected most often across the 25 outer folds by the inner randomised search, with the number of folds in which it was selected.\n")
h = pd.read_csv("results_cardio/exp1_best_hyperparams.csv")
L.append("| Model | Representation | Most frequent configuration | Folds |\n| --- | --- | --- | --- |")
for (mo, rep), sub in h.groupby(["model", "representation"]):
    vc = sub.params.value_counts()
    cfg = json.loads(vc.index[0])
    txt = ", ".join(f"{k.replace('clf__','')}={v}" for k, v in sorted(cfg.items()))
    L.append(f"| {mo} | {code(rep)} | {txt} | {vc.iloc[0]}/25 |")
L.append("")

# ---- S5 learning curve
if os.path.exists("results_learning_curve/lc_summary.csv"):
    sm = pd.read_csv("results_learning_curve/lc_summary.csv")
    L.append("## Table S5. Learning curve: ROC-AUC by training-set size (Kaggle CVD data, mean ± SD over 25 folds)\n")
    reps = ["R1 Raw", "R2 Semantic", "R3 Raw+Semantic"]
    L.append("| Model | Training records | " + " | ".join(reps) + " |")
    L.append("| --- | --- | --- | --- | --- |")
    for (mo, fr), sub in sm.groupby(["model", "fraction"]):
        sub = sub.set_index("representation")
        n = int(sub.n_train.iloc[0])
        cells = [f"{sub.loc[r,'auc_mean']:.3f} ± {sub.loc[r,'auc_sd']:.3f}" for r in reps]
        L.append(f"| {mo} | {n:,} ({fr:.0%}) | " + " | ".join(cells) + " |")
    L.append("")

open("Supplementary_Material.md", "w").write("\n".join(L))
print("ok", len(L))
