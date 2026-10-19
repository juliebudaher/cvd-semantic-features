"""
Learning-curve experiment (Exp 5)
Does the value of semantic representations change with training-set size?

For each of the same 25 outer folds used in the main experiment, the training
fold is subsampled (stratified) to 1%, 5%, 10%, 25%, 50% and 100%; models are
tuned and trained on the subsample and evaluated on the FULL validation fold.
Representations: R1 raw, R2 semantic only, R3 raw + semantic.
Models: logistic regression, decision tree, XGBoost (random forest omitted for time).
"""
import os, sys, time, json
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvd_semantic_experiments as m

OUT = "results_learning_curve"
os.makedirs(OUT, exist_ok=True)
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
REPS = {k: v for k, v in m.representations().items() if k[:2] in ("R1", "R2", "R3")}
MODELS = {k: v for k, v in m.models(-1).items() if k in ("LogReg", "DecisionTree", "XGBoost")}

df, _ = m.clean(m.load_data("data/cardio_train.csv"))
df = m.add_features(df)
y = df.cardio.values
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=m.SEED).split(df, y))
cfg = m.Cfg()
rng = np.random.default_rng(m.SEED)

path = f"{OUT}/lc_fold_results.csv"
done = set()
if os.path.exists(path):
    p = pd.read_csv(path)
    done = set(zip(p.fold, p.fraction, p.representation, p.model))
total = len(folds) * len(FRACTIONS) * len(REPS) * len(MODELS)
t0, n = time.time(), len(done)
for f, (tr, te) in enumerate(folds):
    for frac in FRACTIONS:
        if frac < 1.0:
            sub, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=m.SEED + f)
        else:
            sub = tr
        for rep, cols in REPS.items():
            for mname, (est, space, scale) in MODELS.items():
                if (f, frac, rep, mname) in done:
                    continue
                pipe, _ = m.fit_one(est, space, scale, rep, df[cols].iloc[sub], y[sub], cfg, rng)
                prob = pipe.predict_proba(df[cols].iloc[te])[:, 1]
                row = dict(fold=f, fraction=frac, n_train=len(sub), representation=rep,
                           model=mname, **m.metrics(y[te], prob))
                pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))
                n += 1
        print(f"\r fold {f+1}/25 frac {frac}: {n}/{total} ({(time.time()-t0)/60:.1f} min)", end="", flush=True)
print()

res = pd.read_csv(path)
summ = (res.groupby(["model", "fraction", "representation"])
          .agg(n_train=("n_train", "mean"), auc_mean=("ROC_AUC", "mean"), auc_sd=("ROC_AUC", "std"),
               brier_mean=("Brier", "mean"))
          .reset_index())
summ.to_csv(f"{OUT}/lc_summary.csv", index=False)

rows = []
n_test = len(df) // 5
for (model, frac), g in res.groupby(["model", "fraction"]):
    pv = g.pivot(index="fold", columns="representation", values="ROC_AUC")
    n_train = int(g.n_train.mean())
    for a, b in [("R2 Semantic", "R1 Raw"), ("R3 Raw+Semantic", "R1 Raw")]:
        d = (pv[a] - pv[b]).values
        mean, t, p, ci = m.corrected_ttest(d, n_train, n_test)
        rows.append(dict(model=model, fraction=frac, n_train=n_train, comparison=f"{a} vs {b}",
                         mean_diff=mean, ci95_low=ci[0], ci95_high=ci[1], p_corrected=p,
                         wins=int((d > 0).sum())))
st = pd.DataFrame(rows)
st["p_holm"] = st.groupby("comparison")["p_corrected"].transform(lambda s: m.holm(s.tolist()))
st.round(5).to_csv(f"{OUT}/lc_stats.csv", index=False)
print(st.round(4).to_string(index=False))
