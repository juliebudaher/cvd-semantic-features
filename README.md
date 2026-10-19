# Semantic feature engineering for cardiovascular disease prediction

Code and results for the paper:

> Bu Daher J. *Semantic feature engineering for cardiovascular disease prediction: a controlled study of representation, performance and explainability.* Submitted to *Computers in Biology and Medicine*, 2026.

The study tests whether guideline-defined clinical concepts (blood-pressure stage, BMI class, age group, metabolic and lifestyle risk, and a composite index) change the performance or explainability of CVD prediction models. Six input representations are compared across four learners and two datasets, using 5 × 5 repeated stratified cross-validation, nested tuning and Nadeau–Bengio corrected tests.

## Contents

| Path | What it is |
| --- | --- |
| `cvd_semantic_experiments.py` | Full pipeline: cleaning, semantic features, six representations, tuning, cross-validation, statistics, ablation, SHAP analysis |
| `learning_curve.py` | Learning-curve experiment (1%–100% of each training fold) |
| `make_figures.py` | Figures 1–3 of the paper |
| `make_supplement.py` | Supplementary tables S1–S5 |
| `CVD_Experiments_Colab.ipynb` | The same pipeline as a Google Colab notebook |
| `results_cardio/`, `results_framingham/`, `results_learning_curve/` | All fold-level results, statistical tests, ablation and SHAP outputs |
| `figures/` | Paper figures |
| `data/` | Where to put the two Kaggle datasets (see `data/README.md`) |

## How to reproduce

```bash
pip install -r requirements.txt
# put cardio_train.csv and framingham.csv in data/
python cvd_semantic_experiments.py --data data/cardio_train.csv --dataset cardio --out results_cardio
python cvd_semantic_experiments.py --data data/framingham.csv --dataset framingham --out results_framingham
python learning_curve.py
python make_figures.py 1 2 3
python make_supplement.py
```

Add `--quick` for a fast test run. Every run uses random seed 42. The main Kaggle run takes a few hours on a laptop CPU.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## License

Code: MIT License (see `LICENSE`). The datasets keep their original Kaggle licences.
