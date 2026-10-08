# Undiagnosed Hypertension Manuscript Code

Python code that reproduces the results in *"Comparative Performance of Classical
Statistical and Machine Learning Models in Predicting Undiagnosed Hypertension:
Analysis of the 2022 Rwanda NCD STEPS Survey."*

It derives the undiagnosed hypertension outcome, produces the survey-weighted
descriptives, and trains and evaluates six classifiers (logistic regression,
decision tree, random forest, support vector machine, gradient boosting and a
neural network), including the adjusted odds ratios, reduced models and
calibration reported in the paper.

## Data
Uses the **2022 Rwanda NCD STEPS Survey** dataset (`final dataset_STEP_2022.dta`),
**not included here** — available from the Rwanda Biomedical Centre on reasonable
request. Place the file beside the script or point to it with the `HTN_DATA`
environment variable.

## Run
```bash
pip install pandas numpy scikit-learn statsmodels scipy matplotlib seaborn pyreadstat
HTN_DATA="final dataset_STEP_2022.dta" python analysis.py
```
Class imbalance is handled by class weighting and F1-optimal threshold tuning
(no oversampling); a fixed random seed (42) makes results reproducible.
