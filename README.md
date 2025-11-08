# AutoFinSurv – FinSurvival Challenge Final Submission

## Environment
- OS: Ubuntu 22.04
- CUDA: 12.1
- Python: 3.10
- GPU: NVIDIA Titan RTX (24 GB)
- Key Libraries  
  ```
  xgboost==2.1.1  
  optuna==3.6.0  
  scikit-learn==1.5.0  
  lifelines==0.27.8  
  tqdm==4.66  
  matplotlib==3.9  
  pandas==2.2  
  numpy==1.26
  ```

## Reproducibility Checklist
| Item | Setting |
|------|----------|
| Temporal Split | 2023-07-01 ± 30 days |
| Event Pairs | Borrow / Deposit / Repay / Withdraw (+ Liquidated) |
| Trials × Rounds | 20 × 1000 |
| Metric | Concordance Index (C-index) |
| Avg C-index (Dev / Final) | 0.8483 / 0.8490 |

## Usage
```bash
# 1️⃣ Optuna refinement (約 320 GPU-hours)
python xgb_cox_finetuning.py

# 2️⃣ Generate final submission zip
python xgb_cox_submit.py
```

### Output Files
| File | Description |
|------|--------------|
| `xgb_cox_refined_tuning_results.csv` | Each event pair’s best parameters |
| `my_prediction_submission_xgb_cox_tuned_v3.zip` | Final Codabench submission archive |

## Framework Overview
AutoFinSurv is a reproducible GPU-accelerated XGBoost-Cox pipeline with:
- Temporal data splitting for non-leakage training  
- Unified preprocessing (Top-10 categorical mapping + StandardScaler)  
- Optuna Bayesian hyperparameter search  
- Automatic CSV/JSON/TXT logging for full reproducibility  


