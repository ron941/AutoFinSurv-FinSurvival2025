# -*- coding: utf-8 -*- 
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from datetime import timedelta
import matplotlib.pyplot as plt

# ======= Config =======
LABEL_TIME, LABEL_EVENT = "timeDiff", "status"
PARTICIPANT_DATA_PATH = "/raid/ron/FinSurvival_Challenge/participant_data-001/participant_data"
INDEX_EVENTS = ["Borrow", "Deposit", "Repay", "Withdraw"]
OUTCOME_EVENTS = INDEX_EVENTS + ["Liquidated"]
CUTOFF_DATE = "2023-07-01"
BUFFER_DAYS = 30

# ---------- utils ----------
def fit_top10_and_map(Xtr, Xte):
    for c in Xtr.columns:
        top10 = Xtr[c].value_counts(dropna=True).nlargest(10).index
        Xtr[c] = Xtr[c].where(Xtr[c].isin(top10), "Other").fillna("Other")
        Xte[c] = Xte[c].where(Xte[c].isin(top10), "Other").fillna("Other")
    return Xtr, Xte

def one_hot_align(Xtr, Xte):
    dtr = pd.get_dummies(Xtr, drop_first=False)
    dte = pd.get_dummies(Xte, drop_first=False)
    dte = dte.reindex(columns=dtr.columns, fill_value=0)
    return dtr, dte

def preprocess_temporal(df, cutoff_date=CUTOFF_DATE, buffer_days=BUFFER_DAYS):
    """依時間切割 (官方 FinSurvival 標準)"""
    df = df[df[LABEL_TIME] > 0].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    cutoff = pd.Timestamp(cutoff_date)
    buf = timedelta(days=buffer_days)
    train_mask = df["timestamp"] < (cutoff - buf)
    test_mask  = df["timestamp"] >= (cutoff + buf)
    train_df, test_df = df[train_mask], df[test_mask]

    y_time_tr = train_df[LABEL_TIME].astype(float).values
    y_event_tr = train_df[LABEL_EVENT].astype(int).values
    y_time_te = test_df[LABEL_TIME].astype(float).values
    y_event_te = test_df[LABEL_EVENT].astype(int).values

    drop_cols = [LABEL_TIME, LABEL_EVENT, "id", "user", "pool",
                 "Index Event", "Outcome Event", "type", "timestamp"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test  = test_df.drop(columns=drop_cols, errors="ignore")

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    Xtr_cat, Xte_cat = fit_top10_and_map(X_train[cat_cols], X_test[cat_cols])
    dtr_cat, dte_cat = one_hot_align(Xtr_cat, Xte_cat)

    scaler = StandardScaler()
    Xtr_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    Xte_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
    Xtr = pd.concat([Xtr_num, dtr_cat], axis=1)
    Xte = pd.concat([Xte_num, dte_cat], axis=1)
    nz_cols = Xtr.columns[Xtr.var() > 0]
    Xtr = Xtr[nz_cols].astype(np.float32)
    Xte = Xte.reindex(columns=nz_cols, fill_value=0).astype(np.float32)
    return Xtr, Xte, y_time_tr, y_time_te, y_event_tr, y_event_te

# ---------- baseline ----------
def train_baseline(Xtr, Xte, t_tr, t_te, e_tr, e_te, seed=42):
    """使用提供的 baseline 參數"""
    X_tr, X_va, tt_tr, tt_va, ee_tr, ee_va = train_test_split(
        Xtr, t_tr, e_tr, test_size=0.2, random_state=seed, stratify=e_tr
    )
    y_tr = np.where(ee_tr == 1, tt_tr, -tt_tr)
    y_va = np.where(ee_va == 1, tt_va, -tt_va)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    dtest  = xgb.DMatrix(Xte, label=np.where(e_te == 1, t_te, -t_te))

    params = dict(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="gpu_hist",
        device="cuda",
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.1,
        seed=seed,
        verbosity=0,
        max_bin=64,
    )

    bst = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50, verbose_eval=False
    )

    preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    cidx = concordance_index(t_te, -preds, e_te)
    return cidx

# ---------- tuning objective ----------
def objective(trial, Xtr, Xte, t_tr, t_te, e_tr, e_te):
    params = dict(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        device="cuda",
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.05),
        max_depth=trial.suggest_int("max_depth", 4, 6),
        subsample=trial.suggest_float("subsample", 0.75, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 0.9),
        min_child_weight=trial.suggest_int("min_child_weight", 3, 8),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 2.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.05, 0.5),
        max_bin=64,
        seed=42,
        verbosity=0,
    )

    X_tr, X_va, tt_tr, tt_va, ee_tr, ee_va = train_test_split(
        Xtr, t_tr, e_tr, test_size=0.2, random_state=42, stratify=e_tr
    )
    y_signed_tr = np.where(ee_tr == 1, tt_tr, -tt_tr)
    y_signed_va = np.where(ee_va == 1, tt_va, -tt_va)
    dtrain = xgb.DMatrix(X_tr, label=y_signed_tr)
    dvalid = xgb.DMatrix(X_va, label=y_signed_va)

    bst = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    preds = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))
    cidx = concordance_index(t_te, -preds, e_te)
    return cidx

# ---------- main loop ----------
def main():
    results = []
    best_params_all = {}
    pairs = [(i, o) for i in INDEX_EVENTS for o in OUTCOME_EVENTS if i != o]

    log_path = "xgb_cox_refined_log.txt"
    csv_path = "xgb_cox_refined_tuning_results.csv"
    json_path = "xgb_cox_refined_best_params.json"

    # 初始化 log 檔
    with open(log_path, "w") as f:
        f.write("=== XGBoost Cox Refinement Log ===\n\n")

    for idx_ev, out_ev in tqdm(pairs, desc="All Tasks", unit="task"):
        data_dir = os.path.join(PARTICIPANT_DATA_PATH, idx_ev, out_ev)
        train_csv = os.path.join(data_dir, "train.csv")
        if not os.path.exists(train_csv):
            continue

        print(f"\n=== {idx_ev} -> {out_ev} ===")
        df = pd.read_csv(train_csv)
        Xtr, Xte, t_tr, t_te, e_tr, e_te = preprocess_temporal(df)

        # baseline
        base_cidx = train_baseline(Xtr, Xte, t_tr, t_te, e_tr, e_te)

        # tuning
        study = optuna.create_study(direction="maximize")
        with tqdm(total=20, desc=f"{idx_ev}->{out_ev} tuning", unit="trial") as pbar:
            def cb(study, trial): pbar.update(1)
            study.optimize(lambda trial: objective(trial, Xtr, Xte, t_tr, t_te, e_tr, e_te),
                           n_trials=20, callbacks=[cb])

        best_params = study.best_params
        best_cidx = study.best_value
        delta = best_cidx - base_cidx

        # ===== 即時儲存 =====
        result = dict(
            index_event=idx_ev,
            outcome_event=out_ev,
            baseline=base_cidx,
            tuned=best_cidx,
            delta=delta,
            best_params=json.dumps(best_params)
        )
        results.append(result)
        best_params_all[f"{idx_ev}->{out_ev}"] = best_params

        # ✅ 1️⃣ 寫入 txt log（含主要參數）
        param_summary = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in best_params.items()])
        with open(log_path, "a") as f:
            f.write(f"{idx_ev}->{out_ev}: baseline={base_cidx:.4f}, tuned={best_cidx:.4f}, Δ={delta:+.4f}\n")
            f.write(f"    best_params: {param_summary}\n\n")

        # ✅ 2️⃣ 即時更新 CSV
        pd.DataFrame(results).to_csv(csv_path, index=False)

        # ✅ 3️⃣ 即時更新 JSON
        with open(json_path, "w") as f:
            json.dump(best_params_all, f, indent=2)

        print(f"Baseline: {base_cidx:.4f}, Tuned: {best_cidx:.4f}, Δ={delta:+.4f}")

    # === Final summary ===
    res = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(res)
    print(f"Avg baseline: {res['baseline'].mean():.4f}")
    print(f"Avg tuned:    {res['tuned'].mean():.4f}")
    print(f"Avg ΔC-index: {res['delta'].mean():+.4f}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.barh(res["index_event"] + "→" + res["outcome_event"], res["delta"], color="skyblue")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("ΔC-index (Tuned - Baseline)")
    plt.title("Per-task Improvement after Optuna Refinement")
    plt.tight_layout()
    plt.savefig("xgb_refined_tuning_delta.png")
    plt.show()


if __name__ == "__main__":
    main()
