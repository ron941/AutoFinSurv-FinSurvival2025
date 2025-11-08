# -*- coding: utf-8 -*-
"""
xgb_cox_submit_v3.py
支援新的 test_features 路徑：
/raid/ron/FinSurvival_Challenge/participant_data-001/test_features/{IndexEvent}/{OutcomeEvent}/test_features.csv
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import shutil

# ======= Config =======
LABEL_TIME, LABEL_EVENT = "timeDiff", "status"
PARTICIPANT_DATA_PATH = "/raid/ron/FinSurvival_Challenge/participant_data-001/participant_data/"
TEST_FEATURES_PATH = "/raid/ron/FinSurvival_Challenge/participant_data-001/test_features/"
TUNING_RESULT_PATH = "/raid/ron/FinSurvival_Challenge/xgb_cox_refined_tuning_results.csv"
SUBMISSION_DIR = "/raid/ron/FinSurvival_Challenge/my_prediction_submission_xgb_cox_tuned_v3"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

INDEX_EVENTS = ["Borrow", "Deposit", "Repay", "Withdraw"]
OUTCOME_EVENTS = INDEX_EVENTS + ["Liquidated"]
EVENT_PAIRS = [(i, o) for i in INDEX_EVENTS for o in OUTCOME_EVENTS if i != o]


# ---------- utils ----------
def fit_top10_and_map(train_df, test_df, categorical_cols):
    """只處理類別欄位（原版邏輯）"""
    for c in categorical_cols:
        top10 = train_df[c].value_counts(dropna=True).nlargest(10).index
        train_df[c] = train_df[c].where(train_df[c].isin(top10), "Other").fillna("Other")
        test_df[c] = test_df[c].where(test_df[c].isin(top10), "Other").fillna("Other")
    return train_df, test_df


def preprocess(train_df, test_df):
    """與原版相同的特徵處理流程"""
    y_time = train_df[LABEL_TIME].astype(float).values
    y_event = train_df[LABEL_EVENT].astype(int).values

    drop_cols = [LABEL_TIME, LABEL_EVENT, "id", "user", "pool",
                 "Index Event", "Outcome Event", "type", "timestamp"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # 類別與數值欄位區分
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train, X_test = fit_top10_and_map(X_train, X_test, cat_cols)
    dtr = pd.get_dummies(X_train, drop_first=False)
    dte = pd.get_dummies(X_test, drop_first=False)
    dte = dte.reindex(columns=dtr.columns, fill_value=0)

    num_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    Xtr_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),
                           columns=num_cols, index=X_train.index)
    Xte_num = pd.DataFrame(scaler.transform(X_test[num_cols]),
                           columns=num_cols, index=X_test.index)

    Xtr = pd.concat([Xtr_num, dtr.drop(columns=num_cols, errors="ignore")], axis=1)
    Xte = pd.concat([Xte_num, dte.drop(columns=num_cols, errors="ignore")], axis=1)

    nz_cols = Xtr.columns[Xtr.var() > 0]
    Xtr = Xtr[nz_cols].astype(np.float32)
    Xte = Xte.reindex(columns=nz_cols, fill_value=0).astype(np.float32)

    return Xtr, Xte, y_time, y_event


def load_tuned_params(csv_path):
    """載入 tuning 結果，每任務最佳參數"""
    df = pd.read_csv(csv_path)
    tuned = {}
    if "best_params" not in df.columns:
        print("⚠️ CSV 未包含 best_params 欄位，請確認檔案內容。")
        return tuned

    for _, r in df.iterrows():
        key = f"{r['index_event']}->{r['outcome_event']}"
        try:
            tuned[key] = eval(r["best_params"]) if isinstance(r["best_params"], str) else r["best_params"]
        except Exception as e:
            print(f"⚠️ 無法解析參數 {key}: {e}")
    print(f"✅ 成功載入 {len(tuned)} 組 tuned 參數")
    return tuned


def train_and_predict(Xtr, Xte, y_time, y_event, tuned_dict, task_key, seed=42):
    """使用 tuned 參數（若無則 fallback baseline）"""
    y_signed = np.where(y_event == 1, y_time, -y_time)
    dtrain = xgb.DMatrix(Xtr, label=y_signed)
    dtest = xgb.DMatrix(Xte)

    tuned = tuned_dict.get(task_key, {})
    params = dict(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        device="cuda",
        seed=seed,
        verbosity=1,
        max_bin=64,
        # baseline + tuned override
        learning_rate=tuned.get("learning_rate", 0.04),
        max_depth=tuned.get("max_depth", 5),
        subsample=tuned.get("subsample", 0.85),
        colsample_bytree=tuned.get("colsample_bytree", 0.8),
        min_child_weight=tuned.get("min_child_weight", 5),
        reg_lambda=tuned.get("reg_lambda", 1.0),
        reg_alpha=tuned.get("reg_alpha", 0.1),
    )

    print(f"\n🚀 [{task_key}] 使用參數: {params}")
    bst = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dtrain, "train")],
        verbose_eval=100
    )

    preds = -bst.predict(dtest)
    return preds


def main():
    tuned_dict = load_tuned_params(TUNING_RESULT_PATH)

    for index_event, outcome_event in EVENT_PAIRS:
        task_key = f"{index_event}->{outcome_event}"
        print(f"\n===== 處理任務: {task_key} =====")

        train_path = os.path.join(PARTICIPANT_DATA_PATH, index_event, outcome_event, "train.csv")
        test_path = os.path.join(TEST_FEATURES_PATH, index_event, outcome_event, "test_features.csv")

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"⚠️ 缺少資料：{task_key}")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        Xtr, Xte, y_time, y_event = preprocess(train_df, test_df)
        preds = train_and_predict(Xtr, Xte, y_time, y_event, tuned_dict, task_key)

        pred_path = os.path.join(SUBMISSION_DIR, f"{index_event}_{outcome_event}.csv")
        pd.DataFrame(preds).to_csv(pred_path, header=False, index=False)
        print(f"✅ 儲存: {pred_path}")

    # 打包 zip
    zip_path = "/raid/ron/FinSurvival_Challenge/submission_xgb_cox_tuned_v3"
    shutil.make_archive(zip_path, "zip", SUBMISSION_DIR)
    print(f"\n🎯 已完成提交檔: {zip_path}.zip")


if __name__ == "__main__":
    main()
