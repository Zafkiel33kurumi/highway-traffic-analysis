# ===================== 对"拥堵概率模型_连续概率版.py"的严格验证 =====================
# 目的：用交叉验证 + 多种指标，回答"这个模型到底准不准"
# 说明：正样本只占 6%，所以"准确率"会被负样本稀释，必须与基线并列看

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

SPEED_THRESHOLD = 60.0
SEEDS = [42, 7]
FOLDS = 5

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "01_原始数据" / "高速公路交通流传感器数据集.csv"

SCENARIO_COLS = ["weather", "hour_sin", "hour_cos", "weekday",
                 "is_weekend", "lane_count", "freeway", "incident_flag"]
SENSOR_COLS = ["occupancy_pct", "flow_veh_5min"]


def load():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.dropna(subset=["avg_speed_kmh"])
    df["is_congested"] = (df["avg_speed_kmh"] < SPEED_THRESHOLD).astype(int)
    return df


def design(df, cols):
    X = pd.get_dummies(df[cols], columns=["weather", "freeway"], drop_first=True)
    return X.astype(float)


def make_model():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_depth=8, learning_rate=0.06,
                                       max_iter=250, l2_regularization=1.0,
                                       random_state=42),
        method="isotonic", cv=3)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "准确率": accuracy_score(y, pred),
        "平衡准确率": balanced_accuracy_score(y, pred),
        "精确率(堵)": precision_score(y, pred, zero_division=0),
        "召回率(堵)": recall_score(y, pred, zero_division=0),
        "F1(堵)": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
        "_cm": (tn, fp, fn, tp),
    }


df = load()
y = df["is_congested"].values
print("=" * 70)
print(f"样本 {len(df)} ｜ 车速阈值 {SPEED_THRESHOLD} ｜ 拥堵 {y.sum()} 条 ({y.mean():.1%})")
print("=" * 70)

# ---------- 基线 ----------
print("\n【基线：不用任何模型能做到多少】")
maj = np.zeros_like(y)
print(f"  1) 全部猜'不堵'      准确率 {accuracy_score(y, maj):.3f}   ← 这是准确率的错觉来源")
print(f"  2) 全部猜'堵'        准确率 {accuracy_score(y, 1 - maj):.3f}")
occ_auc = roc_auc_score(y, df["occupancy_pct"])            # 占有率越高越堵
flow_auc = roc_auc_score(y, df["flow_veh_5min"])
print(f"  3) 只用占有率一条规则  AUC {occ_auc:.3f}  PR-AUC {average_precision_score(y, df['occupancy_pct']):.3f}")
print(f"  4) 只用流量一条规则    AUC {flow_auc:.3f}  PR-AUC {average_precision_score(y, df['flow_veh_5min']):.3f}")

# ---------- 交叉验证 ----------
summary, oof_store = {}, {}
for name, cols in [("模式A 情景推演", SCENARIO_COLS),
                   ("模式B 实时判别", SCENARIO_COLS + SENSOR_COLS)]:
    X = design(df, cols)
    runs = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        p = cross_val_predict(make_model(), X, y, cv=skf,
                              method="predict_proba", n_jobs=1)[:, 1]
        runs.append(metrics(y, p))
        oof_store[(name, seed)] = p
    keys = [k for k in runs[0] if not k.startswith("_")]
    summary[name] = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs]))
                     for k in keys}
    summary[name]["_cm"] = runs[0]["_cm"]
    summary[name]["_runs"] = runs

print(f"\n【{FOLDS} 折交叉验证 × {len(SEEDS)} 次随机种子，均值 ± 标准差】")
rows = []
for name, d in summary.items():
    row = {"模式": name}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        m, s = v
        row[k] = f"{m:.3f}±{s:.3f}"
    rows.append(row)
print(pd.DataFrame(rows).set_index("模式").to_string())

print("\n【混淆矩阵（阈值 0.5，第一个种子，全样本合并）】")
for name, d in summary.items():
    tn, fp, fn, tp = d["_cm"]
    print(f"  {name}: 真堵被找到 {tp}/{tp + fn}，误报 {fp}，漏报 {fn}，真不堵 {tn}")

# ---------- 阈值扫描 ----------
print("\n【模式B 阈值扫描：改阈值只会换来换去，不会凭空变准】")
p = oof_store[("模式B 实时判别", SEEDS[0])]
sweep = []
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (p >= thr).astype(int)
    sweep.append({"阈值": thr, "准确率": round(accuracy_score(y, pred), 3),
                  "精确率": round(precision_score(y, pred, zero_division=0), 3),
                  "召回率": round(recall_score(y, pred, zero_division=0), 3),
                  "F1": round(f1_score(y, pred, zero_division=0), 3),
                  "预测为堵的条数": int(pred.sum())})
print(pd.DataFrame(sweep).to_string(index=False))

# ---------- 模式A 的概率是否几乎恒定 ----------
pA = oof_store[("模式A 情景推演", SEEDS[0])]
print("\n【模式A 输出的概率分布】")
print(f"  最小值 {pA.min():.3f} / 中位数 {np.median(pA):.3f} / 最大值 {pA.max():.3f} / 标准差 {pA.std():.4f}")
print(f"  不同取值的个数：{len(np.unique(np.round(pA, 3)))}（样本数 {len(pA)}）")

# ---------- 不同车速阈值的敏感性 ----------
print("\n【更换车速阈值对拥堵占比的影响】")
for th in [60, 65, 70]:
    yy = (df["avg_speed_kmh"] < th).astype(int).values
    Xb = design(df, SCENARIO_COLS + SENSOR_COLS)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEEDS[0])
    pp = cross_val_predict(make_model(), Xb, yy, cv=skf,
                           method="predict_proba", n_jobs=1)[:, 1]
    print(f"  阈值 {th}: 拥堵占比 {yy.mean():.1%}  AUC {roc_auc_score(yy, pp):.3f}  "
          f"PR-AUC {average_precision_score(yy, pp):.3f}  准确率@0.5 {accuracy_score(yy, (pp >= .5)):.3f}")
