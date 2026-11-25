# ===================== 拥堵概率模型（连续概率版） =====================
# 设计要点：
#   1. 真值来自平均车速：avg_speed_kmh < SPEED_THRESHOLD 记为"堵"
#   2. 输出连续概率 P(堵)，不做硬标签
#   3. 两种输入模式：
#        模式A 情景推演：只用事前可得的上下文（时间/天气/车道/事故）
#        模式B 实时判别：额外使用传感器实时值（占有率/流量）
#   4. 概率经等温校准，保证"说 70% 就真有约 70% 的样本是堵"

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ===================== 1. 配置 =====================
SPEED_THRESHOLD = 60.0          # 车速低于该值记为拥堵，可按需调整
RANDOM_STATE = 42
TEST_SIZE = 0.3

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "01_原始数据" / "高速公路交通流传感器数据集.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

SCENARIO_COLS = ["weather", "hour_sin", "hour_cos", "weekday",
                 "is_weekend", "lane_count", "freeway", "incident_flag"]
SENSOR_COLS = ["occupancy_pct", "flow_veh_5min"]

# ===================== 2. 读取与特征构造 =====================
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
# 小时是循环变量，用 sin/cos 编码，避免 0 点与 23 点被当成最远
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df = df.dropna(subset=["avg_speed_kmh"])

# 真值：由车速定义（不进入特征，避免标签泄漏）
df["is_congested"] = (df["avg_speed_kmh"] < SPEED_THRESHOLD).astype(int)

print("=" * 62)
print(f"样本量 {len(df)} ｜ 车速阈值 {SPEED_THRESHOLD} km/h ｜ 拥堵占比 {df['is_congested'].mean():.1%}")
print("=" * 62)


def build_model(kind="hgb"):
    """返回未训练的基学习器。hgb 拟合能力强，lr 可解释性好。"""
    if kind == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    return HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, random_state=RANDOM_STATE)


def evaluate(name, y_true, prob):
    """连续概率的评估：AUC / PR-AUC / Brier / LogLoss。"""
    return {
        "模式": name,
        "AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "LogLoss": log_loss(y_true, prob),
    }


def reliability_table(y_true, prob, bins=10):
    """校准表：预测概率 vs 实际拥堵率，越接近对角线说明概率越可信。"""
    t = pd.DataFrame({"p": prob, "y": y_true})
    t["bin"] = pd.cut(t["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    g = t.groupby("bin", observed=True).agg(
        样本数=("y", "size"), 预测概率=("p", "mean"), 实际拥堵率=("y", "mean"))
    return g.round(3)


def fit_calibrated(X, y, kind="hgb"):
    """先切分，再在训练集内部做等温校准，测试集完全不参与。"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    model = CalibratedClassifierCV(build_model(kind), method="isotonic", cv=3)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    return model, prob, y_te


def to_frame(cols):
    X = df[cols].copy()
    return pd.get_dummies(X, columns=[c for c in ["weather", "freeway"] if c in cols],
                          drop_first=True)


# ===================== 3. 训练两种模式 =====================
results, models, reliab = [], {}, {}

for name, cols in [("模式A 情景推演", SCENARIO_COLS),
                   ("模式B 实时判别", SCENARIO_COLS + SENSOR_COLS)]:
    X = to_frame(cols)
    m, prob, y_te = fit_calibrated(X, df["is_congested"])
    models[name] = (m, list(X.columns))
    results.append(evaluate(name, y_te, prob))
    reliab[name] = (y_te, prob)
    # 对照：逻辑回归（可解释）
    m_lr, prob_lr, y_lr = fit_calibrated(X, df["is_congested"], kind="lr")
    results.append(evaluate(name.replace("模式", "模式") + " · Logistic", y_lr, prob_lr))

print("\n【连续概率模型评估】")
print(pd.DataFrame(results).set_index("模式").round(3).to_string())
print(f"\n基准参照：全猜'堵'的 Brier = {df['is_congested'].mean() * (1 - df['is_congested'].mean()):.3f}"
      f"；全猜'不堵'的 Brier = {df['is_congested'].mean():.3f}")

print("\n【概率校准表 · 模式B】")
print(reliability_table(*reliab["模式B 实时判别"]).to_string())
print("\n【概率校准表 · 模式A】")
print(reliability_table(*reliab["模式A 情景推演"]).to_string())

# ===================== 4. 概率 → 建议 =====================
ADVICE = [
    (0.10, "路况畅通，可正常出行。"),
    (0.30, "路况基本正常，建议按计划出行。"),
    (0.50, "存在拥堵可能，建议提前 15~20 分钟出发。"),
    (0.70, "拥堵概率较高，建议错峰或选择替代路线。"),
    (1.01, "拥堵概率很高，建议推迟出行或改乘公共交通。"),
]
WEATHER_TIP = {
    "Rain": "雨天路滑，请减速慢行、保持车距。",
    "Snow": "雪天路面湿滑，建议加装防滑链或避免驾车。",
    "Fog": "能见度低，请开启雾灯并降低车速。",
    "Clear": "",
}


def advice_for(prob, weather, incident_flag):
    base = next(text for th, text in ADVICE if prob < th)
    extra = WEATHER_TIP.get(weather, "")
    if incident_flag == 1:
        extra += "该路段当前有交通事件，请注意绕行。"
    return base + extra


# ===================== 5. 推理接口 =====================
def predict_congestion_probability(weather, hour, weekday, lane_count, freeway,
                                   incident_flag, occupancy_pct=None,
                                   flow_veh_5min=None):
    """
    返回连续拥堵概率。
    occupancy_pct / flow_veh_5min 若提供，自动走模式B（更准）；否则走模式A。
    """
    mode = "模式B 实时判别" if occupancy_pct is not None and flow_veh_5min is not None \
        else "模式A 情景推演"
    model, cols = models[mode]

    row = {
        "weather": weather, "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24), "weekday": weekday,
        "is_weekend": int(weekday >= 5), "lane_count": lane_count,
        "freeway": freeway, "incident_flag": incident_flag,
        "occupancy_pct": occupancy_pct, "flow_veh_5min": flow_veh_5min,
    }
    X_new = pd.get_dummies(pd.DataFrame([row]),
                           columns=["weather", "freeway"], drop_first=True)
    X_new = X_new.reindex(columns=cols, fill_value=0)

    p = float(model.predict_proba(X_new)[0, 1])
    return {
        "拥堵概率": round(p * 100, 1),
        "概率区间": "低" if p < 0.3 else "中" if p < 0.6 else "高",
        "使用模式": mode,
        "天气提示": WEATHER_TIP.get(weather, ""),
        "出行建议": advice_for(p, weather, incident_flag),
    }


# ===================== 6. 演示 =====================
if __name__ == "__main__":
    joblib.dump({"models": models, "speed_threshold": SPEED_THRESHOLD},
                MODEL_DIR / "拥堵概率模型.pkl")

    demos = [
        ("工作日早高峰 · 晴天 · 无事故", dict(weather="Clear", hour=8, weekday=2,
                                     lane_count=3, freeway="I-80", incident_flag=0)),
        ("工作日晚高峰 · 下雨 · 有事故", dict(weather="Rain", hour=18, weekday=2,
                                      lane_count=2, freeway="I-405", incident_flag=1,
                                      occupancy_pct=58, flow_veh_5min=520)),
        ("周末午后 · 晴天 · 无事故", dict(weather="Clear", hour=14, weekday=6,
                                   lane_count=4, freeway="US-101", incident_flag=0,
                                   occupancy_pct=18, flow_veh_5min=210)),
        ("夜间 · 大雪 · 无事故", dict(weather="Snow", hour=23, weekday=4,
                                lane_count=2, freeway="SR-99", incident_flag=0,
                                occupancy_pct=70, flow_veh_5min=600)),
    ]
    rows = []
    for title, kw in demos:
        r = predict_congestion_probability(**kw)
        rows.append({"场景": title, "拥堵概率%": r["拥堵概率"],
                     "区间": r["概率区间"], "模式": r["使用模式"],
                     "建议": r["出行建议"]})
    print("\n【场景演示】")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n模型已保存：{MODEL_DIR / '拥堵概率模型.pkl'}")
