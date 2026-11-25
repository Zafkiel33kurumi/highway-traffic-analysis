# ===================== 车速预测 · 容量因子模型 =====================
# 结构：需求(时段) → 容量(车道数×天气×事件) → BPR 速度 → 拥堵概率
#
# 各参数的来源，逐项标明：
#   · 需求曲线 g(hour)         → 用本地数据标定
#   · 自由流速度 / α / β        → 用本地数据标定（非线性最小二乘）
#   · 基本通行能力 C0           → HCM 规范值（2200 veh/h/车道）
#   · 天气容量折减 φ_weather    → 文献经验值（本地数据无约束力，见消融输出）
#   · 事件折减 φ_incident(N)    → 按"一条车道受阻、剩余车道承担"推导
#   · 车道效率 φ_lane           → HCM 双车道断面折减
#
# 之所以把容量写成显式变量：天气/车道/事件只通过饱和度 q/C 影响车速，
# 加法权重表达不了这种"平时无感、饱和时放大"的效应。

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ===================== 1. 配置 =====================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "01_原始数据" / "高速公路交通流传感器数据集.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CONGESTION_SPEED = 60.0          # 低于该车速视为拥堵

# HCM：基本路段单车道通行能力 2200 veh/h → 换算到 5 分钟
CAPACITY_PER_LANE_5MIN = 2200 / 12          # 183.33 veh/5min/车道

# 天气容量折减系数（文献经验值，非本地数据标定）
# 取值区间参考 HCM 与 FHWA 不良天气通行能力研究：
#   小雨 0.95~0.98、中雨 0.88~0.92、大雨 0.80~0.85
#   小雪 0.80~0.85、中雪 0.70~0.78、大雪 0.60~0.70
#   雾（能见度<200m）0.80~0.88
# 本地数据仅含 4 种天气，其余取中间值备查
WEATHER_FACTOR = {
    "Clear": 1.00, "Fog": 0.85, "Rain": 0.92, "Snow": 0.75,
    "Light Rain": 0.96, "Moderate Rain": 0.90, "Heavy Rain": 0.82,
    "Light Snow": 0.83, "Moderate Snow": 0.75, "Heavy Snow": 0.65,
    "Haze": 0.90, "Mist": 0.90, "Drizzle": 0.95,
    "Thunderstorm": 0.70, "Unknown": 1.00,
}

# 车道效率：2 车道断面受大型车影响更显著（HCM 双车道折减）
LANE_EFFICIENCY = {2: 0.95, 3: 1.00, 4: 1.00, 5: 1.00}


def lane_efficiency(lanes):
    return LANE_EFFICIENCY.get(int(lanes), 1.0)


def weather_factor(weather):
    return WEATHER_FACTOR.get(weather, 1.00)


def incident_factor(lanes):
    """事件折减：假定占用 1 条车道，容量按剩余车道比例承担。
    2 车道受阻 → 0.50；3 车道 → 0.67；4 车道 → 0.75；5 车道 → 0.80。
    这天然表达了"同样一起事故，车道越多影响越小"。
    """
    return (int(lanes) - 1) / int(lanes)


def capacity(lanes, weather, incident_flag):
    """有效容量（veh/5min），三项相乘。"""
    cap = CAPACITY_PER_LANE_5MIN * int(lanes) * lane_efficiency(lanes) * weather_factor(weather)
    if int(incident_flag) == 1:
        cap *= incident_factor(lanes)
    return cap


def bpr_speed(flow, lanes, weather, incident_flag, v_free, alpha, beta):
    ratio = np.clip(np.asarray(flow, float) / np.maximum(capacity(lanes, weather, incident_flag), 1e-6), 0, None)
    return v_free / (1.0 + alpha * ratio ** beta)


# ===================== 2. 读取数据 =====================
df = pd.read_csv(DATA_PATH)
df["lane_count"] = df["lane_count"].astype(int)
df["incident_flag"] = df["incident_flag"].astype(int)
df["hour_of_day"] = df["hour_of_day"].astype(int)
df = df.dropna(subset=["avg_speed_kmh", "flow_veh_5min", "occupancy_pct"])

print("=" * 66)
print(f"样本量 {len(df)}｜拥堵定义：车速 < {CONGESTION_SPEED:.0f} km/h｜实际拥堵占比 {np.mean(df.avg_speed_kmh < CONGESTION_SPEED):.1%}")
print("=" * 66)

idx = np.arange(len(df))
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(idx)
n_test = int(len(df) * TEST_SIZE)
test_idx, train_idx = idx[:n_test], idx[n_test:]
# 说明：timestamp 已被打散，无法做时间外推验证，此处用随机划分并在报告中注明

# ===================== 3. L1 需求模型：时段 → 流量 =====================
demand = df.iloc[train_idx].groupby("hour_of_day")["flow_veh_5min"].agg(["mean", "std", "count"])
demand.columns = ["q_mean", "q_std", "n"]


def predict_demand(hour):
    """时段 → 期望流量。用训练集的条件均值，24 个取值。"""
    return float(demand.loc[int(hour), "q_mean"])


hour_r2 = 1 - ((df.flow_veh_5min - df.hour_of_day.map(demand.q_mean)) ** 2).sum() / \
             ((df.flow_veh_5min - df.flow_veh_5min.mean()) ** 2).sum()
print(f"\n【L1 需求】时段 → 流量：R² = {hour_r2:.4f}")
print("  高峰时段(q̂≈520):", [h for h in demand.index if demand.loc[h, 'q_mean'] > 400])
print("  平峰时段(q̂≈240):", [h for h in demand.index if demand.loc[h, 'q_mean'] <= 400][:8], "...")

# ===================== 4. L3 速度模型：BPR 参数标定 =====================
tr = df.iloc[train_idx]
te = df.iloc[test_idx]


def resid(p, d):
    v_free, alpha, beta = p
    cap = np.array([capacity(l, w, i) for l, w, i in zip(d.lane_count, d.weather, d.incident_flag)])
    ratio = np.clip(d.flow_veh_5min.values / cap, 0, None)
    return v_free / (1 + alpha * ratio ** beta) - d.avg_speed_kmh.values


fit = least_squares(resid, x0=[110, 0.3, 4.0], args=(tr,),
                    bounds=([60, 1e-3, 0.5], [200, 50, 15]))
V_FREE, ALPHA, BETA = fit.x
print(f"\n【L3 速度】BPR 标定结果：V_free = {V_FREE:.1f} km/h，α = {ALPHA:.3f}，β = {BETA:.2f}")


def speed_array(d, use_actual_flow=True, flow=None, use_weather=True, use_incident=True):
    f = d.flow_veh_5min.values if use_actual_flow else flow
    v = np.empty(len(d))
    for k, (l, w, i) in enumerate(zip(d.lane_count, d.weather, d.incident_flag)):
        cap = capacity(l, w if use_weather else "Clear", i if use_incident else 0)
        r = max(float(f[k]), 0) / max(cap, 1e-6)
        v[k] = V_FREE / (1 + ALPHA * r ** BETA)
    return v


def metrics(y, yhat):
    return {"R²": 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "RMSE": float(np.sqrt(((y - yhat) ** 2).mean())),
            "MAE": float(np.abs(y - yhat).mean())}


# 实时模式：已知流量
v_real = speed_array(te)
# 情景模式：只知道时段，流量走 L1
q_hat = te.hour_of_day.map(demand.q_mean).values
v_scen = speed_array(te, use_actual_flow=False, flow=q_hat)

print("\n【评估 · 测试集】")
v_hour_base = te.hour_of_day.map(tr.groupby("hour_of_day").avg_speed_kmh.mean()).values
coef_lin = np.polyfit(tr.flow_veh_5min, tr.avg_speed_kmh, 1)
v_lin = np.polyval(coef_lin, te.flow_veh_5min.values)
rows_eval = [("基线1：仅时段均值", v_hour_base),
             ("基线2：流量线性回归", v_lin),
             ("本模型：情景模式(仅时段)", v_scen),
             ("本模型：实时模式(已知流量)", v_real)]
for name, pred in rows_eval:
    m = metrics(te.avg_speed_kmh.values, pred)
    print(f"  {name:24s} R²={m['R²']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")

print("\n  分车道看误差（情景模式）：")
for lane in sorted(te.lane_count.unique()):
    msk = (te.lane_count == lane).values
    print(f"    {lane} 车道  实测均值 {te.avg_speed_kmh.values[msk].mean():5.2f}  "
          f"模型预测 {v_scen[msk].mean():5.2f}  "
          f"RMSE {np.sqrt(((te.avg_speed_kmh.values[msk] - v_scen[msk]) ** 2).mean()):5.2f}")

# 残差 → 概率尺度
resid_sd = float(np.std(v_scen - te.avg_speed_kmh.values))
print(f"  情景模式残差标准差 = {resid_sd:.2f} km/h（用于把点预测转成概率分布）")

# ===================== 5. L4 连续拥堵概率 =====================
def congestion_probability(v_hat, sigma=None):
    """P(v < 60)，把点预测展开成正态分布后取左尾概率。"""
    s = resid_sd if sigma is None else sigma
    return norm.cdf((CONGESTION_SPEED - np.asarray(v_hat, float)) / s)


y_cong = (te.avg_speed_kmh.values < CONGESTION_SPEED).astype(int)
p_cong = congestion_probability(v_scen)
from sklearn.metrics import roc_auc_score, brier_score_loss
print(f"\n【L4 概率】AUC = {roc_auc_score(y_cong, p_cong):.4f}｜Brier = {brier_score_loss(y_cong, p_cong):.4f}")
print(f"  参照：全猜'不堵'的 Brier = {(1 - y_cong.mean()) * y_cong.mean():.4f}")

cal = pd.DataFrame({"p": p_cong, "y": y_cong})
cal["bin"] = pd.cut(cal.p, bins=np.linspace(0, 1, 6), include_lowest=True)
print("\n  校准表（预测概率 vs 实际拥堵率）：")
print(cal.groupby("bin", observed=True).agg(样本数=("y", "size"), 预测概率=("p", "mean"),
                                            实际拥堵率=("y", "mean")).round(3).to_string())

# ===================== 6. 消融：天气/事件因子到底贡献了多少 =====================
print("\n【消融实验】逐步去掉容量因子，看 R² 怎么变")
abl = []
for label, uw, ui in [("完整模型", True, True), ("去掉天气因子", False, True),
                      ("去掉事件因子", True, False), ("两个都去掉", False, False)]:
    pred = speed_array(te, use_actual_flow=False, flow=q_hat, use_weather=uw, use_incident=ui)
    m = metrics(te.avg_speed_kmh.values, pred)
    abl.append({"模型": label, "R²": round(m["R²"], 4), "RMSE": round(m["RMSE"], 2)})
print(pd.DataFrame(abl).to_string(index=False))
print("  注意：去掉因子后 R² 反而略升 —— 这是『数据里没有容量效应』的直接体现。")
print("        模型越贴近物理机理，在这份数据上的拟合反而越差；")
print("        φ 的作用不是提高拟合度，而是让模型能回答'换条车道 / 换个天气会怎样'。")

# 只在高饱和子集上看差异（这才是容量因子该显效的区间）
hi = te[te.flow_veh_5min >= 480]
if len(hi) > 50:
    v_hi_with = speed_array(hi, use_actual_flow=False, flow=hi.flow_veh_5min.values, use_weather=True)
    v_hi_without = speed_array(hi, use_actual_flow=False, flow=hi.flow_veh_5min.values, use_weather=False)
    print(f"\n  高饱和子集(n={len(hi)})实测车速均值 {hi.avg_speed_kmh.mean():.1f} km/h；"
          f"模型含天气因子 {v_hi_with.mean():.1f}，不含 {v_hi_without.mean():.1f}")
    real = hi.groupby("weather").avg_speed_kmh.mean().round(2)
    print("  同一子集里数据实测的车速（几乎无差别）：")
    print("   ", real.to_dict())

# ===================== 7. 场景响应表 =====================
print("\n【响应表】固定需求 q=520 辆/5min（晚高峰水平）")
rows = []
from scipy.stats import norm as _norm
for lanes in [2, 3, 4, 5]:
    for w in ["Clear", "Fog", "Rain", "Snow"]:
        for inc in [0, 1]:
            C = capacity(lanes, w, inc)
            r = 520 / C
            v = V_FREE / (1 + ALPHA * r ** BETA)
            p = float(_norm.cdf((CONGESTION_SPEED - v) / resid_sd))
            rows.append({"车道": lanes, "天气": w, "事件": "有" if inc else "无",
                         "容量": round(C, 0), "饱和度": round(r, 2),
                         "车速": round(v, 1), "拥堵概率": f"{p:.1%}"})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
print("\n  2车道晴天(C=348) 与 5车道雪天(C=688)：前者明显更堵 —— 车道数与天气是相乘关系，不是互相抵消。")

# ===================== 8. 对外接口 =====================
def predict_traffic(hour, weather="Clear", lane_count=3, incident_flag=0,
                    flow_veh_5min=None):
    """
    输入：时段 + 天气 + 车道数 + 事件（可选实时流量）
    输出：流量、容量、饱和度、车速、占有率、拥堵概率、建议
    """
    q = float(flow_veh_5min) if flow_veh_5min is not None else predict_demand(hour)
    C = capacity(lane_count, weather, incident_flag)
    r = q / C
    v = V_FREE / (1 + ALPHA * r ** BETA)
    p = float(norm.cdf((CONGESTION_SPEED - v) / resid_sd))
    # 占有率与密度成正比：occ ≈ a + b·(q/v)
    occ = float(occ_coef[0] + occ_coef[1] * (q / max(v, 1e-6)))
    level = "畅通" if p < 0.15 else "轻度拥堵" if p < 0.4 else "中度拥堵" if p < 0.7 else "严重拥堵"
    return {
        "需求流量": round(q, 0), "有效容量": round(C, 0), "饱和度": round(r, 2),
        "预测车速": round(v, 1), "预测占有率": round(occ, 1),
        "拥堵概率": round(p * 100, 1), "拥堵等级": level,
        "数据来源": "实时流量" if flow_veh_5min is not None else "时段推演",
    }


occ_coef = np.polyfit(df.iloc[train_idx].flow_veh_5min / df.iloc[train_idx].avg_speed_kmh,
                      df.iloc[train_idx].occupancy_pct, 1)

if __name__ == "__main__":
    print("\n【接口演示】predict_traffic(hour=18, weather='Snow', lane_count=3, incident_flag=1)")
    for kw in [dict(hour=18, weather="Clear", lane_count=3),
               dict(hour=18, weather="Snow", lane_count=3),
               dict(hour=18, weather="Snow", lane_count=3, incident_flag=1),
               dict(hour=18, weather="Snow", lane_count=5, incident_flag=1),
               dict(hour=3, weather="Snow", lane_count=2, incident_flag=1),
               dict(hour=18, weather="Rain", lane_count=2, flow_veh_5min=560)]:
        print("  ", kw, "→", predict_traffic(**kw))

    joblib.dump({"v_free": V_FREE, "alpha": ALPHA, "beta": BETA,
                 "resid_sd": resid_sd, "demand": demand,
                 "occ_coef": occ_coef, "congestion_speed": CONGESTION_SPEED},
                MODEL_DIR / "容量因子模型.pkl")
    print(f"\n模型已保存：{MODEL_DIR / '容量因子模型.pkl'}")
