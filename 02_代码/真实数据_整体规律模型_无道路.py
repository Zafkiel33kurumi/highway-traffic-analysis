# ===================== 整体规律模型（不分断面） =====================
# 目标：把全部断面的数据合并，学一个"输入条件 → 拥堵概率"的通用函数，
#       而不是给每个断面各训一个、再让用户去挑断面。
# 输入：时段、天气（6 级）、日期类型（工作日/周末/节假日）、车道数
# 输出：拥堵概率 P(车速 < 70 km/h)、每车道流量、平均车速、占有率
#
# 关键设计决定（都有数据依据）：
#   1. 不用 station_id。断面会变成"记住每一条路"，本质是查表不是规律。
#   2. 天气按降水/降雪强度分 6 级。只分 4 级时小雨和暴雨混成一类，效应被平均掉。
#   3. 保留 lane_count。不同车道档的通行能力不同，拥堵率差异明显。
#   4. 概率用分类器 + 等温校准，不用"回归车速再套残差"。残差法在平峰/周末会给出 0%，
#      实际是 9%；分类器直接学 P(拥堵|条件)，等于把条件相同的样本汇总求发生率。

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "01_原始数据" / "波特兰高速_真实数据_多冬季.csv"
MODEL = Path(__file__).resolve().parent / "models" / "真实数据_整体规律模型_无道路.pkl"
SITE_DATA = BASE / "05_可视化系统" / "prediction_data_无道路.js"

FEATURES = ["hour_of_day", "wx", "day_type", "lane_count"]
TARGETS = ["fpl", "avg_speed_kmh", "occupancy_pct"]
CONGESTION_SPEED = 70.0
WEATHERS = ["Clear", "Cloudy", "LightRain", "HeavyRain", "LightSnow", "HeavySnow"]
DAY_TYPES = ["workday", "weekend", "holiday"]
LANES = [2, 3, 4, 5]
RANDOM_STATE = 42


def weather_level(r):
    """按强度分 6 级：降水 1.5 mm/h、降雪 0.5 cm/h 为界。"""
    if r.snow_cm >= 0.5:
        return "HeavySnow"
    if r.snow_cm > 0:
        return "LightSnow"
    if r.precip_mm >= 1.5:
        return "HeavyRain"
    if r.precip_mm > 0.01:
        return "LightRain"
    if r.weather == "Cloudy":
        return "Cloudy"
    return "Clear"


def load():
    df = pd.read_csv(DATA)
    df["date"] = df.timestamp.str[:10]
    df["cong"] = (df.avg_speed_kmh < CONGESTION_SPEED).astype(int)
    df["fpl"] = df.flow_veh_hour / df.lane_count          # 每车道流量
    df["day_type"] = np.where(df.get("is_holiday", 0) == 1, "holiday",
                              np.where(df.is_weekend == 1, "weekend", "workday"))
    df["wx"] = df.apply(weather_level, axis=1)
    return df


def encode(frame):
    return pd.get_dummies(frame[FEATURES], columns=["wx", "day_type"], dtype=float)


def importance_by_input(names, imp):
    total = {"hour_of_day": 0.0, "wx": 0.0, "day_type": 0.0, "lane_count": 0.0}
    for n, v in zip(names, imp):
        key = "wx" if n.startswith("wx_") else ("day_type" if n.startswith("day_type_") else n)
        total[key] = total.get(key, 0.0) + float(v)
    return {k: round(v, 3) for k, v in sorted(total.items(), key=lambda t: -t[1])}


def train():
    df = load()
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    Xtr, Xte = encode(tr), encode(te)

    # --- 概率：分类器 + 等温校准 ---
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                 random_state=RANDOM_STATE, oob_score=True)
    clf.fit(Xtr, tr.cong)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        clf.oob_decision_function_[:, 1], tr.cong.values)
    prob_te = iso.predict(clf.predict_proba(Xte)[:, 1])

    # --- 物理量：多输出回归 ---
    reg = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1,
                                random_state=RANDOM_STATE, oob_score=True)
    reg.fit(Xtr, tr[TARGETS])
    pred = reg.predict(Xte)
    speed_resid = tr.avg_speed_kmh.values - reg.oob_prediction_[:, 1]

    metrics = {
        "flowR2": round(float(r2_score(te.fpl, pred[:, 0])), 3),
        "speedR2": round(float(r2_score(te.avg_speed_kmh, pred[:, 1])), 3),
        "occR2": round(float(r2_score(te.occupancy_pct, pred[:, 2])), 3),
        "auc": round(float(roc_auc_score(te.cong, prob_te)), 3),
        "brier": round(float(brier_score_loss(te.cong, prob_te)), 3),
        "baseRate": round(float(te.cong.mean()), 3),
    }
    print(f"样本 {len(df)}｜训练 {len(tr)}｜测试 {len(te)}"
          f"｜拥堵定义 车速<{CONGESTION_SPEED:.0f} km/h")
    print(f"  每车道流量 R²={metrics['flowR2']:.3f} MAE={mean_absolute_error(te.fpl, pred[:, 0]):.0f}")
    print(f"  车速        R²={metrics['speedR2']:.3f} MAE={mean_absolute_error(te.avg_speed_kmh, pred[:, 1]):.2f}")
    print(f"  占有率      R²={metrics['occR2']:.3f} MAE={mean_absolute_error(te.occupancy_pct, pred[:, 2]):.2f}")
    print(f"  拥堵概率    AUC={metrics['auc']:.3f} Brier={metrics['brier']:.3f}"
          f"  实际拥堵率={metrics['baseRate']:.1%}")

    print("\n【校准检查】预测 vs 实际（工作日 / 周末）：")
    for h in (7, 8, 12, 17, 18):
        row = [f"  {h:2d} 点"]
        for d in ("workday", "weekend"):
            m = ((te.hour_of_day.values == h) & (te.day_type == d).values)
            row.append(f"{d} 预测 {prob_te[m].mean() * 100:5.1f}% 实际 {te.cong.values[m].mean() * 100:5.1f}%")
        print("  ".join(row))

    imp = importance_by_input(list(Xtr.columns), clf.feature_importances_)
    print("\n【特征重要性】", imp)

    print("\n【天气效应 · 工作日高峰 7/8/17/18 点】")
    pk = te[te.day_type.eq("workday") & te.hour_of_day.isin([7, 8, 17, 18])]
    line = []
    for w in WEATHERS:
        s = pk[pk.wx == w]
        if len(s) >= 30:
            line.append(f"{w} {s.cong.mean() * 100:4.1f}%({len(s)})")
    print("  ".join(line))

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"reg": reg, "clf": clf, "iso": iso}, MODEL)
    export_site(df, clf, iso, reg, speed_resid, metrics, imp)
    return clf


def input_effects(grid):
    """各输入单独变化时，拥堵概率能变动多少个百分点（其余输入固定）。"""
    def spread(pairs):
        v = [p for p in pairs]
        return max(v) - min(v)

    wx_s, day_s, hour_s, lane_s = [], [], [], []
    for h in range(24):
        for d in DAY_TYPES:
            for l in LANES:
                wx_s.append(spread([grid[f"{h}|{w}|{d}|{l}"][3] for w in WEATHERS]))
    for h in range(24):
        for w in WEATHERS:
            for l in LANES:
                day_s.append(spread([grid[f"{h}|{w}|{d}|{l}"][3] for d in DAY_TYPES]))
    for w in WEATHERS:
        for d in DAY_TYPES:
            for l in LANES:
                hour_s.append(spread([grid[f"{h}|{w}|{d}|{l}"][3] for h in range(24)]))
    for h in range(24):
        for w in WEATHERS:
            for d in DAY_TYPES:
                lane_s.append(spread([grid[f"{h}|{w}|{d}|{l}"][3] for l in LANES]))

    def stat(v):
        return {"avg": round(sum(v) / len(v), 1), "max": round(max(v), 1)}
    return {"hour_of_day": stat(hour_s), "wx": stat(wx_s),
            "day_type": stat(day_s), "lane_count": stat(lane_s)}


def export_site(df, clf, iso, reg, speed_resid, metrics, imp):
    ref_cap = float(df.fpl.quantile(0.90))            # 常态高峰每车道负荷参考值
    lo, hi = np.percentile(speed_resid, [10, 90])

    lane_stations = {int(k): int(v) for k, v in
                     df.groupby("lane_count").station_id.nunique().items()}

    grid, counts = {}, {}
    for day_type in DAY_TYPES:
        for hour in range(24):
            for w in WEATHERS:
                for lanes in LANES:
                    x = pd.DataFrame([{"hour_of_day": hour, "wx": w, "day_type": day_type,
                                       "lane_count": lanes}])
                    X = pd.get_dummies(x, columns=["wx", "day_type"], dtype=float)
                    X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
                    fpl, speed, occ = reg.predict(X)[0]
                    prob = float(iso.predict(clf.predict_proba(X)[:, 1])[0])
                    k = f"{hour}|{w}|{day_type}|{lanes}"
                    grid[k] = [
                        int(round(fpl)), round(float(speed), 1), round(float(occ), 1),
                        round(prob * 100, 1),
                        round(max(5.0, speed + lo), 1), round(min(130.0, speed + hi), 1),
                    ]
                    counts[k] = int(
                        ((df.hour_of_day == hour) & (df.wx == w) & (df.day_type == day_type)
                         & (df.lane_count == lanes)).sum())

    payload = {
        "meta": {
            "source": "高速公路断面检测数据 + 气象观测数据",
            "period": f"{pd.to_datetime(df.timestamp).min():%Y-%m-%d} ~ "
                      f"{pd.to_datetime(df.timestamp).max():%Y-%m-%d}",
            "rows": int(len(df)),
            "stations": int(df.station_id.nunique()),
            "threshold": CONGESTION_SPEED,
            "refCap": int(round(ref_cap)),
            "weathers": WEATHERS,
            "dayTypes": DAY_TYPES,
            "lanes": LANES,
            "laneStations": lane_stations,
            "metrics": metrics,
            "importance": imp,
            "effects": input_effects(grid),
            "counts": counts,
            "weatherMeans": {w: round(float(v), 1) for w, v in
                             df.groupby("wx").avg_speed_kmh.mean().items()},
            "weatherCong": {w: round(float(v) * 100, 1) for w, v in
                            df.groupby("wx").cong.mean().items()},
            "dayTypeCong": {d: round(float(v) * 100, 1) for d, v in
                            df.groupby("day_type").cong.mean().items()},
            "dayTypeFlow": {d: int(v) for d, v in df.groupby("day_type").fpl.mean().items()},
        },
        "grid": grid,
    }
    SITE_DATA.parent.mkdir(exist_ok=True)
    SITE_DATA.write_text("window.TRAFFIC = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                         encoding="utf-8")
    print(f"\n网页数据已导出：{SITE_DATA}（{len(grid)} 个组合，{SITE_DATA.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    train()
