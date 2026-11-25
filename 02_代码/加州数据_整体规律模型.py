# ===================== 加州数据 · 整体规律模型（不分断面） =====================
# 目标：把 500 个断面的数据合并，学一个"输入条件 → 拥堵概率"的通用函数，
#       而不是给每个断面各训一个、再让用户去挑断面。
# 输入：时段、天气（6 级）、日期类型（工作日/周末/节假日）、车道数、道路等级
# 输出：拥堵概率 P(车速 < 70 km/h)、每车道流量、平均车速、占有率
#
# 数据：加州高速公路断面检测数据（2025-01-01 ~ 2025-04-08，5 条干线、500 个断面）
#
# 关键设计决定（都有数据依据）：
#   1. 不用 station_id。断面会变成"记住 500 条路"，本质是查表不是规律。
#   2. 天气按等级归并到页面统一的 6 档：晴→Clear、雾→Cloudy、雨→LightRain、雪→LightSnow。
#      数据里没有降水强度，大雨/大雪两档没有样本，页面会提示"外推"。
#   3. 保留 lane_count。不同车道档的断面通行能力不同，拥堵率差异明显。
#   4. 概率用分类器 + 等温校准，不用"回归车速再套残差"，避免平峰/周末给出 0%。
#   5. 道路等级（road）作为普通类别特征进模型，不由前端乘固定倍数。
#      数据里没有坐标，按所属干线的功能分 3 档：
#        city  = I-5 / I-80 / I-405（穿越城市核心的州际干线）
#        urban = US-101（城市快速路）
#        rural = SR-99（州道，穿城际与农业区）
#      这是粗粒度代理，不是因果变量。

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
DATA = BASE / "01_原始数据" / "高速公路交通流传感器数据集.csv"
MODEL = Path(__file__).resolve().parent / "models" / "加州数据_整体规律模型.pkl"
SITE_DATA = BASE / "05_可视化系统" / "prediction_data.js"

FEATURES = ["hour_of_day", "wx", "day_type", "lane_count", "road"]
TARGETS = ["fpl", "avg_speed_kmh", "occupancy_pct"]
CONGESTION_SPEED = 70.0
WEATHERS = ["Clear", "Cloudy", "LightRain", "HeavyRain", "LightSnow", "HeavySnow"]
DAY_TYPES = ["workday", "weekend", "holiday"]
LANES = [2, 3, 4, 5]
ROADS = ["city", "urban", "rural"]
ROAD_LABELS = {"city": "市中心", "urban": "城区", "rural": "城乡"}
# 干线 → 道路等级
ROAD_BY_FREEWAY = {"I-5": "city", "I-80": "city", "I-405": "city",
                   "US-101": "urban", "SR-99": "rural"}
# 天气 → 页面 6 档
WX_MAP = {"Clear": "Clear", "Fog": "Cloudy", "Rain": "LightRain", "Snow": "LightSnow"}
# 数据覆盖期内的联邦假日
HOLIDAYS = {"2025-01-01", "2025-01-20", "2025-02-17"}
RANDOM_STATE = 42


def load():
    df = pd.read_csv(DATA)
    df["ts"] = pd.to_datetime(df.timestamp)
    df["date"] = df.ts.dt.strftime("%Y-%m-%d")
    df["hour_of_day"] = df.ts.dt.hour
    df["weekday"] = df.ts.dt.weekday
    df["cong"] = (df.avg_speed_kmh < CONGESTION_SPEED).astype(int)
    df["fpl"] = df.flow_veh_5min * 12 / df.lane_count      # 5 分钟流量 → 每小时每车道
    df["day_type"] = np.where(df.date.isin(HOLIDAYS), "holiday",
                              np.where(df.weekday < 5, "workday", "weekend"))
    df["wx"] = df.weather.map(WX_MAP)
    df["road"] = df.freeway.map(ROAD_BY_FREEWAY)
    df = df.dropna(subset=["wx", "road"]).reset_index(drop=True)
    return df


def encode(frame):
    return pd.get_dummies(frame[FEATURES], columns=["wx", "day_type", "road"], dtype=float)


def importance_by_input(names, imp):
    total = {"hour_of_day": 0.0, "wx": 0.0, "day_type": 0.0, "lane_count": 0.0, "road": 0.0}
    for n, v in zip(names, imp):
        key = "wx" if n.startswith("wx_") else ("day_type" if n.startswith("day_type_")
              else ("road" if n.startswith("road_") else n))
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
    print(f"样本 {len(df)}｜断面 {df.station_id.nunique()}｜训练 {len(tr)}｜测试 {len(te)}"
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

    print("\n【分道路等级评估】")
    for r in ROADS:
        m = (te.road == r).values
        if m.sum() == 0:
            continue
        print(f"  {ROAD_LABELS[r]:4s} 样本 {m.sum():6d}  AUC={roc_auc_score(te.cong.values[m], prob_te[m]):.3f}"
              f"  车速MAE={mean_absolute_error(te.avg_speed_kmh.values[m], pred[m, 1]):.2f}"
              f"  实际拥堵率={te.cong.values[m].mean():5.1%}")

    print("\n【天气效应 · 同等时段下按道路等级切开】工作日高峰 7/8/17/18 点")
    pk = te[te.day_type.eq("workday") & te.hour_of_day.isin([7, 8, 17, 18])]
    for r in ROADS:
        sub = pk[pk.road == r]
        if len(sub) < 50:
            continue
        line = [f"  {ROAD_LABELS[r]:4s}"]
        for w in WEATHERS:
            s = sub[sub.wx == w]
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

    wx_s, day_s, hour_s, lane_s, road_s = [], [], [], [], []
    for h in range(24):
        for d in DAY_TYPES:
            for l in LANES:
                for r in ROADS:
                    wx_s.append(spread([grid[f"{h}|{w}|{d}|{l}|{r}"][3] for w in WEATHERS]))
    for h in range(24):
        for w in WEATHERS:
            for l in LANES:
                for r in ROADS:
                    day_s.append(spread([grid[f"{h}|{w}|{d}|{l}|{r}"][3] for d in DAY_TYPES]))
    for w in WEATHERS:
        for d in DAY_TYPES:
            for l in LANES:
                for r in ROADS:
                    hour_s.append(spread([grid[f"{h}|{w}|{d}|{l}|{r}"][3] for h in range(24)]))
    for h in range(24):
        for w in WEATHERS:
            for d in DAY_TYPES:
                for r in ROADS:
                    lane_s.append(spread([grid[f"{h}|{w}|{d}|{l}|{r}"][3] for l in LANES]))
    for h in range(24):
        for w in WEATHERS:
            for d in DAY_TYPES:
                for l in LANES:
                    road_s.append(spread([grid[f"{h}|{w}|{d}|{l}|{r}"][3] for r in ROADS]))

    def stat(v):
        return {"avg": round(sum(v) / len(v), 1), "max": round(max(v), 1)}
    return {"hour_of_day": stat(hour_s), "wx": stat(wx_s),
            "day_type": stat(day_s), "lane_count": stat(lane_s), "road": stat(road_s)}


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
                    for road in ROADS:
                        x = pd.DataFrame([{"hour_of_day": hour, "wx": w, "day_type": day_type,
                                           "lane_count": lanes, "road": road}])
                        X = pd.get_dummies(x, columns=["wx", "day_type", "road"], dtype=float)
                        X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
                        fpl, speed, occ = reg.predict(X)[0]
                        prob = float(iso.predict(clf.predict_proba(X)[:, 1])[0])
                        k = f"{hour}|{w}|{day_type}|{lanes}|{road}"
                        grid[k] = [
                            int(round(fpl)), round(float(speed), 1), round(float(occ), 1),
                            round(prob * 100, 1),
                            round(max(5.0, speed + lo), 1), round(min(130.0, speed + hi), 1),
                        ]
                        counts[k] = int(
                            ((df.hour_of_day == hour) & (df.wx == w) & (df.day_type == day_type)
                             & (df.lane_count == lanes) & (df.road == road)).sum())

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
            "roads": ROADS,
            "roadLabels": ROAD_LABELS,
            "roadStations": {r: int(df[df.road == r].station_id.nunique()) for r in ROADS},
            "roadCong": {r: round(float(df[df.road == r].cong.mean()) * 100, 1) for r in ROADS},
            "roadSpeed": {r: round(float(df[df.road == r].avg_speed_kmh.mean()), 1) for r in ROADS},
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
