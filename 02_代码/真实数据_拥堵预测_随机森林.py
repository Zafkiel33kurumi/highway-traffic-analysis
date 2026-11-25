# ===================== 真实数据 · 断面级拥堵预测（随机森林） =====================
# 输入：断面 station_id、时段 hour、天气 weather、工作日/周末 is_weekend
#       车道数不是输入，而是断面的固定属性（同一条路不会今天 3 车道明天 5 车道）
# 输出：流量(辆/小时)、车速(km/h)、占有率(%)、拥堵概率 P(车速 < 70 km/h)
#
# 数据：PORTAL 波特兰高速线圈（2019-01-15 ~ 02-28，逐小时）+ Open-Meteo 实测天气
# 为什么必须用断面：把车道数当输入，等于把不同方向、不同位置的断面混在一起，
#   实测车速 R² 只有 0.25；换成断面后升到 0.73，因为早晚高峰本来就发生在不同方向的特定断面。

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (brier_score_loss, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "01_原始数据" / "波特兰高速_真实数据_2019冬.csv"
MODEL = Path(__file__).resolve().parent / "models" / "真实数据_断面随机森林.pkl"
SITE_DATA = BASE / "05_可视化系统" / "prediction_data.js"

FEATURES = ["hour_of_day", "weather", "station_id", "is_weekend"]
CATS = ["weather", "station_id"]
TARGETS = ["flow_veh_hour", "avg_speed_kmh", "occupancy_pct"]
CONGESTION_SPEED = 70.0          # km/h，低于此值视为拥堵/缓行
WEATHERS = ["Clear", "Cloudy", "Rain", "Snow"]   # 该时段波特兰未出现雾/雷暴
RANDOM_STATE = 42


def load():
    return pd.read_csv(DATA).dropna(subset=FEATURES + TARGETS)


def build():
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATS)], remainder="passthrough")
    return Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(n_estimators=80, min_samples_leaf=5,
                                     n_jobs=-1, random_state=RANDOM_STATE, oob_score=True)),
    ])


def train():
    df = load()
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGETS], test_size=0.2,
                                          random_state=RANDOM_STATE)
    pipe = build().fit(Xtr, ytr)
    oob = pipe.named_steps["rf"].oob_prediction_
    pipe.oob_speed_resid = ytr.avg_speed_kmh.values - oob[:, 1]
    pipe.resid_q = np.percentile(ytr.values - oob, [5, 95], axis=0)

    pred = pipe.predict(Xte)
    print(f"样本 {len(df)}｜站点 {df.station_id.nunique()}｜训练 {len(Xtr)}｜测试 {len(Xte)}"
          f"｜拥堵定义 车速<{CONGESTION_SPEED:.0f} km/h")
    for i, t in enumerate(TARGETS):
        print(f"  {t:16s} R²={r2_score(yte.iloc[:, i], pred[:, i]):6.3f} "
              f"MAE={mean_absolute_error(yte.iloc[:, i], pred[:, i]):7.2f}")

    p = (pipe.oob_speed_resid[None, :] < CONGESTION_SPEED - pred[:, 1][:, None]).mean(axis=1)
    y = (yte.avg_speed_kmh.values < CONGESTION_SPEED).astype(int)
    print(f"  拥堵概率        AUC={roc_auc_score(y, p):.3f}  Brier={brier_score_loss(y, p):.3f}"
          f"  实际拥堵率={y.mean():.1%}")

    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["rf"].feature_importances_
    print("\n【特征重要性（按输入合并）】", importance_by_input(names, imp))

    joblib.dump(pipe, MODEL)
    export_site(pipe, df, yte, pred, p)
    return pipe


def importance_by_input(names, imp):
    total = {"hour_of_day": 0.0, "weather": 0.0, "station_id": 0.0, "is_weekend": 0.0}
    for n, v in zip(names, imp):
        key = next((c for c in total if c in str(n)), str(n).split("__")[-1])
        total[key] = total.get(key, 0.0) + float(v)
    return {k: round(v, 3) for k, v in sorted(total.items(), key=lambda t: -t[1]) if v > 0.001}


def predict_row(pipe, station, hour, weather, weekend=0):
    x = pd.DataFrame([{"hour_of_day": hour, "weather": weather,
                       "station_id": station, "is_weekend": weekend}])
    flow, speed, occ = pipe.predict(x)[0]
    q05, q95 = pipe.resid_q
    p = float((pipe.oob_speed_resid < CONGESTION_SPEED - speed).mean())
    f_lo, f_hi = max(0.0, flow + q05[0]), max(0.0, flow + q95[0])
    s_lo, s_hi = max(5.0, speed + q05[1]), min(130.0, speed + q95[1])
    return [round(float(flow)), round(float(speed), 1), round(float(occ), 1),
            round(p * 100, 1), round(float(f_lo)), round(float(f_hi)),
            round(float(s_lo), 1), round(float(s_hi), 1)]


def export_site(pipe, df, yte, pred, p_test):
    stations = []
    for sid, g in df.groupby("station_id"):
        stations.append({
            "id": int(sid),
            "freeway": g.freeway.iloc[0],
            "direction": g.direction.iloc[0],
            "lanes": int(g.lane_count.iloc[0]),
            "flow": int(g.flow_veh_hour.mean()),
            "speed": round(float(g.avg_speed_kmh.mean()), 1),
            "congRate": round(float((g.avg_speed_kmh < CONGESTION_SPEED).mean()), 3),
            "label": f'{g.freeway.iloc[0]} {g.direction.iloc[0]} · {int(sid)} 号断面 · '
                     f'{int(g.lane_count.iloc[0])} 车道',
        })
    stations.sort(key=lambda s: -s["flow"])

    grid = {}
    for s in stations:
        for weekend in (0, 1):
            for hour in range(24):
                for weather in WEATHERS:
                    grid[f'{s["id"]}|{hour}|{weather}|{weekend}'] = predict_row(
                        pipe, s["id"], hour, weather, weekend)

    y = (yte.avg_speed_kmh.values < CONGESTION_SPEED).astype(int)
    payload = {
        "meta": {
            "source": "PORTAL 波特兰高速线圈 + Open-Meteo 天气",
            "period": "2019-01-15 ~ 2019-02-28",
            "rows": int(len(df)),
            "stations": len(stations),
            "threshold": CONGESTION_SPEED,
            "weathers": WEATHERS,
            "metrics": {
                "flowR2": round(float(r2_score(yte.flow_veh_hour, pred[:, 0])), 3),
                "speedR2": round(float(r2_score(yte.avg_speed_kmh, pred[:, 1])), 3),
                "occR2": round(float(r2_score(yte.occupancy_pct, pred[:, 2])), 3),
                "auc": round(float(roc_auc_score(y, p_test)), 3),
                "brier": round(float(brier_score_loss(y, p_test)), 3),
                "baseRate": round(float(y.mean()), 3),
            },
            "importance": importance_by_input(
                pipe.named_steps["prep"].get_feature_names_out(),
                pipe.named_steps["rf"].feature_importances_),
        },
        "stations": stations,
        "grid": grid,
    }
    SITE_DATA.parent.mkdir(exist_ok=True)
    SITE_DATA.write_text("window.TRAFFIC = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                         encoding="utf-8")
    print(f"\n网页数据已导出：{SITE_DATA}（{len(stations)} 个断面，{len(grid)} 个组合，"
          f"{SITE_DATA.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    train()
