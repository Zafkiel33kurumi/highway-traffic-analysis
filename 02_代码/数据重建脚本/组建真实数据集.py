# ===================== 组建真实数据集：PORTAL 线圈 + Open-Meteo 天气 =====================
# 数据来源（均为公开数据，无需密钥）：
#   · 交通：PORTAL (Portland State University) 高速线圈，波特兰 I-5 双向、I-84 东向，逐小时
#           2018-11 ~ 2020-04（两个冬季，分月文件在 01_原始数据/原始下载/扩展/）
#           https://new.portal.its.pdx.edu/highways/api/freewaydata/
#   · 天气：Open-Meteo 历史再分析（波特兰机场 45.59N, 122.60W），同一时段逐小时
#           https://archive-api.open-meteo.com/v1/archive
# 处理：检测器(单车道) → 站点(断面) 聚合；车道数 = 该站检测器数量；
#       天气按 WMO 天气码归为 Clear / Cloudy / Rain / Snow / Fog / Thunderstorm；
#       丢弃读数不足的整点（countreadings < 120，即一小时少于 40 分钟有效）。

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
TMP = BASE / "01_原始数据" / "原始下载"
EXT = TMP / "扩展"
OUT = BASE / "01_原始数据" / "波特兰高速_真实数据_多冬季.csv"

HIGHWAYS = [1, 2, 7]

# 美国联邦假日（落在数据时段内的），这些日子的出行规律与普通工作日不同
HOLIDAYS = {
    "2018-11-12", "2018-11-22", "2018-12-25", "2019-01-01", "2019-01-21",
    "2019-02-18", "2019-11-11", "2019-11-28", "2019-12-25", "2020-01-01",
    "2020-01-20", "2020-02-17",
}

WEATHER_CODE = {
    0: "Clear", 1: "Clear", 2: "Cloudy", 3: "Cloudy",
    45: "Fog", 48: "Fog",
    51: "Rain", 53: "Rain", 55: "Rain", 56: "Rain", 57: "Rain",
    61: "Rain", 63: "Rain", 65: "Rain", 66: "Rain", 67: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain", 81: "Rain", 82: "Rain", 85: "Snow", 86: "Snow",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def load_traffic():
    frames = []
    for hw in HIGHWAYS:
        files = [TMP / f"hw{hw}.csv"]
        files += sorted(EXT.glob(f"hw{hw}_*.csv")) if EXT.exists() else []
        for f in files:
            if not f.exists() or f.stat().st_size < 1000:
                continue
            d = pd.read_csv(f)
            d["highway_id"] = hw
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # 分月文件与整段文件有重叠，按 检测器+时刻 去重
    df = df.drop_duplicates(subset=["detector_id", "starttime"])

    # 每条记录是一个车道检测器；用官方元数据映射到站点（断面）
    det = pd.read_csv(TMP / "det.csv")
    df["station_id"] = df.detector_id.map(det.set_index("detectorid").stationid)
    df = df.dropna(subset=["station_id"])
    df["station_id"] = df.station_id.astype(int)
    df = df[(df.countreadings >= 120) & (df.volume > 0) & (df.speed > 0)]
    df["lane_count"] = df.groupby("station_id").detector_id.transform("nunique")
    df = df[df.lane_count.between(2, 6)]

    g = df.groupby(["station_id", "starttime"])
    out = pd.DataFrame({
        "flow_veh_hour": g.volume.sum(),
        "avg_speed_mph": g.apply(lambda x: np.average(x.speed, weights=x.volume), include_groups=False),
        "occupancy_pct": g.occupancy.mean(),
        "lane_count": g.lane_count.first(),
        "n_lanes_obs": g.size(),
    }).reset_index()
    out = out.rename(columns={"starttime": "timestamp"})
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    out["hour_of_day"] = out.timestamp.dt.hour
    out["is_weekend"] = (out.timestamp.dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = out.timestamp.dt.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)
    hw_meta = {1: ("I-5", "北向"), 2: ("I-5", "南向"), 3: ("I-205", "北向"),
               4: ("I-205", "南向"), 7: ("I-84", "东向"), 8: ("I-84", "西向")}
    hw_id = out.station_id.map(df.drop_duplicates("station_id").set_index("station_id").highway_id)
    out["freeway"] = hw_id.map({k: v[0] for k, v in hw_meta.items()})
    out["direction"] = hw_id.map({k: v[1] for k, v in hw_meta.items()})
    out["avg_speed_kmh"] = out.avg_speed_mph * 1.60934
    return out.drop(columns="avg_speed_mph")


def load_weather():
    src = EXT / "weather_ext.json"
    if not src.exists():
        src = TMP / "weather.json"
    w = json.loads(src.read_text())["hourly"]
    wx = pd.DataFrame({
        "timestamp": pd.to_datetime(w["time"]),
        "weather": [WEATHER_CODE.get(c, "Cloudy") for c in w["weather_code"]],
        "temp_c": w["temperature_2m"],
        "precip_mm": w["precipitation"],
        "snow_cm": w["snowfall"],
        "wind_kmh": w["wind_speed_10m"],
    })
    # 降雪优先于降雨：气温接近 0 且 snow_cm>0 才算雪
    wx.loc[wx.snow_cm > 0, "weather"] = "Snow"
    return wx


def main():
    tr = load_traffic()
    wx = load_weather()
    df = tr.merge(wx, on="timestamp", how="left")
    df = df.dropna(subset=["weather", "flow_veh_hour", "avg_speed_kmh", "occupancy_pct"])
    df = df[(df.avg_speed_kmh.between(5, 160)) & (df.occupancy_pct.between(0, 100))]
    df["record_id"] = np.arange(1, len(df) + 1)
    cols = ["record_id", "station_id", "freeway", "direction", "timestamp", "hour_of_day",
            "is_weekend", "is_holiday", "flow_veh_hour", "avg_speed_kmh", "occupancy_pct",
            "lane_count", "weather", "temp_c", "precip_mm", "snow_cm"]
    df = df[cols].sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"已生成 {OUT.name}：{len(df)} 行，{df.station_id.nunique()} 个站点，"
          f"{df.timestamp.dt.date.nunique()} 天，{df.timestamp.min()} ~ {df.timestamp.max()}")
    print("\n天气分布：")
    print(df.weather.value_counts().to_string())
    print("\n各指标：")
    print(df[["flow_veh_hour", "avg_speed_kmh", "occupancy_pct", "lane_count"]].describe().round(1).to_string())
    print("\n天气 × 车速（km/h）：")
    print(df.groupby("weather").avg_speed_kmh.agg(["mean", "count"]).round(1).to_string())
    print("\n车道数 × 流量（辆/小时）：")
    print(df.groupby("lane_count").flow_veh_hour.agg(["mean", "count"]).round(0).to_string())


if __name__ == "__main__":
    main()
