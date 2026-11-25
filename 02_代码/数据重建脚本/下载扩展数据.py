# ===================== 扩展下载：PORTAL 交通流 + Open-Meteo 天气 =====================
# 现有数据只有 2019-01-15 ~ 02-28（42 天），导致"某天气 × 某小时 × 某日期类型"的格子大量为空。
# 这个脚本按"月"分批补拉两个冬季（2018-11 ~ 2019-04、2019-11 ~ 2020-04），
# 单月约 4~5 分钟，并发 6 路，可中断重跑（已存在的文件会跳过）。
#
# 用法：python3 02_代码/下载扩展数据.py

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "01_原始数据" / "原始下载" / "扩展"
OUT.mkdir(parents=True, exist_ok=True)

HIGHWAYS = [1, 2, 7]          # I-5 北向、I-5 南向、I-84 东向
MONTHS = [f"{y}-{m:02d}" for y, ms in [(2018, [11, 12]), (2019, [1, 2, 3, 4]),
                                      (2019, [11, 12]), (2020, [1, 2, 3, 4])]
          for m in ms]
API = "https://new.portal.its.pdx.edu/highways/api/freewaydata/"


def month_range(ym):
    """'2019-02' -> ('2019-02-01', '2019-03-01')，end 为开区间。"""
    y, m = map(int, ym.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"


def fetch(hw, ym):
    dst = OUT / f"hw{hw}_{ym}.csv"
    if dst.exists() and dst.stat().st_size > 200_000:
        return f"skip {dst.name}"
    start, end = month_range(ym)
    url = f"{API}?start_date={start}&end_date={end}&highway_id={hw}&format=csv"
    r = subprocess.run(["curl", "-sL", "--max-time", "900", url, "-o", str(dst)],
                       capture_output=True)
    if r.returncode != 0 or not dst.exists():
        return f"FAIL hw{hw} {ym} rc={r.returncode}"
    return f"ok   hw{hw} {ym} {dst.stat().st_size // 1024} KB"


def fetch_weather():
    dst = OUT / "weather_ext.json"
    url = ("https://archive-api.open-meteo.com/v1/archive"
           "?latitude=45.5898&longitude=-122.5951&start_date=2018-11-01&end_date=2020-05-01"
           "&hourly=temperature_2m,precipitation,rain,snowfall,weather_code,wind_speed_10m"
           "&timezone=America%2FLos_Angeles")
    r = subprocess.run(["curl", "-sL", "--max-time", "300", url, "-o", str(dst)],
                       capture_output=True)
    return f"ok   weather_ext.json {dst.stat().st_size // 1024} KB" if r.returncode == 0 \
        else f"FAIL weather rc={r.returncode}"


def main():
    tasks = [(hw, ym) for hw in HIGHWAYS for ym in MONTHS]
    print(f"共 {len(tasks)} 个月份文件，并发 6 路，预计 25~45 分钟", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, msg in enumerate(pool.map(lambda t: fetch(*t), tasks), 1):
            print(f"[{i}/{len(tasks)}] {msg}", flush=True)
    print(fetch_weather(), flush=True)
    json.dump({"highways": HIGHWAYS, "months": MONTHS, "source": API},
              open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print("全部完成", flush=True)


if __name__ == "__main__":
    sys.exit(main())
