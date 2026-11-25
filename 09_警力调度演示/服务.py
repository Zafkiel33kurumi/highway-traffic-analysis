# ===== 交警警力调度演示 · 服务端 =====
# 演示系统：所有数据由"预测网格 + 断面基线"模拟生成，不依赖外部接口。
# 三个模块：警力分配与调动、任务下发与监控、设备状态监测。
#
# 运行：python3 服务.py   然后打开 http://127.0.0.1:8766

import atexit
import copy
import json
import math
import random
import secrets
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
ROADS = json.loads((BASE / "数据" / "路段.json").read_text(encoding="utf-8"))
GRID = json.loads((BASE / "数据" / "网格.json").read_text(encoding="utf-8"))

STEP_MIN = 30                     # 一个 tick = 30 分钟
DAY_TICKS = 48                    # 一天 48 个时段（06:00 到次日 06:00）
GROUP_COUNT = 20                  # 警组编制
ONSITE_TICKS = 3                  # 到场后处置 3 个 tick
RETURN_TICKS = 1
COVER_MAX = 8                     # 一名警员最多负责的连续拥堵断面数（缺人时自动拉大）
MAX_PER_AREA = 3                  # 同一拥堵段最多投入几个警组
STANDBY_AHEAD = 2                 # 提前 2 个时段（1 小时）预测待命点
STANDBY_PROB = 30                 # 预测概率达到多少才去预置待命
RECORD_FILE = BASE / "数据" / "任务记录.json"

LEVELS = [(50, "严重拥堵"), (30, "拥堵"), (14, "缓行"), (0, "畅通")]
LEVEL_KEY = {"严重拥堵": "bad", "拥堵": "warn", "缓行": "slow", "畅通": "ok"}

# 人为注入的设备故障：(断面序号, 类型, 开始时, 持续小时)
FAULTS = [
    (7, "stuck", 7, 5),
    (33, "zero", 8, 3),
    (19, "spike", 14, 4),
    (51, "bias", 16, 6),
    (68, "stuck", 18, 5),
]
FAULT_LABEL = {"stuck": "数据冻结", "zero": "速度掉零", "spike": "数值尖峰", "bias": "占有率漂移"}

app = Flask(__name__, template_folder="模板")
app.config["TEMPLATES_AUTO_RELOAD"] = True   # 改了页面刷新就能看到，不用重启服务
STATE = {}          # 当前打开任务的状态（计算函数都读它）
TASKS = {}          # 全部任务：id -> {id,name,lines,groupCount,status,createdAt,state,ticks}
SEQ = 0             # 任务编号
LOCK = threading.Lock()   # 多个页面同时推进时串行处理，避免互相踩状态


def weather_at(hour):
    if hour < 6:
        return "Clear"
    if hour < 10:
        return "Cloudy"
    if hour < 16:
        return "LightRain"
    if hour < 20:
        return "HeavyRain"
    return "LightSnow"


WEATHER_CN = {"Clear": "晴", "Cloudy": "多云", "LightRain": "小雨",
              "HeavyRain": "大雨", "LightSnow": "小雪", "HeavySnow": "大雪"}


def level_of(prob):
    for cut, name in LEVELS:
        if prob >= cut:
            return name
    return "畅通"


def nearest_hour(minutes):
    h = minutes / 60.0
    return min(23, max(0, int(round(h))))


def grid_row(hour, wx, lanes):
    lane = min(5, max(2, int(lanes)))
    row = GRID.get(f"{hour}|{wx}|workday|{lane}")
    if row is None:
        row = GRID.get(f"{hour}|Clear|workday|{lane}") or [300, 85, 20, 10]
    return row


def fault_at(idx, hour):
    """返回该断面此刻被注入的故障类型（没有则 None）。"""
    for sid, kind, start, dur in FAULTS:
        if sid == idx and start <= hour < start + dur:
            return kind
    return None


def apply_fault(kind, flow, speed, occ, base, tick):
    """把故障叠加到观测值上，返回 (flow, speed, occ)。"""
    if kind == "zero":
        return flow, 0.0, max(0.2, occ * 0.3)
    if kind == "spike":
        return flow * 1.15, min(160.0, speed * 2.6), occ
    if kind == "bias":
        return flow, speed, min(98.0, occ + 28)
    if kind == "stuck":
        return base["baseFlow"], base["baseSpeed"], base["baseOcc"]   # 停在基线上不动
    return flow, speed, occ


def new_state(group_count=GROUP_COUNT, jurisdiction=None):
    jurisdiction = list(jurisdiction or [r["idx"] for r in ROADS])
    groups = []
    for i in range(group_count):
        teams = ["一中队", "二中队", "三中队", "四中队"]
        home = jurisdiction[min(len(jurisdiction) - 1,
                                int(round((i + 0.5) * len(jurisdiction) / group_count)))]
        groups.append({
            "id": f"QG-{i + 1:02d}",
            "name": f"{teams[i % 4]}·第{i // 4 + 1}警组",
            "home": home,
            "status": "idle",          # idle / enroute / onsite / returning
            "station": None,
            "taskId": None,
            "eta": 0,
            "remain": 0,
            "done": 0,
            "post": None,              # 预置待命点（断面 idx），None 表示在驻地待命
        })
    return {
        "tick": 0,
        "minutes": 6 * 60,           # 每个任务都从早上 6 点开局
        "token": secrets.token_hex(6),   # 只允许当前页面推进时钟，避免多个标签页同时加速
        "day": 1,
        "playing": True,
        "seq": 0,
        "taskId": "", "taskName": "", "taskStatus": "running",
        "jurisdiction": jurisdiction,
        "groups": groups,
        "tasks": [],
        "events": [],
        "stationState": {},     # idx -> {assigned, lastSpeeds, health, fault}
        "lastStations": [],
        "kpiHistory": [],
    }


def ensure_station(idx):
    st = STATE["stationState"].setdefault(idx, {
        "assigned": 0, "speeds": [], "flows": [], "occs": [], "health": 100, "detected": None, "alarm": 0,
    })
    return st


def log(text, kind="info"):
    events = STATE["events"]
    events.insert(0, {
        "t": f"{STATE['minutes'] // 60:02d}:{STATE['minutes'] % 60:02d}",
        "text": text, "kind": kind, "tick": STATE["tick"],
    })
    del events[120:]


def compute_stations(detect=True):
    """按当前时间算出 86 个断面的观测值、拥堵等级与设备健康度。
    detect=False 时只取值、不推进设备判据（用于页面刷新，避免重复计数）。
    """
    hour = STATE["minutes"] / 60.0
    h = nearest_hour(STATE["minutes"])
    wx = weather_at(int(hour))
    out = []
    for r in ROADS:
        row = grid_row(h, wx, r["lanes"])
        clear = grid_row(h, "Clear", r["lanes"])
        fpl, speed, occ, _ = row[:4]
        # 天气影响 = 同条件下有雨雪 / 晴天的概率之比，限制在 0.5~2.5 倍
        factor = 1.0
        if clear[3] > 1:
            factor = max(0.5, min(2.5, row[3] / clear[3]))
        rng = random.Random(STATE["tick"] * 977 + r["idx"])
        base_prob = r["hourly"][h] * 100 * factor
        prob = max(0.0, min(96.0, base_prob * (1 + rng.uniform(-0.06, 0.06))))
        flow = fpl * r["lanes"] * r["flowRatio"] * (1 + rng.uniform(-0.05, 0.05))
        speed = speed * r["speedRatio"] * (1 + rng.uniform(-0.03, 0.03))
        occ = occ * r["occRatio"] * (1 + rng.uniform(-0.05, 0.05))

        st = ensure_station(r["idx"])
        kind = fault_at(r["idx"], hour)
        flow, speed, occ = apply_fault(kind, flow, speed, occ, r, STATE["tick"])

        # --- 设备诊断：规则 + 与近几个时段的偏离 ---
        detected, penalty = None, 0
        if detect:
            st["speeds"] = (st["speeds"] + [round(speed, 1)])[-6:]
            st["flows"] = (st["flows"] + [int(flow)])[-6:]
            st["occs"] = (st["occs"] + [round(occ, 1)])[-6:]
            same_speed = len(st["speeds"]) >= 4 and len(set(st["speeds"][-4:])) == 1
            same_flow = len(st["flows"]) >= 4 and len(set(st["flows"][-4:])) == 1
            occ_median = sorted(st["occs"][:-1])[len(st["occs"][:-1]) // 2] if len(st["occs"]) >= 4 else occ
            spd_median = sorted(st["speeds"][:-1])[len(st["speeds"][:-1]) // 2] if len(st["speeds"]) >= 4 else speed
            if speed < 3 and flow > 50:
                detected, penalty = "zero", 60
            elif same_speed and same_flow:
                detected, penalty = "stuck", 55
            elif speed > r["baseSpeed"] * 1.7:
                detected, penalty = "spike", 50
            elif (len(st["occs"]) >= 4 and abs(occ - occ_median) > 22
                  and abs(speed - spd_median) < 8):        # 只是占有率跑了，车速仍正常
                detected, penalty = "bias", 45
            health = max(0, 100 - penalty) if detected else 100
            if detected and st["detected"] != detected:
                log(f"{r['sid']} 设备异常：{FAULT_LABEL[detected]}（健康度 {health}）", "alarm")
            st["detected"], st["health"] = detected, health
        else:
            detected, health = st["detected"], st["health"]

        level = level_of(prob)
        need = 2 if prob >= 50 else (1 if prob >= 30 else 0)
        out.append({
            "idx": r["idx"], "sid": r["sid"], "line": r["line"], "dir": r["dir"],
            "lanes": r["lanes"], "speed": round(speed, 1), "flow": int(flow),
            "occ": round(occ, 1), "prob": round(prob, 1), "level": level,
            "levelKey": LEVEL_KEY[level], "need": need,
            "assigned": st["assigned"], "health": health,
            "detected": detected, "fault": FAULT_LABEL.get(detected, ""),
        })
    return out


def forecast_prob(road, minutes):
    """用训练好的"整体规律模型"（随机森林，已导出为网格）预测指定时刻的拥堵概率。
    输入：时段、天气、日期类型（本演示固定工作日）、车道数。
    """
    h = nearest_hour(minutes)
    wx = weather_at(int((minutes / 60.0) % 24))
    row = grid_row(h, wx, road["lanes"])
    clear = grid_row(h, "Clear", road["lanes"])
    factor = 1.0
    if clear[3] > 1:
        factor = max(0.5, min(2.5, row[3] / clear[3]))
    return max(0.0, min(96.0, road["hourly"][h] * 100 * factor))


def active_tasks():
    return [t for t in STATE["tasks"] if t["status"] in ("enroute", "onsite")]


def coverage_of(task):
    return task.get("zone") or [task["idx"]]


def recompute_coverage(stations):
    """在岗警力按"负责路段"算覆盖：一名警员管一段，段内断面都算已覆盖。"""
    by_idx = {s["idx"]: s for s in stations}
    for s in stations:
        s["assigned"] = 0
    for t in active_tasks():
        for idx in coverage_of(t):
            if idx in by_idx:
                by_idx[idx]["assigned"] += 1
    for s in stations:
        st = STATE["stationState"].get(s["idx"])
        if st:
            st["assigned"] = s["assigned"]


def hot_areas(stations):
    """把辖区里连续的拥堵断面（≥30%）合成一段；间隔 1 个断面也并进来。
    返回按严重程度排序的路段，每段带建议投入警组数。
    """
    juris = set(STATE["jurisdiction"])
    raw = []
    for line in ("干线一", "干线二"):
        seq = sorted([s for s in stations if s["line"] == line and s["idx"] in juris],
                     key=lambda s: s["idx"])
        cur = []
        for s in seq:
            if s["prob"] >= 30:
                if cur and s["idx"] - cur[-1]["idx"] > 2:
                    raw.append(cur)
                    cur = []
                cur.append(s)
            elif cur:
                raw.append(cur)
                cur = []
        if cur:
            raw.append(cur)

    areas = []
    for st in raw:
        mx = max(s["prob"] for s in st)
        demand = 1
        if mx >= 55 and len(st) >= 3:
            demand += 1                       # 特别严重的段加一组
        if len(st) >= COVER_MAX:
            demand += 1                       # 很长的段加一组
        areas.append({
            "line": st[0]["line"],
            "stations": st,
            "ids": [s["idx"] for s in st],
            "maxProb": mx,
            "demand": min(MAX_PER_AREA, demand),
        })
    areas.sort(key=lambda a: (-a["maxProb"], -len(a["stations"])))
    return areas


def choose_zone(area, covered_ids):
    """在没有被覆盖的部分里，挑一段连续的断面交给这名警员（最多 COVER_MAX 个）。"""
    ids = area["ids"]
    span = min(COVER_MAX, len(ids))
    prob_of = {s["idx"]: s["prob"] for s in area["stations"]}
    best, best_score = ids[:span], None
    for lo in range(0, len(ids) - span + 1):
        win = ids[lo:lo + span]
        score = (sum(1 for i in win if i not in covered_ids),
                 max(prob_of[i] for i in win))
        if best_score is None or score > best_score:
            best, best_score = win, score
    return best


def send_task(g, hot, zone, manual=False):
    """给一个警组下派单（负责一段 zone，不止一个断面）。"""
    base_idx = g["home"]
    if g.get("station"):
        cur = next((r["idx"] for r in ROADS if r["sid"] == g["station"]), None)
        if cur:
            base_idx = cur
    elif g.get("post") is not None:
        base_idx = g["post"]
    dist = abs(base_idx - hot["idx"])
    eta = max(1, min(6, 1 + dist // 15))

    STATE["seq"] += 1
    tid = f"RW{STATE['day']:02d}{STATE['seq']:03d}"
    task = {
        "id": tid, "sid": hot["sid"], "idx": hot["idx"], "line": hot["line"],
        "level": hot["level"], "levelKey": hot["levelKey"],
        "group": g["id"], "groupName": g["name"],
        "status": "enroute", "eta": eta, "remain": ONSITE_TICKS,
        "created": STATE["minutes"], "arrived": None,
        "zone": zone, "manual": manual,
    }
    STATE["tasks"].insert(0, task)
    g.update(status="enroute", station=hot["sid"], taskId=tid, eta=eta, post=None)
    span = ""
    if len(zone) > 1:
        first = next(r["sid"] for r in ROADS if r["idx"] == min(zone))
        last = next(r["sid"] for r in ROADS if r["idx"] == max(zone))
        span = f"，负责 {first}–{last} 共 {len(zone)} 个断面"
    log(f"{'手动改派' if manual else '派单'} {tid}：{g['name']} 前往 "
        f"{hot['line']}{hot['sid']}（{hot['level']}，{hot['prob']}%）{span}",
        "dispatch")
    return task


def dispatch(stations):
    """均衡覆盖：先把每个拥堵段都安排上人，再给严重 / 很长的段加人。
    警力不足时，一名警员负责一整段连续拥堵断面，避免所有人挤在同一个点。
    """
    if not stations:
        return
    areas = hot_areas(stations)
    active = active_tasks()
    idle = [g for g in STATE["groups"] if g["status"] == "idle"]

    for rnd in range(1, MAX_PER_AREA + 1):
        for a in areas:
            if a["demand"] < rnd:
                continue
            ids = set(a["ids"])
            here = [t for t in active if t["idx"] in ids]
            if len(here) >= rnd:
                continue
            if not idle:
                return
            covered = set()
            for t in here:
                covered |= set(coverage_of(t))
            zone = choose_zone(a, covered)
            hot = max([s for s in a["stations"] if s["idx"] in zone],
                      key=lambda s: s["prob"])
            g = min(idle, key=lambda x: abs(
                (x["post"] if x.get("post") is not None else x["home"]) - hot["idx"]))
            idle.remove(g)
            active.append(send_task(g, hot, zone))


def plan_standby(stations):
    """用模型预测 1 小时后可能拥堵的地方，把还没出警的警组预置到附近待命。"""
    juris = set(STATE["jurisdiction"])
    idle = [g for g in STATE["groups"] if g["status"] == "idle"]
    if not idle:
        return
    ahead = STATE["minutes"] + STANDBY_AHEAD * STEP_MIN
    roads = {r["idx"]: r for r in ROADS if r["idx"] in juris}
    covered = set()
    for t in active_tasks():
        covered |= set(coverage_of(t))

    cand = []
    for idx, r in roads.items():
        if idx in covered:
            continue
        p = forecast_prob(r, ahead)
        if p >= STANDBY_PROB:
            cand.append((p, idx))
    cand.sort(reverse=True)

    taken = set()
    for g in idle:                       # 原来的待命点还有效就保留，避免来回折腾
        post = g.get("post")
        if post is None or post not in roads:
            g["post"] = None
            continue
        p = forecast_prob(roads[post], ahead)
        if p >= STANDBY_PROB - 5 and post not in covered:
            taken.add(post)
        else:
            g["post"] = None

    free = [g for g in idle if g.get("post") is None]
    for p, idx in cand:
        if not free or idx in taken or idx in covered:
            continue
        g = min(free, key=lambda x: abs(x["home"] - idx))
        g["post"] = idx
        free.remove(g)
        taken.add(idx)


def advance_tasks(stations):
    by_idx = {s["idx"]: s for s in stations}
    prob_of = {s["idx"]: s["prob"] for s in stations}
    groups = {g["id"]: g for g in STATE["groups"]}

    for t in list(STATE["tasks"]):
        g = groups[t["group"]]
        if t["status"] == "enroute":
            t["eta"] -= 1
            g["eta"] = t["eta"]
            if prob_of.get(t["idx"], 0) < 14 and not t.get("manual"):
                t["status"] = "cancel"
                g.update(status="returning", remain=RETURN_TICKS)
                by_idx[t["idx"]]["assigned"] = max(0, by_idx[t["idx"]]["assigned"] - 1)
                ensure_station(t["idx"])["assigned"] = by_idx[t["idx"]]["assigned"]
                log(f"{t['id']} 撤单：{t['sid']} 路况已缓解，{t['group']} 返回待命", "cancel")
            elif t["eta"] <= 0:
                t["status"] = "onsite"
                t["arrived"] = STATE["minutes"]
                g["status"] = "onsite"
                log(f"{t['id']} 到达：{t['group']} 已在 {t['sid']} 现场处置", "arrive")
        elif t["status"] == "onsite":
            t["remain"] -= 1
            if t["remain"] <= 0:
                t["status"] = "done"
                g.update(status="returning", remain=RETURN_TICKS)
                g["done"] += 1
                by_idx[t["idx"]]["assigned"] = max(0, by_idx[t["idx"]]["assigned"] - 1)
                ensure_station(t["idx"])["assigned"] = by_idx[t["idx"]]["assigned"]
                log(f"{t['id']} 完成：{t['group']} 处置结束，恢复巡控", "done")
        elif t["status"] in ("done", "cancel"):
            pass
    for g in STATE["groups"]:
        if g["status"] == "returning":
            g["remain"] -= 1
            if g["remain"] <= 0:
                g.update(status="idle", station=None, taskId=None, remain=0)


def tick():
    STATE["minutes"] += STEP_MIN
    if STATE["minutes"] >= 24 * 60:
        STATE["minutes"] = 0
        STATE["day"] += 1
        log(f"进入第 {STATE['day']} 天，班次交接", "info")
    STATE["tick"] += 1
    stations = compute_stations()
    advance_tasks(stations)
    recompute_coverage(stations)     # 先把"谁管哪段"算清楚
    dispatch(stations)               # 再按均衡覆盖补人
    recompute_coverage(stations)
    plan_standby(stations)           # 剩余警力去预测热点预置待命
    STATE["lastStations"] = stations
    active = [t for t in STATE["tasks"] if t["status"] in ("enroute", "onsite")]
    STATE["kpiHistory"] = (STATE["kpiHistory"] + [{
        "t": STATE["minutes"], "cong": sum(1 for s in stations if s["prob"] >= 45),
        "active": len(active),
    }])[-48:]
    return stations


def snapshot():
    stations = STATE.get("lastStations") or compute_stations()
    managed = set(STATE["jurisdiction"])
    for s in stations:                      # 警力占用以最新派单结果为准
        s["assigned"] = ensure_station(s["idx"])["assigned"]
        s["managed"] = s["idx"] in managed
    active = [t for t in STATE["tasks"] if t["status"] in ("enroute", "onsite")]
    done = [t for t in STATE["tasks"] if t["status"] == "done"]
    idle = [g for g in STATE["groups"] if g["status"] == "idle"]
    demand = sum(s["need"] for s in stations if s["idx"] in managed)
    onsite = [t for t in active if t["status"] == "onsite"]
    arrive = [max(0, (t["arrived"] or 0) - t["created"]) for t in STATE["tasks"] if t["arrived"]]
    alarms = [s for s in stations if s["detected"] and s["idx"] in managed]
    hour = STATE["minutes"] // 60

    # 预测热点：用训练好的整体规律模型看 1 小时后哪里可能堵，给待命和提前布警用
    ahead_min = STATE["minutes"] + STANDBY_AHEAD * STEP_MIN
    road_by_idx = {r["idx"]: r for r in ROADS}
    pts = sorted(((forecast_prob(r, ahead_min), r) for r in ROADS if r["idx"] in managed),
                 key=lambda x: -x[0])[:6]
    forecast = {
        "source": "整体规律模型（随机森林：时段 / 天气 / 日期 / 车道）",
        "ahead": f"{STANDBY_AHEAD * STEP_MIN} 分钟后",
        "points": [{"sid": r["sid"], "line": r["line"], "prob": round(p, 1)} for p, r in pts],
    }

    # 给前端补上"预置待命点"的断面编号和预测概率
    groups = []
    for g in STATE["groups"]:
        gg = dict(g)
        post = g.get("post")
        if post is not None and post in road_by_idx:
            gg["postSid"] = road_by_idx[post]["sid"]
            gg["postProb"] = round(forecast_prob(road_by_idx[post], ahead_min), 1)
        groups.append(gg)

    return {
        "taskId": STATE.get("taskId", ""),
        "taskName": STATE.get("taskName", ""),
        "status": STATE.get("taskStatus", "running"),
        "tick": STATE["tick"],
        "totalTicks": DAY_TICKS,
        "clock": f"{hour:02d}:{STATE['minutes'] % 60:02d}",
        "day": STATE["day"],
        "token": STATE["token"],
        "jurisdiction": STATE["jurisdiction"],
        "groupCount": len(STATE["groups"]),
        "weather": WEATHER_CN[weather_at(hour)],
        "weatherRaw": weather_at(hour),
        "playing": STATE["playing"],
        "kpi": {
            "congested": sum(1 for s in stations if s["prob"] >= 30 and s["idx"] in managed),
            "severe": sum(1 for s in stations if s["prob"] >= 50 and s["idx"] in managed),
            "demand": demand,
            "assigned": len(active),
            "idle": len(idle),
            "onsite": len(onsite),
            "doneToday": len(done),
            "avgArrive": round(sum(arrive) / len(arrive)) if arrive else 0,
            "faults": len(alarms),
        },
        "stations": stations,
        "groups": groups,
        "forecast": forecast,
        "tasks": STATE["tasks"][:40],
        "events": STATE["events"][:50],
        "alarms": [{"sid": s["sid"], "line": s["line"], "fault": s["fault"],
                    "health": s["health"], "key": s["detected"]} for s in alarms],
        "trend": STATE["kpiHistory"],
    }


# ===== 任务工作台：新建 / 打开 / 推进 / 回看 / 归档 =====

def default_name(lines, count):
    scope = "全线" if len(lines) > 1 else (lines[0] if lines else "全线")
    return f"{scope} · {count} 组"


def task_meta(t):
    """任务列表用的摘要：不把 86 个断面全塞进去。"""
    snap = t["ticks"][-1] if t.get("ticks") else None
    kpi = (snap or {}).get("kpi") or {}
    events = (snap or {}).get("events") or []
    return {
        "id": t["id"], "name": t["name"], "lines": t["lines"],
        "groupCount": t["groupCount"], "status": t["status"],
        "createdAt": t["createdAt"],
        "clock": (snap or {}).get("clock", "06:00"),
        "day": (snap or {}).get("day", 1),
        "tick": len(t.get("ticks") or []) - 1,
        "totalTicks": DAY_TICKS,
        "kpi": kpi,
        "lastEvent": events[0]["text"] if events else "",
        "snapshots": len(t.get("ticks") or []),
    }


def create_task(name, lines, groups):
    global STATE, SEQ
    lines = [x for x in (lines or []) if x in ("干线一", "干线二")] or ["干线一", "干线二"]
    try:
        count = int(groups or GROUP_COUNT)
    except (TypeError, ValueError):
        count = GROUP_COUNT
    count = max(4, min(40, count))
    allowed = [r["idx"] for r in ROADS if r["line"] in lines]

    SEQ += 1
    tid = f"T{SEQ:03d}"
    name = (name or "").strip()[:40] or default_name(lines, count)

    st = new_state(group_count=count, jurisdiction=allowed)
    st["taskId"], st["taskName"], st["taskStatus"] = tid, name, "running"
    STATE = st
    log(f"任务创建：{name} · 辖区 {'+'.join(lines)} · {count} 个警组待命", "info")
    st["lastStations"] = compute_stations(detect=False)   # 06:00 的初始画面

    t = {
        "id": tid, "name": name, "lines": lines, "groupCount": count,
        "status": "running",
        "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "state": st,
        "ticks": [copy.deepcopy(snapshot())],   # 下标 = 第几个时段，0 就是 06:00
    }
    TASKS[tid] = t
    save_records()
    return t


def record_tick(t):
    """推进一个时段，并把这一刻的完整画面存进历史。"""
    global STATE
    STATE = t["state"]
    tick()
    st = t["state"]
    if st["tick"] >= DAY_TICKS:          # 跑满一天自动归档
        st["taskStatus"] = "done"
        st["playing"] = False
        t["status"] = "done"
        log("任务已跑满一天，自动归档到历史任务", "done")
        t["ticks"].append(copy.deepcopy(snapshot()))
        save_records()
    else:
        t["ticks"].append(copy.deepcopy(snapshot()))
    return t["ticks"][-1]


def save_records():
    data = {"tasks": [
        {k: t[k] for k in ("id", "name", "lines", "groupCount", "status", "createdAt", "state", "ticks")}
        for t in TASKS.values()
    ]}
    RECORD_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_records():
    global TASKS, SEQ
    if not RECORD_FILE.exists():
        return
    try:
        data = json.loads(RECORD_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for t in data.get("tasks", []):
        if not t.get("ticks") or not t.get("state"):
            continue
        TASKS[t["id"]] = t
    nums = [int(k[1:]) for k in TASKS if k[1:].isdigit()]
    SEQ = max(nums or [0])


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/tasks")
def api_tasks():
    with LOCK:
        items = [task_meta(t) for t in reversed(list(TASKS.values()))]
        return jsonify({
            "tasks": items,
            "running": sum(1 for t in TASKS.values() if t["status"] == "running"),
            "done": sum(1 for t in TASKS.values() if t["status"] == "done"),
        })


@app.post("/api/tasks")
def api_task_create():
    data = request.get_json(silent=True) or {}
    with LOCK:
        t = create_task(data.get("name"), data.get("lines"), data.get("groups"))
        return jsonify({"task": task_meta(t), "snap": t["ticks"][-1]})


@app.get("/api/tasks/<tid>")
def api_task_get(tid):
    with LOCK:
        t = TASKS.get(tid)
        if not t:
            return jsonify({"error": "task not found"}), 404
        if t["status"] == "done":          # 历史任务打开就是回看
            return jsonify({"task": task_meta(t), "snap": copy.deepcopy(t["ticks"][-1]), "replay": True})
        global STATE
        STATE = t["state"]
        return jsonify({"task": task_meta(t), "snap": snapshot(), "replay": False})


@app.post("/api/tasks/<tid>/tick")
def api_task_tick(tid):
    with LOCK:
        t = TASKS.get(tid)
        if not t:
            return jsonify({"error": "task not found"}), 404
        if t["status"] == "done":
            return jsonify(copy.deepcopy(t["ticks"][-1]))
        if request.args.get("token") != t["state"]["token"]:
            global STATE
            STATE = t["state"]
            return jsonify(snapshot())      # 不是当前页面的推进请求，只回状态
        return jsonify(copy.deepcopy(record_tick(t)))


@app.post("/api/tasks/<tid>/assign")
def api_task_assign(tid):
    """手动改派：把某个警组指到指定断面，再让 AI 小改一版方案。
    重排原则：只给没人管的拥堵段补位，已经在途 / 在场的不推翻。
    """
    data = request.get_json(silent=True) or {}
    gid = data.get("group")
    try:
        idx = int(data.get("idx"))
    except (TypeError, ValueError):
        return jsonify({"error": "目标断面不合法"}), 400
    with LOCK:
        t = TASKS.get(tid)
        if not t:
            return jsonify({"error": "task not found"}), 404
        if t["status"] == "done":
            return jsonify({"error": "历史任务不能改派"}), 400
        global STATE
        STATE = t["state"]
        g = next((x for x in STATE["groups"] if x["id"] == gid), None)
        if not g:
            return jsonify({"error": "警组不存在"}), 404
        if idx not in STATE["jurisdiction"]:
            return jsonify({"error": "目标不在辖区范围内"}), 400

        stations = STATE.get("lastStations") or compute_stations()
        hot = next((s for s in stations if s["idx"] == idx), None)
        if not hot:
            return jsonify({"error": "目标断面不存在"}), 400
        # 目标附近连续的缓行以上断面组成负责路段，最多 COVER_MAX 个
        zone = [idx]
        for step in (-1, 1):
            cur = idx + step
            while 0 <= cur < len(ROADS) and len(zone) < COVER_MAX:
                nxt = next((s for s in stations if s["idx"] == cur), None)
                if not nxt or nxt["line"] != hot["line"] or nxt["prob"] < 14:
                    break
                zone.append(cur)
                cur += step
        zone.sort()

        for x in STATE["tasks"]:          # 这个组原来的活儿先撤掉
            if x["group"] == gid and x["status"] in ("enroute", "onsite"):
                x["status"] = "cancel"
        g.update(status="idle", station=None, taskId=None, eta=0, remain=0, post=None)
        log(f"手动改派：{g['name']} 被指到 {hot['line']}{hot['sid']}", "cancel")

        send_task(g, hot, zone, manual=True)
        recompute_coverage(stations)
        dispatch(stations)
        recompute_coverage(stations)
        plan_standby(stations)

        if t.get("ticks"):
            t["ticks"][-1] = copy.deepcopy(snapshot())
        save_records()
        return jsonify(copy.deepcopy(snapshot()))


@app.post("/api/tasks/<tid>/play")
def api_task_play(tid):
    with LOCK:
        t = TASKS.get(tid)
        if not t:
            return jsonify({"error": "task not found"}), 404
        if t["status"] == "done":
            return jsonify({"playing": False, "status": "done"})
        t["state"]["playing"] = not t["state"]["playing"]
        return jsonify({"playing": t["state"]["playing"], "status": "running"})


@app.post("/api/tasks/<tid>/reset")
def api_task_reset(tid):
    global STATE
    with LOCK:
        t = TASKS.get(tid)
        if not t:
            return jsonify({"error": "task not found"}), 404
        allowed = [r["idx"] for r in ROADS if r["line"] in t["lines"]]
        st = new_state(group_count=t["groupCount"], jurisdiction=allowed)
        st["taskId"], st["taskName"], st["taskStatus"] = t["id"], t["name"], "running"
        STATE = st
        log(f"任务重置：{t['name']} 回到 06:00，{t['groupCount']} 个警组待命", "info")
        st["lastStations"] = compute_stations(detect=False)
        t["state"] = st
        t["ticks"] = [copy.deepcopy(snapshot())]
        t["status"] = "running"
        save_records()
        return jsonify(copy.deepcopy(snapshot()))


@app.get("/api/tasks/<tid>/replay/<int:i>")
def api_task_replay(tid, i):
    t = TASKS.get(tid)
    if not t:
        return jsonify({"error": "task not found"}), 404
    i = max(0, min(len(t["ticks"]) - 1, i))
    snap = copy.deepcopy(t["ticks"][i])
    snap["replay"] = True
    return jsonify(snap)


@app.delete("/api/tasks/<tid>")
def api_task_delete(tid):
    with LOCK:
        if TASKS.pop(tid, None) is None:
            return jsonify({"error": "task not found"}), 404
        save_records()
        return jsonify({"ok": True})


def boot():
    existed = RECORD_FILE.exists()
    load_records()
    if not TASKS and not existed:      # 第一次启动给一个现成样例，演示可以直接点进去
        create_task("典型样例 · 全线 20 组", ["干线一", "干线二"], GROUP_COUNT)
    atexit.register(save_records)


if __name__ == "__main__":
    boot()
    app.run(host="127.0.0.1", port=8766, debug=False)
