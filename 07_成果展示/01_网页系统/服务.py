"""本地服务：提供网页 + 接收 CSV 导入并重新训练。

用法：
    python3 01_网页系统/服务.py
    浏览器打开 http://localhost:8000

导入流程：上传 CSV → 校验列 → 调用 04_模型代码/整体规律模型.py 重训
→ 替换 models/整体规律模型.pkl 和 01_网页系统/prediction_data.js → 页面刷新。
原来的模型和网页数据会自动备份为 .bak。
"""

from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time

from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parents[1]
SITE = BASE / "01_网页系统"
SCRIPT = BASE / "04_模型代码" / "整体规律模型.py"
MODEL = BASE / "04_模型代码" / "models" / "整体规律模型.pkl"
SITE_DATA = SITE / "prediction_data.js"
DATA_DIR = BASE / "03_数据"
PAGE = "交通拥堵预测系统.html"

app = Flask(__name__, static_folder=None)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


def load_script():
    spec = importlib.util.spec_from_file_location("traffic_model", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    body = text[text.index("{"): text.rstrip().rstrip(";").rindex("}") + 1]
    return json.loads(body)


@app.get("/")
def index():
    return send_from_directory(SITE, PAGE)


@app.get("/<path:name>")
def static_file(name):
    return send_from_directory(SITE, name)


@app.post("/api/import")
def import_data():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return jsonify(ok=False, error="请选择 CSV 文件")

    DATA_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    upload = DATA_DIR / f"导入_{stamp}.csv"
    f.save(upload)

    # 1. 校验列和基本取值范围
    try:
        df = load_script().read_any(upload)
    except SystemExit as e:
        return jsonify(ok=False, error=str(e))
    except Exception as e:
        return jsonify(ok=False, error=f"文件读不了：{e}")
    if len(df) < 200:
        return jsonify(ok=False, error=f"只有 {len(df)} 行，至少需要 200 行")
    lanes = set(df.lane_count.dropna().astype(int))
    if not lanes or min(lanes) < 1 or max(lanes) > 8:
        return jsonify(ok=False, error=f"lane_count 超出 1~8：{sorted(lanes)}")

    # 2. 用临时目录接住训练产物，成功了再替换
    with tempfile.TemporaryDirectory() as td:
        out_js = Path(td) / "prediction_data.js"
        out_pkl = Path(td) / "model.pkl"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--data", str(upload),
             "--model-out", str(out_pkl), "--site-out", str(out_js)],
            capture_output=True, text=True, cwd=str(SCRIPT.parent),
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            return jsonify(ok=False, error="训练失败：" + (tail[-1] if tail else "未知错误"))

        payload = read_payload(out_js)
        for p in (SITE_DATA, MODEL):
            if p.exists():
                shutil.copy2(p, p.with_name(p.name + ".bak"))
        shutil.move(str(out_js), SITE_DATA)
        shutil.move(str(out_pkl), MODEL)

    shutil.copy2(upload, DATA_DIR / "当前数据.csv")

    meta = payload["meta"]
    return jsonify(
        ok=True,
        file=f.filename,
        rows=int(len(df)),
        stations=int(df.station_id.nunique()),
        metrics=meta["metrics"],
        classes=len(payload["grid"]),
        period=meta.get("period", ""),
    )


if __name__ == "__main__":
    print(f"服务已启动：http://localhost:8000\n数据目录：{DATA_DIR}")
    app.run(host="127.0.0.1", port=8000, debug=False)
