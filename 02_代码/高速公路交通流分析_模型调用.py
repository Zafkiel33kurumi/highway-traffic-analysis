# ===================== 高速公路交通流分析 - 独立模型调用 =====================
# 需要先运行"数据处理与模型训练.py"生成 traffic_congestion_model.pkl
# 加载pkl后即可调用，无需重新定义专家组

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from IPython.display import display, Markdown

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

# ===================== 1. 加载模型（含专家组） =====================
display(Markdown("# 🚀 高速公路交通流 - 模型调用"))
display(Markdown("> 加载一个pkl文件，包含：模型 + 专家组 + 建议库"))

tm = joblib.load("traffic_congestion_model.pkl")
display(Markdown("✅ 模型加载完成（含专家组+建议库）"))

# ===================== 2. 调用演示 =====================
display(Markdown("## 调用演示"))

def print_report(title, result):
    display(Markdown(f"---\n#### {title}"))
    display(Markdown(f"### {result['level_desc']}"))
    display(Markdown(f"**堵车概率**：{result['congestion_prob']}%"))
    display(Markdown(f"**拥堵等级**：{result['level']}"))
    display(Markdown(f"**专家组等级**：{result['expert_level']}"))
    display(Markdown(f"**路况说明**：{result['level_detail']}"))
    display(Markdown(f"**当前条件**：时间{result['factors']['time']}，天气{result['factors']['weather']}（{result['factors']['weather_cat']}），"
                     f"车速{result['factors']['speed']}，占有率{result['factors']['occupancy']}，{result['factors']['incident']}"))
    display(Markdown("**🚗 驾驶员出行建议：**"))
    for i, adv in enumerate(result['driver_advice'], 1):
        display(Markdown(f"{i}. {adv}"))
    display(Markdown("**🚦 交管部门建议：**"))
    for i, adv in enumerate(result['authority_advice'], 1):
        display(Markdown(f"{i}. {adv}"))

r1 = tm.predict(hour_of_day=8, weekday=2, is_weekend=0, flow_veh_5min=90,
                lane_count=2, weather='Rain', incident_flag=1, avg_speed_kmh=30, occupancy_pct=45)
print_report("场景1：早8点，工作日，下雨，有事故", r1)

r2 = tm.predict(hour_of_day=15, weekday=2, is_weekend=0, flow_veh_5min=150,
                lane_count=3, weather='Clear', incident_flag=0, avg_speed_kmh=95, occupancy_pct=12)
print_report("场景2：下午3点，工作日，晴天，无事故", r2)

r3 = tm.predict(hour_of_day=19, weekday=5, is_weekend=1, flow_veh_5min=60,
                lane_count=2, weather='Snow', incident_flag=0, avg_speed_kmh=20, occupancy_pct=55)
print_report("场景3：晚7点，周末，大雪", r3)

r4 = tm.predict(hour_of_day=17, weekday=3, is_weekend=0, flow_veh_5min=130,
                lane_count=3, weather='Clear', incident_flag=0, avg_speed_kmh=50, occupancy_pct=35)
print_report("场景4：下午5点，工作日，晴天，无事故，晚高峰", r4)

# ===================== 3. 批量预测：未来24小时 =====================
display(Markdown("## 批量预测：未来24小时堵车概率（工作日，晴天，无事故）"))

future_list = []
for h in range(24):
    r = tm.predict(hour_of_day=h, weekday=2, is_weekend=0, flow_veh_5min=100,
                   lane_count=3, weather='Clear', incident_flag=0, avg_speed_kmh=70, occupancy_pct=20)
    future_list.append({'小时': f"{h}:00", '堵车概率': f"{r['congestion_prob']}%", '堵车等级': r['level']})

future_df = pd.DataFrame(future_list)
display(future_df)

plt.figure(figsize=(14, 6))
probs = [float(p.replace('%', '')) for p in future_df['堵车概率']]
colors = ['#2ecc71' if p < 30 else '#f1c40f' if p < 50 else '#e67e22' if p < 70 else '#e74c3c' for p in probs]
plt.bar(range(24), probs, color=colors)
plt.xticks(range(24), [f"{h}:00" for h in range(24)], rotation=45)
plt.ylabel('堵车概率（%）')
plt.title('未来24小时堵车概率预测（工作日，晴天，无事故）')
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('未来24小时堵车概率.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`未来24小时堵车概率.png`"))

# ===================== 4. 使用说明 =====================
display(Markdown("## 使用说明"))
display(Markdown("""
**调用方式（只需加载一个pkl）：**
```python
import joblib
tm = joblib.load("traffic_congestion_model.pkl")

result = tm.predict(
    hour_of_day=17,       # 小时（0-23）
    weekday=2,            # 星期（0=周一）
    is_weekend=0,         # 是否周末
    flow_veh_5min=120,    # 5分钟车流量
    lane_count=3,         # 车道数
    weather='Clear',      # 天气
    incident_flag=0,      # 是否有事件
    avg_speed_kmh=65,     # 平均车速
    occupancy_pct=25      # 车道占有率
)

print(result['congestion_prob'])    # 堵车概率（0-100%）
print(result['level'])              # 堵车等级（中文）
print(result['expert_level'])       # 专家组等级
print(result['driver_advice'])      # 驾驶员建议（列表）
print(result['authority_advice'])   # 交管建议（列表）
```

**返回字段说明：**
| 字段 | 说明 |
|------|------|
| `congestion_prob` | 堵车概率（0-100%） |
| `level` | 模型预测的拥堵等级 |
| `expert_level` | 专家组判定的等级 |
| `level_desc` | 等级描述 |
| `level_detail` | 详细说明 |
| `driver_advice` | 驾驶员建议列表 |
| `authority_advice` | 交管部门建议列表 |
| `factors` | 当前条件详情 |
"""))
