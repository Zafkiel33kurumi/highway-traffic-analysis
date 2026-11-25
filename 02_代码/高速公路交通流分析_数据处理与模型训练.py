# ===================== 1. 导入所需库 & 全局配置 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import warnings
import joblib
try:
    from IPython.display import display, Markdown
except ImportError:  # 直接用 python 运行（非 Jupyter）时的降级实现
    class Markdown:
        def __init__(self, data):
            self.data = data

    def display(obj):
        print(getattr(obj, 'data', obj))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from scipy.optimize import least_squares, minimize_scalar
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

# ===================== 2. 加载数据集 =====================
DATA_PATH = Path(__file__).resolve().parents[1] / "01_原始数据" / "高速公路交通流传感器数据集.csv"
df = pd.read_csv(DATA_PATH)
display(Markdown("# 📑高速公路交通流传感器数据集分析报告｜三层链式预测模型"))
display(Markdown(f"- 数据集形状：{df.shape[0]}行 × {df.shape[1]}列"))
display(Markdown("### 原始数据前5行"))
display(df.head())
display(Markdown("### 列名列表"))
display(df.columns.tolist())

# ===================== 3. 数据预处理【按照真实数据集字段修复】 =====================
display(Markdown("## 🧹 一、数据预处理"))

display(Markdown("### 3.1 缺失值普查"))
null_counts = df.isnull().sum()
display(null_counts[null_counts > 0].to_frame(name='缺失数量'))

if 'weather' in df.columns:
    df['weather'] = df['weather'].fillna('Unknown')
if 'incident_description' in df.columns:
    df['incident_description'] = df['incident_description'].fillna('无事件描述')

num_cols = df.select_dtypes(include=[np.number]).columns
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# ----------⚠️重点修复：从timestamp衍生时间特征，数据集没有现成weekday/is_weekend/hour字段！----------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour_of_day'] = df['timestamp'].dt.hour   # 原始数据集真实小时字段 hour_of_day
df['minute'] = df['timestamp'].dt.minute
df['weekday'] = df['timestamp'].dt.weekday    # 0周一 ~6周日
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >=5 else 0)

display(Markdown("### 3.2 数值型变量统计检查"))
display(df.describe())

# 数值范围过滤
if 'avg_speed_kmh' in df.columns:
    df.loc[df['avg_speed_kmh'] < 0, 'avg_speed_kmh'] = np.nan
    df.loc[df['avg_speed_kmh'] > 200, 'avg_speed_kmh'] = np.nan
if 'flow_veh_5min' in df.columns:
    df.loc[df['flow_veh_5min'] < 0, 'flow_veh_5min'] = np.nan
if 'occupancy_pct' in df.columns:
    df.loc[df['occupancy_pct'] < 0, 'occupancy_pct'] = np.nan
    df.loc[df['occupancy_pct'] > 100, 'occupancy_pct'] = np.nan

df[num_cols] = num_imputer.fit_transform(df[num_cols])

display(Markdown("### 3.3 数据预处理后验证"))
display(Markdown(f"- 缺失值总数：**{df.isnull().sum().sum()}**"))
display(Markdown(f"- 重复行数：**{df.duplicated().sum()}**"))
display(Markdown(f"- 数据形状：{df.shape[0]}行 × {df.shape[1]}列"))

# ===================== 4. 整体统计分析 =====================
display(Markdown("## 📊 二、整体统计分析"))

if 'flow_veh_5min' in df.columns:
    core_stats = df['flow_veh_5min'].describe().round(2)
    display(Markdown("#### 🚗 5分钟车流量核心统计指标"))
    display(core_stats.to_frame(name="flow_veh_5min"))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['flow_veh_5min'], kde=True, bins=30, color='#1f77b4')
    plt.title('5分钟车流量整体分布')
    plt.xlabel('车流量（辆/5分钟）')
    plt.ylabel('样本数量')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('车流量分布.png', bbox_inches='tight')
    plt.close()
    display(Markdown("✅ 图表已保存：`车流量分布.png`"))

if 'avg_speed_kmh' in df.columns:
    display(Markdown("#### 🚄 平均车速核心统计指标"))
    display(df['avg_speed_kmh'].describe().round(2).to_frame(name="avg_speed_kmh"))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_speed_kmh'], kde=True, bins=30, color='#2ca02c')
    plt.title('平均车速整体分布')
    plt.xlabel('车速（km/h）')
    plt.ylabel('样本数量')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('车速分布.png', bbox_inches='tight')
    plt.close()
    display(Markdown("✅ 图表已保存：`车速分布.png`"))

# ===================== 5. 时间维度分析 (使用hour_of_day) =====================
display(Markdown("## ⏰ 三、时间维度分析"))

hourly_stats = df.groupby('hour_of_day')['flow_veh_5min'].agg(['mean', 'median', 'std']).round(2).reset_index()
display(Markdown("#### 🕒 小时级车流量统计（hour_of_day）"))
display(hourly_stats.sort_values("mean", ascending=False))
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour_of_day', y='mean', data=hourly_stats, marker='o', color='#ff7f0e', linewidth=2)
plt.title('小时级平均车流量趋势')
plt.xlabel('小时')
plt.ylabel('平均车流量（辆/5分钟）')
plt.xticks(range(0, 24))
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('小时级车流量趋势.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`小时级车流量趋势.png`"))

weekend_stats = df.groupby('is_weekend')['flow_veh_5min'].agg(['mean', 'median', 'std', 'count']).round(2)
weekend_stats.index = ["工作日", "周末"]
display(Markdown("#### 📅 工作日 vs 周末车流量"))
display(weekend_stats)
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_weekend', y='flow_veh_5min', data=df, palette=['#2ca02c', '#d62728'])
plt.title('工作日 vs 周末车流量对比')
plt.xlabel("")
plt.ylabel("车流量（辆/5分钟）")
plt.xticks([0, 1], ["工作日", "周末"])
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('工作日周末车流量对比.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`工作日周末车流量对比.png`"))

monthly_stats = df.groupby('month')['flow_veh_5min'].agg(['mean', 'median', 'std']).round(2).reset_index()
display(Markdown("#### 📆 月度车流量统计"))
display(monthly_stats.sort_values("month"))
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='mean', data=monthly_stats, marker='o', color='#9467bd', linewidth=2)
plt.title('月度平均车流量趋势')
plt.xlabel("月份")
plt.ylabel("平均车流量（辆/5分钟）")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('月度车流量趋势.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`月度车流量趋势.png`"))

# ===================== 6. 天气因素影响分析 =====================
display(Markdown("## 🌤️ 四、天气因素影响分析"))

if 'weather' in df.columns:
    weather_stats = df.groupby('weather')['flow_veh_5min'].agg(['mean', 'median', 'std', 'count']).round(2)
    weather_stats = weather_stats.sort_values("mean", ascending=False)
    display(Markdown("#### 🌦️ 不同天气类型车流量"))
    display(weather_stats)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=weather_stats.index, y='mean', data=weather_stats, palette="viridis")
    plt.title("不同天气类型下平均车流量")
    plt.xlabel("天气类型")
    plt.ylabel("平均车流量（辆/5分钟）")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig('不同天气类型车流量.png', bbox_inches='tight')
    plt.close()
    display(Markdown("✅ 图表已保存：`不同天气类型车流量.png`"))

    if 'avg_speed_kmh' in df.columns:
        weather_speed = df.groupby('weather')['avg_speed_kmh'].agg(['mean', 'median']).round(2)
        weather_speed = weather_speed.sort_values("mean", ascending=False)
        display(Markdown("#### 🚄 不同天气类型平均车速"))
        display(weather_speed)

# ===================== 7. 交通事件影响分析 =====================
display(Markdown("## ⚠️ 五、交通事件影响分析"))

if 'incident_flag' in df.columns:
    incident_stats = df.groupby('incident_flag')['flow_veh_5min'].agg(['mean', 'median', 'std', 'count']).round(2)
    incident_stats.index = ["无事件", "有事件"]
    display(Markdown("#### 🚨 有/无事件时车流量对比"))
    display(incident_stats)
    if 'avg_speed_kmh' in df.columns:
        incident_speed = df.groupby('incident_flag')['avg_speed_kmh'].agg(['mean', 'median']).round(2)
        incident_speed.index = ["无事件", "有事件"]
        display(Markdown("#### 🚄 有/无事件时车速对比"))
        display(incident_speed)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='incident_flag', y='flow_veh_5min', data=df, palette=['#2ca02c', '#d62728'])
    plt.title('交通事件对车流量的影响')
    plt.xlabel("")
    plt.ylabel("车流量（辆/5分钟）")
    plt.xticks([0, 1], ["无事件", "有事件"])
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig('交通事件影响.png', bbox_inches='tight')
    plt.close()
    display(Markdown("✅ 图表已保存：`交通事件影响.png`"))

# ===================== 8. 车道数影响分析 =====================
display(Markdown("## 🛣️ 六、车道数影响分析"))
if 'lane_count' in df.columns:
    lane_stats = df.groupby('lane_count')['flow_veh_5min'].agg(['mean', 'median', 'std', 'count']).round(2)
    display(Markdown("#### 不同车道数的车流量"))
    display(lane_stats)

# ===================== 9. 相关性分析 =====================
display(Markdown("## 🔗 七、相关性分析"))
numeric_features = [c for c in ['flow_veh_5min', 'avg_speed_kmh', 'occupancy_pct', 'lane_count',
                                 'hour_of_day', 'is_weekend', 'weekday', 'month'] if c in df.columns]
corr_matrix = df[numeric_features].corr().round(4)
display(Markdown("#### 特征与flow_veh_5min相关系数排序"))
display(corr_matrix['flow_veh_5min'].sort_values(ascending=False).to_frame())
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f', linewidths=0.5)
plt.title("特征相关性热力图")
plt.tight_layout()
plt.savefig('相关性热力图.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`相关性热力图.png`"))

# ===================== 10. 专家组：基于常识的交通拥堵判定【可调试参数区】 =====================
display(Markdown("## 🧠 八、专家组：基于人类常识的交通拥堵判定（权重可调试）"))
# ============【调试专家组参数在这里修改】============
WEIGHT_SPEED = 0.30
WEIGHT_OCC = 0.25
WEIGHT_INCIDENT = 0.15
WEIGHT_TIME = 0.15
WEIGHT_WEATHER = 0.10
LANE_FACTOR = {2:1.2,3:1.0,4:0.8}
WEEKEND_FACTOR = 1.1
# ====================================================

def speed_congestion_score(speed):
    if speed >= 100:   return 1
    elif speed >= 80:  return 2
    elif speed >= 65:  return 3
    elif speed >= 50:  return 4
    elif speed >= 40:  return 5
    elif speed >= 30:  return 6
    elif speed >= 20:  return 7
    elif speed >= 10:  return 8
    elif speed >= 5:   return 9
    else:              return 10

df['speed_congestion_score'] = df['avg_speed_kmh'].apply(speed_congestion_score)

def occupancy_congestion_score(occ):
    if occ <= 5:    return 1
    elif occ <= 10: return 2
    elif occ <= 15: return 3
    elif occ <= 20: return 4
    elif occ <= 30: return 5
    elif occ <= 40: return 6
    elif occ <= 50: return 7
    elif occ <= 65: return 8
    elif occ <= 80: return 9
    else:           return 10

df['occupancy_congestion_score'] = df['occupancy_pct'].apply(occupancy_congestion_score)

time_congestion_map = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 9, 9: 7,
    10: 4, 11: 4, 12: 5, 13: 4, 14: 4, 15: 5, 16: 6, 17: 9, 18: 8, 19: 6,
    20: 4, 21: 3, 22: 2, 23: 1
}
df['time_congestion_score'] = df['hour_of_day'].map(time_congestion_map)

weather_congestion_map = {
    'Clear': 1, 'Sunny': 1, 'Clouds': 2, 'Cloudy': 2, 'Overcast': 2, 'Partly Cloudy': 2,
    'Mist': 3, 'Haze': 3, 'Fog': 4, 'Light Fog': 3,
    'Drizzle': 5, 'Light Rain': 5, 'Showers': 6, 'Rain': 6, 'Moderate Rain': 7, 'Heavy Rain': 8, 'Torrential Rain': 9,
    'Light Snow': 6, 'Snow': 7, 'Moderate Snow': 8, 'Heavy Snow': 9, 'Blizzard': 10, 'Sleet': 8, 'Ice': 10,
    'Thunderstorm': 9, 'T‑Storm': 9, 'Severe Thunderstorm': 10, 'Hail': 9, 'Dust': 7, 'Sand': 7, 'Unknown': 3
}
if 'weather' in df.columns:
    df['weather_congestion_score'] = df['weather'].map(weather_congestion_map).fillna(3)

if 'incident_flag' in df.columns:
    def incident_severity(desc):
        if pd.isna(desc) or desc == '无事件描述':
            return 1
        desc_lower = str(desc).lower()
        if any(kw in desc_lower for kw in ['追尾', '碰撞', '翻车', '爆炸', '火灾', '多车', '伤亡', 'fatal', 'crash', 'multi']):
            return 9
        elif any(kw in desc_lower for kw in ['故障', '抛锚', '违停', '散落物', '施工', 'stall', 'breakdown', 'construction']):
            return 6
        elif any(kw in desc_lower for kw in ['轻微', '剐蹭', '擦碰', 'minor', 'fender']):
            return 4
        else:
            return 5
    df['incident_severity_score'] = df['incident_description'].apply(incident_severity)
    df.loc[(df['incident_flag'] == 1) & (df['incident_severity_score'] == 1), 'incident_severity_score'] = 5
    df.loc[df['incident_flag'] == 0, 'incident_severity_score'] = 1

# 场景组合库
display(Markdown("### 10.6 场景组合库"))
weather_types = ['Clear', 'Clouds', 'Rain', 'Heavy Rain', 'Snow', 'Heavy Snow', 'Fog', 'Thunderstorm']
time_periods = list(range(24))
speed_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
occ_ranges = [(0, 15), (15, 30), (30, 50), (50, 100)]
incident_states = [0, 1]
lane_counts = [2, 3, 4]
day_types = [0, 1]

total_combos = len(weather_types) * len(time_periods) * len(speed_ranges) * len(occ_ranges) * len(incident_states) * len(lane_counts) * len(day_types)
display(Markdown(f"**总组合数：{total_combos} 种场景**"))

scenario_records = []
for weather in weather_types:
    for hour_of_day in time_periods:
        for speed_range in speed_ranges:
            for occ_range in occ_ranges:
                for incident in incident_states:
                    for lanes in lane_counts:
                        for day_type in day_types:
                            w_score = weather_congestion_map.get(weather, 3)
                            t_score = time_congestion_map.get(hour_of_day, 3)
                            s_score = speed_congestion_score((speed_range[0] + speed_range[1]) / 2)
                            o_score = occupancy_congestion_score((occ_range[0] + occ_range[1]) / 2)
                            i_score = 7 if incident == 1 else 1
                            lane_score = LANE_FACTOR.get(lanes, 1.0)
                            day_score = WEEKEND_FACTOR if day_type == 1 else 1.0
                            raw_score = (s_score * WEIGHT_SPEED + o_score * WEIGHT_OCC + i_score * WEIGHT_INCIDENT + t_score * WEIGHT_TIME + w_score * WEIGHT_WEATHER) * lane_score * day_score
                            congestion_prob = min(max(raw_score / 10 * 100, 0), 100)
                            if congestion_prob <= 20: level = '畅通'
                            elif congestion_prob <= 35: level = '基本畅通'
                            elif congestion_prob <= 55: level = '轻度拥堵'
                            elif congestion_prob <= 75: level = '中度拥堵'
                            else: level = '严重拥堵'
                            scenario_records.append({
                                'weather': weather, 'hour_of_day': hour_of_day,
                                'speed_range': f"{speed_range[0]}-{speed_range[1]}km/h",
                                'occupancy_range': f"{occ_range[0]}-{occ_range[1]}%",
                                'incident': '有事件' if incident == 1 else '无事件',
                                'lane_count': lanes,
                                'day_type': '周末' if day_type == 1 else '工作日',
                                'raw_score': round(raw_score, 2),
                                'congestion_prob': round(congestion_prob, 1),
                                'congestion_level': level
                            })

scenario_df = pd.DataFrame(scenario_records)
display(Markdown(f"✅ 场景组合库生成完成，共 **{len(scenario_df)}** 种场景"))

expert_features = [c for c in ['speed_congestion_score', 'occupancy_congestion_score',
                                'time_congestion_score', 'weather_congestion_score',
                                'incident_severity_score'] if c in df.columns]
if expert_features:
    display(df[expert_features].describe().round(2))

# ===================== 11. 综合拥堵评分体系 =====================
display(Markdown("## 📊 九、综合交通拥堵评分体系"))

df['congestion_score'] = (
    df['speed_congestion_score'] * WEIGHT_SPEED +
    df['occupancy_congestion_score'] * WEIGHT_OCC +
    df['incident_severity_score'] * WEIGHT_INCIDENT +
    df['time_congestion_score'] * WEIGHT_TIME +
    df['weather_congestion_score'] * WEIGHT_WEATHER
)

df['congestion_level'] = pd.cut(
    df['congestion_score'],
    bins=[0, 2.5, 4.0, 5.5, 7.0, 10.5],
    labels=['畅通', '基本畅通', '轻度拥堵', '中度拥堵', '严重拥堵']
)

display(Markdown("### 堵车等级分布"))
congestion_dist = df['congestion_level'].value_counts().sort_index()
display(congestion_dist.to_frame(name='样本数量'))

plt.figure(figsize=(8, 8))
colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
congestion_dist.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('交通拥堵等级分布')
plt.ylabel('')
plt.tight_layout()
plt.savefig('堵车等级分布.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`堵车等级分布.png`"))

# ===================== 12. 出行建议系统 =====================
display(Markdown("## 💡 十、出行建议系统"))

def generate_travel_advice(level, weather, incident_flag):
    advice_dict = {
        "畅通": "当前路况畅通，适合出行。",
        "基本畅通": "路况基本正常，可正常出行。",
        "轻度拥堵": "存在轻度拥堵，建议适当提前出发或选择替代路线。",
        "中度拥堵": "中度拥堵，建议错峰出行或选择公共交通。预计延误20‑40分钟。",
        "严重拥堵": "严重拥堵，强烈建议推迟出行或改乘地铁/公交。预计延误40分钟以上。"
    }
    advice = advice_dict.get(level, "路况未知，请谨慎出行")
    weather_advice = ""
    if weather in ['Rain', 'Heavy Rain', 'Showers', 'Light Rain']:
        weather_advice = " 雨天路滑，请减速慢行，保持安全车距。"
    elif weather in ['Snow', 'Heavy Snow', 'Light Snow', 'Sleet']:
        weather_advice = " 雪天路面结冰，尽量避免驾车出行。"
    elif weather in ['Thunderstorm', 'T‑Storm']:
        weather_advice = " 雷暴天气，建议留在室内，避免出行。"
    elif weather in ['Fog', 'Mist', 'Haze']:
        weather_advice = " 能见度较低，请打开雾灯，注意行车安全。"
    incident_advice = " 该路段有交通事件，请注意绕行。" if incident_flag == 1 else ""
    return advice + weather_advice + incident_advice

df['travel_advice'] = df.apply(
    lambda r: generate_travel_advice(r['congestion_level'], r.get('weather', 'Unknown'), r.get('incident_flag', 0)),
    axis=1)

# ===================== 13. 高峰时段识别 =====================
display(Markdown("## 🕐 十一、高峰时段识别与避堵推荐"))

peak_analysis = df.groupby(['hour_of_day', 'is_weekend']).agg(
    avg_flow=('flow_veh_5min', 'mean'),
    avg_speed=('avg_speed_kmh', 'mean'),
    avg_congestion=('congestion_score', 'mean'),
    congestion_ratio=('congestion_level', lambda x: (x.isin(['中度拥堵', '严重拥堵'])).mean())
).round(3).reset_index()
peak_analysis['day_type'] = peak_analysis['is_weekend'].apply(lambda x: '周末' if x == 1 else '工作日')

display(Markdown("### 工作日各时段拥堵情况"))
workday_peaks = peak_analysis[peak_analysis['is_weekend'] == 0].sort_values('avg_congestion', ascending=False)
display(workday_peaks[['hour_of_day', 'avg_flow', 'avg_congestion', 'congestion_ratio']].head(10))

display(Markdown("### 周末各时段拥堵情况"))
weekend_peaks = peak_analysis[peak_analysis['is_weekend'] == 1].sort_values('avg_congestion', ascending=False)
display(weekend_peaks[['hour_of_day', 'avg_flow', 'avg_congestion', 'congestion_ratio']].head(10))

workday_best = workday_peaks.sort_values('avg_congestion').head(3)
weekend_best = weekend_peaks.sort_values('avg_congestion').head(3)
display(Markdown(f"""
### 🎯 最佳出行时段推荐
- **工作日推荐**：{', '.join([str(int(h)) + ':00' for h in workday_best['hour_of_day'].values])} 出行最顺畅
- **周末推荐**：{', '.join([str(int(h)) + ':00' for h in weekend_best['hour_of_day'].values])} 出行最顺畅
"""))

# ===================== 14. 三层链式模型训练【修复真实字段：hour_of_day】 =====================
display(Markdown("# 🤖 十二、三层链式预测模型训练"))
display(Markdown("""
> **模型A：前置预测** 输入：weather、hour_of_day、weekday、is_weekend、lane_count
> 输出4个：occupancy_pct、flow_veh_5min、avg_speed_kmh、incident_flag（交通事故0/1）
>
> **模型B：中间推理** 输入模型A预测的占有率/流量/事件（**不含车速**），输出：congestion_prob_1（P(≥轻度拥堵)）
>
> **模型C：基线直接预测** 原始输入直接预测 congestion_prob_2（P(≥轻度拥堵)）
>
> 概率口径：校准后 predict_proba 中「等级 ≥ 轻度拥堵」的概率之和 × 100
>
> 最终融合：final_congestion_prob = 0.6 * congestion_prob_1 + 0.4 * congestion_prob_2
"""))

# 【⚠️ 修复：基础输入特征，小时字段使用 hour_of_day】
base_input_cols = ['weather','hour_of_day','weekday','is_weekend','lane_count']
level_map = {'畅通':0,'基本畅通':1,'轻度拥堵':2,'中度拥堵':3,'严重拥堵':4}
level_reverse_map = {v: k for k, v in level_map.items()}
df['congestion_label'] = df['congestion_level'].map(level_map)

all_model_cols = base_input_cols + ['occupancy_pct','flow_veh_5min','avg_speed_kmh','incident_flag','congestion_label']
df_model = df.dropna(subset=all_model_cols).copy()

# 获取真实存在标签，自适应类别数量
valid_labels = sorted(df_model['congestion_label'].unique())
valid_level_names = [level_reverse_map[i] for i in valid_labels]
class_num = len(valid_labels)

display(Markdown(f"✅ 当前数据真实拥堵类别数量：**{class_num}**"))
display(pd.Series(df_model['congestion_label']).value_counts().sort_index().to_frame("样本数"))
CONGESTION_LEVEL_THRESHOLD = 2   # 轻度拥堵及以上视为"拥堵"
display(Markdown(f"✅ 拥堵概率口径：P(等级 ≥ {level_reverse_map[CONGESTION_LEVEL_THRESHOLD]})，由校准后的 predict_proba 求和得到"))

# --------模型A：前置多输出（3个回归+1个分类）---------
X_A = df_model[base_input_cols].copy()
yA_occ = df_model['occupancy_pct']
yA_flow = df_model['flow_veh_5min']
yA_speed = df_model['avg_speed_kmh']
yA_incident = df_model['incident_flag']

X_A_train, X_A_test = train_test_split(X_A, test_size=0.3, random_state=42)

cat_cols_A = ['weather']
prep_A = ColumnTransformer([
    ("cat",OneHotEncoder(sparse_output=False,handle_unknown='ignore'),cat_cols_A)
],remainder="passthrough")

pipe_occ = Pipeline([("prep",prep_A),("rf",RandomForestRegressor(n_estimators=120,max_depth=14,random_state=42))])
pipe_flow = Pipeline([("prep",prep_A),("rf",RandomForestRegressor(n_estimators=120,max_depth=14,random_state=42))])
pipe_speed = Pipeline([("prep",prep_A),("rf",RandomForestRegressor(n_estimators=120,max_depth=14,random_state=42))])
pipe_incident = Pipeline([("prep",prep_A),("rf",RandomForestClassifier(n_estimators=120,max_depth=14,class_weight="balanced",random_state=42))])

pipe_occ.fit(X_A_train,yA_occ.loc[X_A_train.index])
pipe_flow.fit(X_A_train,yA_flow.loc[X_A_train.index])
pipe_speed.fit(X_A_train,yA_speed.loc[X_A_train.index])
pipe_incident.fit(X_A_train,yA_incident.loc[X_A_train.index])

joblib.dump({
    "pipe_occ":pipe_occ,
    "pipe_flow":pipe_flow,
    "pipe_speed":pipe_speed,
    "pipe_incident":pipe_incident,
    "valid_labels":valid_labels
},"modelA.pkl")
display(Markdown("✅模型A(前置多输出)保存 modelA.pkl"))

# 评估模型A
pred_occ = pipe_occ.predict(X_A_test)
pred_flow = pipe_flow.predict(X_A_test)
pred_speed = pipe_speed.predict(X_A_test)
pred_incident = pipe_incident.predict(X_A_test)

display(Markdown("#### 模型A评估"))
display(Markdown(f"occupancy_pct RMSE: {np.sqrt(mean_squared_error(yA_occ.loc[X_A_test.index],pred_occ)):.2f} R2:{r2_score(yA_occ.loc[X_A_test.index],pred_occ):.3f}"))
display(Markdown(f"flow_veh_5min RMSE: {np.sqrt(mean_squared_error(yA_flow.loc[X_A_test.index],pred_flow)):.2f} R2:{r2_score(yA_flow.loc[X_A_test.index],pred_flow):.3f}"))
display(Markdown(f"avg_speed_kmh RMSE: {np.sqrt(mean_squared_error(yA_speed.loc[X_A_test.index],pred_speed)):.2f} R2:{r2_score(yA_speed.loc[X_A_test.index],pred_speed):.3f}"))
display(Markdown("##### 交通事故分类报告"))
print(classification_report(yA_incident.loc[X_A_test.index],pred_incident))

# --------模型B：中间链路：输入【occupancy_pct,flow_veh_5min,avg_speed_kmh,incident_flag】预测拥堵等级---------
input_B_cols = ['occupancy_pct','flow_veh_5min','incident_flag']
X_B = df_model[input_B_cols]
y_B = df_model['congestion_label']
X_B_train,X_B_test,yB_train,yB_test = train_test_split(X_B,y_B,test_size=0.3,random_state=42,stratify=y_B)

pipe_B = Pipeline([
    ("pre",ColumnTransformer([],remainder="passthrough")),
    ("clf",CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=150,max_depth=14,class_weight="balanced",random_state=42),
        method="sigmoid", cv=3))
])
pipe_B.fit(X_B_train,yB_train)
yB_pred = pipe_B.predict(X_B_test)
joblib.dump(pipe_B,"modelB.pkl")
display(Markdown("✅模型B(中间链路)保存 modelB.pkl"))
display(Markdown("#### 模型B评估"))
print(classification_report(yB_test,yB_pred,target_names=valid_level_names,labels=valid_labels))

# --------模型C：基线链路，原始基础特征直接预测拥堵等级---------
X_C = df_model[base_input_cols]
y_C = df_model['congestion_label']
X_C_train,X_C_test,yC_train,yC_test = train_test_split(X_C,y_C,test_size=0.3,random_state=42,stratify=y_C)

prep_C = ColumnTransformer([
    ("cat",OneHotEncoder(sparse_output=False,handle_unknown='ignore'),['weather'])
],remainder="passthrough")
pipe_C = Pipeline([
    ("pre",prep_C),
    ("clf",CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=150,max_depth=14,class_weight="balanced",random_state=42),
        method="sigmoid", cv=3))
])
pipe_C.fit(X_C_train,yC_train)
yC_pred = pipe_C.predict(X_C_test)
joblib.dump(pipe_C,"modelC.pkl")
display(Markdown("✅模型C(基线直接预测)保存 modelC.pkl"))
display(Markdown("#### 模型C评估"))
print(classification_report(yC_test,yC_pred,target_names=valid_level_names,labels=valid_labels))

# 全局加载模型（推理函数使用）
modelA = joblib.load("modelA.pkl")
pipe_occ = modelA['pipe_occ']
pipe_flow = modelA['pipe_flow']
pipe_speed = modelA['pipe_speed']
pipe_incident = modelA['pipe_incident']
valid_labels_global = modelA['valid_labels']
levels_global = np.array(modelA['valid_labels'])


def congestion_probability(proba, levels):
    """把多分类概率聚合成「至少轻度拥堵」的真实概率（0~1）。
    只对模型实际训练过的类别求和，不再用类别序号充当权重。"""
    proba = np.asarray(proba).ravel()
    levels = np.asarray(levels).ravel()
    return float(proba[levels >= CONGESTION_LEVEL_THRESHOLD].sum())

modelB = joblib.load("modelB.pkl")
modelC = joblib.load("modelC.pkl")


def predict_full_chain(input_dict):
    """
    input_dict keys: weather, hour_of_day, weekday, is_weekend, lane_count
    ⚠️小时字段：hour_of_day（0‑23），不再使用hour
    """
    in_df = pd.DataFrame([input_dict])
    # 模型A推理
    pred_occ = np.clip(pipe_occ.predict(in_df)[0],0,100)
    pred_flow = np.clip(pipe_flow.predict(in_df)[0],0,9999)
    pred_speed = np.clip(pipe_speed.predict(in_df)[0],0,200)
    pred_incident_prob = pipe_incident.predict_proba(in_df)[0,1]
    pred_incident_flag = 1 if pred_incident_prob >=0.5 else 0

    # 模型B输入
    B_input = pd.DataFrame([{
        "occupancy_pct":pred_occ,
        "flow_veh_5min":pred_flow,
        "avg_speed_kmh":pred_speed,
        "incident_flag":pred_incident_flag
    }])
    prob_B = modelB.predict_proba(B_input)[0]
    congestion_prob_1 = congestion_probability(prob_B, levels_global) * 100

    # 模型C
    prob_C = modelC.predict_proba(in_df)[0]
    congestion_prob_2 = congestion_probability(prob_C, levels_global) * 100

    final_prob = np.clip(0.6 * congestion_prob_1 + 0.4 * congestion_prob_2, 0,100)

    if final_prob <=20:
        level = "畅通"
    elif final_prob <=35:
        level = "基本畅通"
    elif final_prob <=55:
        level = "轻度拥堵"
    elif final_prob <=75:
        level = "中度拥堵"
    else:
        level = "严重拥堵"

    advice = generate_travel_advice(level, input_dict["weather"], pred_incident_flag)
    return {
        "pred_occupancy_pct":round(pred_occ,2),
        "pred_flow_veh_5min":round(pred_flow,1),
        "pred_avg_speed_kmh":round(pred_speed,1),
        "pred_incident_prob":round(pred_incident_prob,3),
        "pred_incident_flag":pred_incident_flag,
        "congestion_prob_1_modelB":round(congestion_prob_1,2),
        "congestion_prob_2_modelC":round(congestion_prob_2,2),
        "final_congestion_prob":round(final_prob,2),
        "congestion_level":level,
        "travel_advice":advice
    }

display(Markdown("## 🧪 三层链式模型接口演示测试【⚠️小时字段：hour_of_day】"))
test_input = {
    "weather":"Light Rain",
    "hour_of_day":17,
    "weekday":2,
    "is_weekend":0,
    "lane_count":3
}
res = predict_full_chain(test_input)
display(res)

# ===================== 15. 物理模型：容量因子 + BPR 阻抗函数 =====================
display(Markdown("## 🧮 十四、物理模型：车道数 / 天气 / 事件 → 有效通行能力 → 车速"))

# 【可调试参数区】
CAPACITY_PER_LANE_5MIN = 2200 / 12          # 单车道 2200 veh/h，折算为 veh/5min ≈ 183.3

# 天气对通行能力的折减系数（HCM 与实证研究的常用区间）
WEATHER_CAPACITY_FACTOR = {
    'Clear': 1.00, 'Sunny': 1.00,
    'Clouds': 0.98, 'Cloudy': 0.98, 'Overcast': 0.98, 'Partly Cloudy': 0.98,
    'Mist': 0.94, 'Haze': 0.94, 'Light Fog': 0.92, 'Fog': 0.90,
    'Drizzle': 0.92, 'Light Rain': 0.92, 'Showers': 0.90, 'Rain': 0.88,
    'Moderate Rain': 0.85, 'Heavy Rain': 0.80, 'Torrential Rain': 0.70,
    'Light Snow': 0.85, 'Snow': 0.78, 'Moderate Snow': 0.72,
    'Heavy Snow': 0.65, 'Blizzard': 0.58, 'Sleet': 0.68, 'Ice': 0.55,
    'Thunderstorm': 0.70, 'T‑Storm': 0.70, 'Severe Thunderstorm': 0.60,
    'Hail': 0.65, 'Dust': 0.85, 'Sand': 0.85, 'Unknown': 1.00,
}

# 交通事件对通行能力的折减（占用一条车道即损失约 1/N 的通行能力）
INCIDENT_CAPACITY_FACTOR = {0: 1.00, 1: 0.75}

# 车道数的通行效率（多车道存在汇入汇出与换道干扰）
LANE_EFFICIENCY = {2: 0.95, 3: 1.00, 4: 1.02, 5: 1.03}

CONGESTION_SPEED_THRESHOLD = 60.0           # 判定拥堵的车速阈值（km/h）


def capacity_factor(lanes, weather, incident):
    """三个容量因子的乘积：车道效率 × 天气折减 × 事件折减"""
    return (LANE_EFFICIENCY.get(int(lanes), 1.0)
            * WEATHER_CAPACITY_FACTOR.get(weather, 1.0)
            * INCIDENT_CAPACITY_FACTOR.get(int(incident), 1.0))


def effective_capacity(lanes, weather, incident):
    """有效通行能力 = 单车道基准 × 车道数 × 容量因子（veh/5min）"""
    return CAPACITY_PER_LANE_5MIN * int(lanes) * capacity_factor(lanes, weather, incident)


def bpr_speed(flow, lanes, weather, incident, v_free, alpha, beta):
    """BPR 阻抗函数：v = V_free / (1 + α · (q / C_eff)^β)"""
    capacity = effective_capacity(lanes, weather, incident)
    ratio = np.clip(np.asarray(flow, dtype=float) / np.maximum(capacity, 1e-6), 0, None)
    return v_free / (1.0 + alpha * ratio ** beta)


# ---------- 15.1 用数据标定 V_free / α / β ----------
lanes_arr = df['lane_count'].astype(int).values
weather_arr = df['weather'].astype(str).values
incident_arr = df['incident_flag'].astype(int).values
flow_arr = df['flow_veh_5min'].astype(float).values
speed_arr = df['avg_speed_kmh'].astype(float).values

capacity_arr = np.array([effective_capacity(l, w, i) for l, w, i in zip(lanes_arr, weather_arr, incident_arr)])
ratio_arr = flow_arr / capacity_arr


def _bpr_resid(p):
    v_free, alpha, beta = p
    return v_free / (1.0 + alpha * ratio_arr ** beta) - speed_arr


fit = least_squares(_bpr_resid, x0=[110.0, 0.15, 4.0],
                    bounds=([60.0, 1e-3, 0.5], [200.0, 50.0, 15.0]))
V_FREE, ALPHA, BETA = fit.x
pred_phys = V_FREE / (1.0 + ALPHA * ratio_arr ** BETA)

# 专家加法基线（去掉车速与占有率，仅保留时段/天气/事件/车道）
def _expert_prior(hour_series):
    score = (0.15 * hour_series.map(time_congestion_map).fillna(3)
             + 0.10 * df['weather_congestion_score'].fillna(3)
             + 0.15 * df['incident_severity_score'].fillna(1)) / 0.40
    return score * df['lane_count'].map(LANE_FACTOR).fillna(1.0)

hour_raw = pd.read_csv(DATA_PATH, usecols=['hour_of_day'])['hour_of_day']
pred_expert = np.polyval(np.polyfit(_expert_prior(df['hour_of_day']), speed_arr, 1), _expert_prior(df['hour_of_day']))
rows = []
for name, pred in [
    ("物理模型 BPR（容量因子）", pred_phys),
    ("专家组加法（原始 hour_of_day）", np.polyval(np.polyfit(_expert_prior(hour_raw), speed_arr, 1), _expert_prior(hour_raw))),
    ("专家组加法（当前 hour_of_day，被 timestamp 覆盖）", pred_expert),
]:
    rows.append({"模型": name,
                 "R²": round(r2_score(speed_arr, pred), 4),
                 "RMSE": round(float(np.sqrt(mean_squared_error(speed_arr, pred))), 2),
                 "MAE": round(float(mean_absolute_error(speed_arr, pred)), 2)})
display(Markdown(f"**标定结果：V_free = {V_FREE:.1f} km/h，α = {ALPHA:.3f}，β = {BETA:.2f}**"))
display(pd.DataFrame(rows))

# ---------- 15.2 反推各天气的容量因子（诊断数据是否真的存在天气效应）----------
base_no_weather = (CAPACITY_PER_LANE_5MIN * lanes_arr
                   * np.array([LANE_EFFICIENCY.get(l, 1.0) for l in lanes_arr])
                   * np.array([INCIDENT_CAPACITY_FACTOR.get(i, 1.0) for i in incident_arr]))
s_base = flow_arr / base_no_weather

diag = []
for w, idx in df.groupby('weather').groups.items():
    idx = np.asarray(idx)
    q_w, v_w, s_w = flow_arr[idx], speed_arr[idx], s_base[idx]

    def _sse(k):
        return float(np.sum((V_FREE / (1.0 + ALPHA * (s_w / k) ** BETA) - v_w) ** 2))

    k_fit = minimize_scalar(_sse, bounds=(0.4, 2.5), method='bounded').x
    diag.append({"天气": w, "样本数": len(idx),
                 "专家设定因子": WEATHER_CAPACITY_FACTOR.get(w, 1.0),
                 "数据反推因子": round(float(k_fit), 3)})
display(Markdown("### 15.2 各天气的容量因子：专家设定 vs 数据反推"))
display(pd.DataFrame(diag))

# ---------- 15.3 拥堵概率（连续值）----------
resid_std = float(np.std(speed_arr - pred_phys))
prob_scale = resid_std * np.sqrt(3) / np.pi      # 逻辑分布与正态的尺度换算
prob_phys = 1.0 / (1.0 + np.exp((pred_phys - CONGESTION_SPEED_THRESHOLD) / prob_scale))
y_true = (speed_arr < CONGESTION_SPEED_THRESHOLD).astype(int)
auc_phys = roc_auc_score(y_true, prob_phys)
display(Markdown(f"**拥堵概率（连续）：AUC = {auc_phys:.4f}，残差标准差 = {resid_std:.2f} km/h**"))

# ---------- 15.4 场景推演：不同容量因子下的车速与拥堵概率 ----------
scen = [(2, 'Clear', 0, 240), (2, 'Clear', 0, 520), (5, 'Clear', 0, 520),
        (3, 'Clear', 0, 520), (3, 'Rain', 0, 520), (3, 'Snow', 0, 520),
        (3, 'Snow', 1, 520), (2, 'Snow', 1, 600)]
scen_rows = []
for ln, wt, inc, qv in scen:
    cap = effective_capacity(ln, wt, inc)
    v = float(bpr_speed(qv, ln, wt, inc, V_FREE, ALPHA, BETA))
    p = float(1.0 / (1.0 + np.exp((v - CONGESTION_SPEED_THRESHOLD) / prob_scale)))
    scen_rows.append({"车道数": ln, "天气": wt, "事件": inc, "5分钟流量": qv,
                      "有效容量": round(cap, 1), "饱和度": round(qv / cap, 3),
                      "预测车速": round(v, 1), "拥堵概率": f"{p * 100:.1f}%"})
display(Markdown("### 15.4 场景推演"))
display(pd.DataFrame(scen_rows))

# ===================== 16. 生成Excel报告 =====================
display(Markdown("## 📑 十三、生成Excel分析报告"))
wb = Workbook()
wb.remove(wb.active)

header_font = Font(name='SimHei', size=12, bold=True, color='FFFFFF')
header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
center_alignment = Alignment(horizontal='center', vertical='center')
border = Border(left=Side('thin'), right=Side('thin'), top=Side('thin'), bottom=Side('thin'))

def write_sheet(wb, sheet_name, headers, data, col_widths=None):
    ws = wb.create_sheet(sheet_name)
    for col_idx, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col_idx, value=h)
        c.font, c.fill, c.alignment, c.border = header_font, header_fill, center_alignment, border
    for row_idx, row_data in enumerate(data, 2):
        for col_idx, val in enumerate(row_data, 1):
            c = ws.cell(row=row_idx, column=col_idx, value=val)
            c.alignment, c.border = center_alignment, border
    if col_widths:
        for i, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

headers = df.columns.tolist()
write_sheet(wb, '原始数据概览', headers, df.head(100).values.tolist(), [15] * len(headers))
write_sheet(wb, '整体统计分析', ['指标', '数值'], [
    ['总样本量', len(df)],
    ['5分钟车流量均值', round(df['flow_veh_5min'].mean(), 2)],
    ['5分钟车流量中位数', round(df['flow_veh_5min'].median(), 2)],
    ['平均车速均值', round(df['avg_speed_kmh'].mean(), 2)] if 'avg_speed_kmh' in df.columns else ['平均车速', 'N/A'],
], [20, 20])
write_sheet(wb, '小时级车流量分析', ['小时', '平均流量', '中位数流量', '流量标准差'], hourly_stats.values.tolist(), [18] * 4)
write_sheet(wb, '工作日周末对比', ['类型', '平均流量', '中位数流量', '流量标准差', '样本数量'], weekend_stats.reset_index().values.tolist(), [18] * 5)
if 'weather' in df.columns:
    write_sheet(wb, '天气影响分析', ['天气类型', '平均流量', '中位数流量', '流量标准差', '样本数量'], weather_stats.reset_index().values.tolist(), [18] * 5)
write_sheet(wb, '堵车等级分布', ['堵车等级', '样本数量', '占比'],
            [[level, count, f"{count / len(df) * 100:.1f}%"] for level, count in zip(congestion_dist.index, congestion_dist.values)], [18, 12, 12])

excel_file = "高速公路交通流数据分析报告_三层模型.xlsx"
wb.save(excel_file)
display(Markdown(f"✅ Excel报告已生成：`{excel_file}`"))

# ===================== 16. 最终总结 =====================
display(Markdown("## 📌 十四、最终总结"))
display(Markdown("""
### 本文件完成内容：
1. 数据预处理（基于真实数据集，timestamp衍生weekday/is_weekend/hour_of_day）
2. 多维度统计分析（时间/天气/事件/车道数）
3. 专家组场景组合库，专家组权重参数开放可调试
4. 综合拥堵评分 + 五级分类
5. 出行建议系统
6. 高峰时段识别与避堵推荐
7. **三层链式预测模型**
    - 模型A输入：weather、hour_of_day、weekday、is_weekend、lane_count；推演占有率、车流量、车速、交通事故概率
    - 模型B：A推演指标预测堵车概率（链路1）
    - 模型C：原始特征直接预测堵车概率（链路2基线）
    - 对外接口`predict_full_chain()`入参字段：`weather, hour_of_day, weekday, is_weekend, lane_count`
8. 输出pkl模型：modelA.pkl、modelB.pkl、modelC.pkl
9. Excel分析报告、绘图png图表输出
"""))
display(Markdown("🎉 **三层链式模型训练完成！测试调用示例：predict_full_chain(dict(weather='Light Rain',hour_of_day=17,weekday=2,is_weekend=0,lane_count=3))**"))
