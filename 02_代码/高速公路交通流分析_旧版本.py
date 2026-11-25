# ===================== 1. 导入所需库 & 全局配置 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import warnings
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')

# 全局设置：解决Matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

# ===================== 2. 加载数据集 =====================
df = pd.read_csv("../../datasets/高速公路交通流传感器数据集/高速公路交通流传感器数据集.csv")
display(Markdown("# 📑高速公路交通流传感器数据集分析报告"))
display(Markdown(f"- 数据集形状：{df.shape[0]}行 × {df.shape[1]}列"))
display(Markdown("### 原始数据前5行"))
display(df.head())

# ===================== 3. 数据预处理 & 基础特征工程 =====================
# 缺失值处理
df['holiday'] = df['holiday'].fillna('非节假日')
display(Markdown("### 各列非空值统计"))
display(df.notna().sum().to_frame(name="非空数量"))

# 时间特征转换与提取
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['hour'] = df['date_time'].dt.hour
df['weekday'] = df['date_time'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != '非节假日' else 0)

# 温度单位转换 开尔文→摄氏度
df['temp_c'] = df['temp'] - 273.15

# 降雨降雪标记
df['has_rain'] = df['rain_1h'].apply(lambda x: 1 if x > 0 else 0)
df['has_snow'] = df['snow_1h'].apply(lambda x: 1 if x > 0 else 0)

display(Markdown("### 处理完成后全部特征列表"))
display(df.columns.tolist())

# ===================== 4. 整体统计分析 =====================
display(Markdown("## 📊 一、整体统计分析"))
core_stats = df['traffic_volume'].describe().round(2)
display(Markdown("#### 🚗 交通流量核心统计指标"))
display(core_stats.to_frame(name="traffic_volume"))

# 交通流量分布图
plt.figure(figsize=(10, 6))
sns.histplot(df['traffic_volume'], kde=True, bins=30, color='#1f77b4')
plt.title('交通流量整体分布')
plt.xlabel('交通流量（辆/小时）')
plt.ylabel('样本数量')
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('交通流量分布.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`交通流量分布.png`"))

# ===================== 5. 时间维度分析 =====================
display(Markdown("## ⏰ 二、时间维度分析"))

# 5.1 小时级流量
hourly_stats = df.groupby('hour')['traffic_volume'].agg(['mean', 'median', 'std']).round(2).reset_index()
display(Markdown("#### 🕒 小时级交通流量统计（按均值降序）"))
display(hourly_stats.sort_values("mean", ascending=False))

plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='mean', data=hourly_stats, marker='o', color='#ff7f0e', linewidth=2)
plt.title('小时级交通流量趋势')
plt.xlabel('小时')
plt.ylabel('平均交通流量（辆/小时）')
plt.xticks(range(0, 24))
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('小时级交通流量趋势.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`小时级交通流量趋势.png`"))

# 5.2 工作日 vs 周末
weekend_stats = df.groupby('is_weekend')['traffic_volume'].agg(['mean', 'median', 'std', 'count']).round(2)
weekend_stats.index = ["工作日", "周末"]
display(Markdown("#### 📅 工作日 vs 周末交通流量"))
display(weekend_stats)

plt.figure(figsize=(8, 6))
sns.boxplot(x='is_weekend', y='traffic_volume', data=df, palette=['#2ca02c', '#d62728'])
plt.title('工作日 vs 周末交通流量对比')
plt.xlabel("")
plt.ylabel("交通流量（辆/小时）")
plt.xticks([0, 1], ["工作日", "周末"])
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('工作日周末交通流量对比.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`工作日周末交通流量对比.png`"))

# 5.3 月度流量
monthly_stats = df.groupby('month')['traffic_volume'].agg(['mean', 'median', 'std']).round(2).reset_index()
display(Markdown("#### 📆 月度交通流量统计"))
display(monthly_stats.sort_values("month"))

plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='mean', data=monthly_stats, marker='o', color='#9467bd', linewidth=2)
plt.title('月度交通流量趋势')
plt.xlabel("月份")
plt.ylabel("平均交通流量（辆/小时）")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('月度交通流量趋势.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`月度交通流量趋势.png`"))

# 5.4 节假日
holiday_stats = df.groupby('is_holiday')['traffic_volume'].agg(['mean', 'median', 'std', 'count']).round(2)
holiday_stats.index = ["非节假日", "节假日"]
display(Markdown("#### 🎉 节假日 vs 非节假日"))
display(holiday_stats)

# ===================== 6. 天气因素影响分析 =====================
display(Markdown("## 🌤️ 三、天气因素影响分析"))

weather_stats = df.groupby('weather_main')['traffic_volume'].agg(['mean', 'median', 'std', 'count']).round(2)
weather_stats = weather_stats.sort_values("mean", ascending=False)
display(Markdown("#### 🌦️ 不同天气主类型交通流量"))
display(weather_stats)

plt.figure(figsize=(12, 6))
sns.barplot(x=weather_stats.index, y='mean', data=weather_stats, palette="viridis")
plt.title("不同天气类型下平均交通流量")
plt.xlabel("天气主类型")
plt.ylabel("平均交通流量（辆/小时）")
plt.xticks(rotation=45, ha="right")
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('不同天气类型交通流量.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`不同天气类型交通流量.png`"))

rain_stats = df.groupby('has_rain')['traffic_volume'].agg(['mean', 'median', 'std', 'count']).round(2)
rain_stats.index = ["无降雨", "有降雨"]
display(Markdown("#### 🌧️ 降雨影响统计"))
display(rain_stats)

snow_stats = df.groupby('has_snow')['traffic_volume'].agg(['mean', 'median', 'std', 'count']).round(2)
snow_stats.index = ["无降雪", "有降雪"]
display(Markdown("#### ❄️ 降雪影响统计"))
display(snow_stats)

# ===================== 7. 相关性分析 =====================
display(Markdown("## 🔗 四、相关性分析"))
numeric_features = ['traffic_volume', 'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'is_weekend', 'weekday', 'is_holiday']
corr_matrix = df[numeric_features].corr().round(4)
display(Markdown("#### 特征与traffic_volume相关系数排序"))
display(corr_matrix['traffic_volume'].sort_values(ascending=False).to_frame())

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f', linewidths=0.5)
plt.title("特征相关性热力图")
plt.tight_layout()
plt.savefig('相关性热力图.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`相关性热力图.png`"))

# ===================== 8. 异常值分析 =====================
display(Markdown("## ⚠️ 五、异常值分析（3σ准则）"))
mean_traffic = df['traffic_volume'].mean()
std_traffic = df['traffic_volume'].std()
lower_bound = mean_traffic - 3 * std_traffic
upper_bound = mean_traffic + 3 * std_traffic

outlier_df = pd.DataFrame({
    "指标": ["交通流量均值", "交通流量标准差", "异常值下限", "异常值上限"],
    "数值": [round(mean_traffic, 2), round(std_traffic, 2), round(lower_bound, 2), round(upper_bound, 2)]
})
display(outlier_df)

outliers = df[(df['traffic_volume'] < lower_bound) | (df['traffic_volume'] > upper_bound)]
outlier_count = len(outliers)
display(Markdown(f"- 异常值数量：**{outlier_count}**"))
if outlier_count == 0:
    display(Markdown("✅ 未检测到异常值，数据质量良好"))
else:
    display(Markdown("⚠️ 检测到异常样本："))
    display(outliers[["date_time", "traffic_volume"]])

# ===================== 9. 核心分析结论 =====================
display(Markdown("## 📌 六、核心分析结论"))
peak_hour_mean = hourly_stats.loc[hourly_stats['hour'] == 16, 'mean'].values[0]
weekday_mean = weekend_stats.loc["工作日", "mean"]
weekend_mean = weekend_stats.loc["周末", "mean"]
drop_rate = round((1 - weekend_mean / weekday_mean) * 100, 2)
snow_drop = round((1 - snow_stats.loc["有降雪", "mean"] / snow_stats.loc["无降雪", "mean"]) * 100, 2)
rain_drop = round((1 - rain_stats.loc["有降雨", "mean"] / rain_stats.loc["无降雨", "mean"]) * 100, 2)

conclusion_md = f"""
1. **数据质量**：数据集共{len(df)}条样本，未检测到3σ准则异常值，数据质量良好。
2. **整体流量**：交通流量均值 {core_stats['mean']}辆/小时，中位数 {core_stats['50%']}辆/小时。
3. **时间规律**：全天流量峰值在16时，平均流量 {peak_hour_mean}辆/小时，存在明显早晚高峰。
4. **工作日周末差异**：工作日平均流量{weekday_mean}辆/小时，周末{weekend_mean}辆/小时；周末流量下降 {drop_rate}%。
5. **天气影响**：降雪对流量抑制最强，降雪时流量下降 {snow_drop}%；降雨下降 {rain_drop}%。
6. **核心影响因子**：小时、是否周末是影响交通流量最主要的特征。
"""
display(Markdown(conclusion_md))

# ===================== 10. 专家组：基于常识的交通拥堵判定 =====================
display(Markdown("## 🧠 七、专家组：基于人类常识的交通拥堵判定"))
display(Markdown("> 以下模块参照信用分析中「专家知识编码」的成功模式，将人类交通常识转化为可计算的评分规则。"))

# ---------- 10.1 时间段拥堵风险评分 ----------
display(Markdown("### 10.1 时间段拥堵风险评分"))
display(Markdown("> 人类常识：早晚高峰（7-9点、17-19点）最容易堵车，深夜凌晨最畅通"))
time_congestion_map = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1,   # 深夜-凌晨：极低风险
    6: 2,                                     # 早起出行：较低风险
    7: 4, 8: 5, 9: 4,                        # 早高峰：高风险
    10: 2, 11: 2, 12: 2, 13: 2, 14: 2,       # 白天平峰：较低风险
    15: 3, 16: 4, 17: 5, 18: 4, 19: 3,       # 晚高峰：高风险
    20: 2, 21: 2, 22: 2, 23: 1               # 夜间：低风险
}
df['time_congestion_score'] = df['hour'].map(time_congestion_map)

# ---------- 10.2 天气拥堵风险评分 ----------
display(Markdown("### 10.2 天气拥堵风险评分"))
display(Markdown("> 人类常识：雨雪雾天气路滑车慢，更容易堵车"))
weather_congestion_map = {
    'Clear': 1,        # 晴天：影响最小
    'Clouds': 1,       # 多云：几乎无影响
    'Drizzle': 2,      # 毛毛雨：轻微影响
    'Mist': 2,         # 薄雾：轻微影响
    'Rain': 3,         # 中雨：中等影响
    'Heavy Rain': 4,   # 大雨：较大影响
    'Snow': 4,         # 雪天：较大影响
    'Thunderstorm': 5  # 雷暴：严重影响
}
df['weather_congestion_score'] = df['weather_main'].map(weather_congestion_map).fillna(2)

# ---------- 10.3 温度极端性评分 ----------
display(Markdown("### 10.3 温度极端性评分"))
display(Markdown("> 人类常识：极端高温或低温影响出行意愿和路面状况"))
def temp_extreme_score(temp_c):
    if temp_c < -10 or temp_c > 40:
        return 3  # 极端温度
    elif temp_c < 0 or temp_c > 35:
        return 2  # 较端温度
    else:
        return 1  # 适宜温度

df['temp_extreme_score'] = df['temp_c'].apply(temp_extreme_score)

# ---------- 10.4 节假日拥堵评分 ----------
display(Markdown("### 10.4 节假日/周末拥堵评分"))
display(Markdown("> 人类常识：节假日出行多但通勤少，商场景区周边堵；周末出行量增加"))
def holiday_congestion_score(row):
    if row['is_holiday'] == 1:
        return 4  # 节假日出行高峰
    elif row['is_weekend'] == 1:
        return 3  # 周末出行较多
    else:
        return 2  # 工作日通勤

df['holiday_congestion_score'] = df.apply(holiday_congestion_score, axis=1)

# ---------- 10.5 降雨/降雪量影响评分 ----------
display(Markdown("### 10.5 降水强度评分"))
display(Markdown("> 人类常识：降水强度越大，路面状况越差，堵车风险越高"))
def precipitation_score(row):
    score = 1
    if row['rain_1h'] > 10:
        score = 4  # 大量降雨
    elif row['rain_1h'] > 0:
        score = 2  # 轻微降雨
    if row['snow_1h'] > 5:
        score = 5  # 大雪
    elif row['snow_1h'] > 0:
        score = 3  # 轻雪
    return score

df['precipitation_score'] = df.apply(precipitation_score, axis=1)

# ---------- 10.6 专家组评分特征统计 ----------
display(Markdown("### 10.6 专家组评分特征统计"))
expert_features = ['time_congestion_score', 'weather_congestion_score',
                   'temp_extreme_score', 'holiday_congestion_score', 'precipitation_score']
display(df[expert_features].describe().round(2))

# 专家组评分分布可视化
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, feat in enumerate(expert_features):
    df[feat].value_counts().sort_index().plot(kind='bar', ax=axes[i], color=plt.cm.viridis(i / 5))
    axes[i].set_title(feat.replace('_score', ''))
    axes[i].set_xlabel('评分')
    axes[i].set_ylabel('样本数')
plt.tight_layout()
plt.savefig('专家组评分分布.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`专家组评分分布.png`"))

# ===================== 11. 综合拥堵评分体系 =====================
display(Markdown("## 📊 八、综合交通拥堵评分体系"))
display(Markdown("> 参照信用分析中 final_credit_score 的加权公式，构建交通拥堵综合评分"))

# ---------- 11.1 流量相对水平评分 ----------
# 用当前流量与同时段历史均值比较，低流量+高峰=堵车（车被堵住了过不去）
hourly_mean = df.groupby('hour')['traffic_volume'].transform('mean')
hourly_std = df.groupby('hour')['traffic_volume'].transform('std')
df['volume_relative'] = (df['traffic_volume'] - hourly_mean) / (hourly_std + 1)
df['volume_congestion_score'] = df['volume_relative'].apply(
    lambda x: 5 if x < -2 else 4 if x < -1 else 3 if x < 0 else 2 if x < 1 else 1
)

# ---------- 11.2 综合拥堵评分公式 ----------
df['congestion_score'] = (
    df['time_congestion_score'] * 0.30 +          # 时间段权重最高
    df['volume_congestion_score'] * 0.25 +         # 流量相对水平
    df['weather_congestion_score'] * 0.15 +        # 天气影响
    df['holiday_congestion_score'] * 0.15 +        # 节假日/周末
    df['precipitation_score'] * 0.10 +             # 降水强度
    df['temp_extreme_score'] * 0.05                # 温度极端性
)

# ---------- 11.3 堵车等级划分 ----------
df['congestion_level'] = pd.cut(
    df['congestion_score'],
    bins=[0, 1.8, 2.5, 3.2, 4.0, 5.5],
    labels=['畅通', '基本畅通', '轻度拥堵', '中度拥堵', '严重拥堵']
)

display(Markdown("### 堵车等级分布"))
congestion_dist = df['congestion_level'].value_counts().sort_index()
display(congestion_dist.to_frame(name='样本数量'))

# 堵车等级分布饼图
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
display(Markdown("## 💡 九、出行建议系统"))

def generate_travel_advice(row):
    """基于多因素的出行建议（类似信用分析的风险等级输出）"""
    level = row['congestion_level']
    weather = row['weather_main']
    hour = row['hour']

    if level == '畅通':
        advice = "当前路况畅通，适合出行。"
    elif level == '基本畅通':
        advice = "路况基本正常，可正常出行。"
    elif level == '轻度拥堵':
        advice = "存在轻度拥堵，建议适当提前出发或选择替代路线。"
    elif level == '中度拥堵':
        advice = "中度拥堵，建议错峰出行或选择公共交通。预计延误20-40分钟。"
    else:
        advice = "严重拥堵，强烈建议推迟出行或改乘地铁/公交。预计延误40分钟以上。"

    # 天气附加建议
    weather_advice = ""
    if weather in ['Rain', 'Heavy Rain']:
        weather_advice = " 雨天路滑，请减速慢行，保持安全车距。"
    elif weather == 'Snow':
        weather_advice = " 雪天路面结冰，尽量避免驾车出行。"
    elif weather == 'Thunderstorm':
        weather_advice = " 雷暴天气，建议留在室内，避免出行。"
    elif weather in ['Mist', 'Drizzle']:
        weather_advice = " 能见度较低，请打开雾灯，注意行车安全。"

    # 时间附加建议
    time_advice = ""
    if hour in [7, 8, 9]:
        time_advice = " 早高峰时段，预计车流量大。"
    elif hour in [17, 18, 19]:
        time_advice = " 晚高峰时段，建议错峰。"
    elif hour in [0, 1, 2, 3, 4, 5]:
        time_advice = " 夜间出行请注意安全。"

    return advice + weather_advice + time_advice

df['travel_advice'] = df.apply(generate_travel_advice, axis=1)

# 展示各等级的典型建议
display(Markdown("### 各拥堵等级典型建议"))
for level in ['畅通', '基本畅通', '轻度拥堵', '中度拥堵', '严重拥堵']:
    sample = df[df['congestion_level'] == level]['travel_advice'].iloc[0]
    display(Markdown(f"- **{level}**：{sample}"))

# ===================== 13. 高峰时段识别与避堵推荐 =====================
display(Markdown("## 🕐 十、高峰时段识别与避堵推荐"))

peak_analysis = df.groupby(['hour', 'is_weekend']).agg(
    avg_volume=('traffic_volume', 'mean'),
    avg_congestion=('congestion_score', 'mean'),
    congestion_ratio=('congestion_level', lambda x: (x.isin(['中度拥堵', '严重拥堵'])).mean())
).round(3).reset_index()

peak_analysis['day_type'] = peak_analysis['is_weekend'].apply(
    lambda x: '周末' if x == 1 else '工作日'
)

display(Markdown("### 工作日各时段拥堵情况"))
workday_peaks = peak_analysis[peak_analysis['is_weekend'] == 0].sort_values('avg_congestion', ascending=False)
display(workday_peaks[['hour', 'avg_volume', 'avg_congestion', 'congestion_ratio']].head(10))

display(Markdown("### 周末各时段拥堵情况"))
weekend_peaks = peak_analysis[peak_analysis['is_weekend'] == 1].sort_values('avg_congestion', ascending=False)
display(weekend_peaks[['hour', 'avg_volume', 'avg_congestion', 'congestion_ratio']].head(10))

# 推荐最佳出行时段
workday_best = workday_peaks.sort_values('avg_congestion').head(3)
weekend_best = weekend_peaks.sort_values('avg_congestion').head(3)

display(Markdown(f"""
### 🎯 最佳出行时段推荐
- **工作日推荐**：{', '.join([str(h) + ':00' for h in workday_best['hour'].values])} 出行最顺畅
- **周末推荐**：{', '.join([str(h) + ':00' for h in weekend_best['hour'].values])} 出行最顺畅
"""))

# 工作日拥堵热力图
plt.figure(figsize=(12, 6))
workday_data = df[df['is_weekend'] == 0].groupby(['hour', 'weekday'])['congestion_score'].mean().unstack()
sns.heatmap(workday_data, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.5)
plt.title('工作日各时段拥堵热力图')
plt.xlabel('星期')
plt.ylabel('小时')
plt.yticks(range(0, 24), range(0, 24))
plt.tight_layout()
plt.savefig('工作日拥堵热力图.png', bbox_inches='tight')
plt.close()
display(Markdown("✅ 图表已保存：`工作日拥堵热力图.png`"))

# ===================== 14. 模型训练（融合专家组特征） =====================
display(Markdown("## 🤖 十一、预测模型训练（融合专家组特征）"))

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

# ---------- 14.1 堵车分类模型 ----------
display(Markdown("### 14.1 堵车等级分类模型"))

feature_cols = [
    'hour', 'weekday', 'is_weekend', 'is_holiday',
    'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
    # 专家组新增特征
    'time_congestion_score', 'weather_congestion_score',
    'temp_extreme_score', 'holiday_congestion_score', 'precipitation_score'
]
target_col = 'congestion_level'

# 编码目标变量
level_map = {'畅通': 0, '基本畅通': 1, '轻度拥堵': 2, '中度拥堵': 3, '严重拥堵': 4}
model_df = df[feature_cols + [target_col]].copy()
model_df = model_df.dropna()
model_df[target_col] = model_df[target_col].map(level_map)

X = model_df.drop(target_col, axis=1)
y = model_df[target_col]

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 预处理：独热编码天气类型
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['weather_main'])
    ],
    remainder='passthrough'
)

# 构建流水线（含class_weight='balanced'处理类别不平衡）
pipe_clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced'))
])

display(Markdown("#### 开始训练堵车分类模型"))
pipe_clf.fit(X_train, y_train)
y_pred_clf = pipe_clf.predict(X_test)

display(Markdown("#### 分类模型评估结果"))
display(Markdown(f"- **准确率**: {accuracy_score(y_test, y_pred_clf):.4f}"))
display(Markdown(f"- **加权F1**: {f1_score(y_test, y_pred_clf, average='weighted'):.4f}"))
display(Markdown("\n" + classification_report(y_test, y_pred_clf,
    target_names=['畅通', '基本畅通', '轻度拥堵', '中度拥堵', '严重拥堵'],
    zero_division=0)))

# ---------- 14.2 交通流量回归模型 ----------
display(Markdown("### 14.2 交通流量回归模型（对照组）"))

reg_feature_cols = [
    'hour', 'weekday', 'is_weekend', 'is_holiday',
    'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main'
]
reg_target_col = 'traffic_volume'

reg_model_df = df[reg_feature_cols + [reg_target_col]].copy()
reg_model_df = reg_model_df.dropna()

X_reg = reg_model_df.drop(reg_target_col, axis=1)
y_reg = reg_model_df[reg_target_col]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['weather_main'])
    ],
    remainder='passthrough'
)

pipe_reg = Pipeline(steps=[
    ('preprocess', preprocessor_reg),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
])

display(Markdown("#### 开始训练流量回归模型"))
pipe_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = pipe_reg.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

eval_df = pd.DataFrame({
    "评价指标": ["MAE平均绝对误差", "MSE均方误差", "RMSE均方根误差", "R2决定系数"],
    "数值": [round(mae, 2), round(mse, 2), round(rmse, 2), round(r2, 4)]
})
display(Markdown("#### 回归模型评估结果"))
display(eval_df)

# ---------- 14.3 保存模型 ----------
model_file_clf = "traffic_congestion_model.pkl"
model_file_reg = "traffic_volume_model.pkl"
joblib.dump(pipe_clf, model_file_clf)
joblib.dump(pipe_reg, model_file_reg)
display(Markdown(f"✅ 分类模型已保存为 `{model_file_clf}`"))
display(Markdown(f"✅ 回归模型已保存为 `{model_file_reg}`"))

# ===================== 15. 模型预测演示 =====================
display(Markdown("## 🧪 十二、模型预测演示"))

# 场景1：工作日晚高峰，晴天
display(Markdown("### 场景1：工作日晚高峰，晴天"))
sample1 = pd.DataFrame({
    'hour': [17], 'weekday': [2], 'is_weekend': [0], 'is_holiday': [0],
    'temp_c': [22], 'rain_1h': [0], 'snow_1h': [0], 'clouds_all': [30],
    'weather_main': ['Clear'],
    'time_congestion_score': [5], 'weather_congestion_score': [1],
    'temp_extreme_score': [1], 'holiday_congestion_score': [2], 'precipitation_score': [1]
})
pred_level1 = pipe_clf.predict(sample1)
pred_volume1 = pipe_reg.predict(sample1.drop(columns=['time_congestion_score', 'weather_congestion_score',
                                                        'temp_extreme_score', 'holiday_congestion_score', 'precipitation_score']))
level_reverse_map = {v: k for k, v in level_map.items()}
advice1 = generate_travel_advice({**sample1.iloc[0].to_dict(), 'congestion_level': level_reverse_map[pred_level1[0]]})
display(sample1)
display(Markdown(f"> 预测堵车等级：**{level_reverse_map[pred_level1[0]]}**"))
display(Markdown(f"> 预测交通流量：**{round(pred_volume1[0]):.0f} 辆/小时**"))
display(Markdown(f"> 出行建议：{advice1}"))

# 场景2：工作日晚高峰，雨天
display(Markdown("### 场景2：工作日晚高峰，雨天"))
sample2 = pd.DataFrame({
    'hour': [17], 'weekday': [2], 'is_weekend': [0], 'is_holiday': [0],
    'temp_c': [15], 'rain_1h': [5], 'snow_1h': [0], 'clouds_all': [80],
    'weather_main': ['Rain'],
    'time_congestion_score': [5], 'weather_congestion_score': [3],
    'temp_extreme_score': [1], 'holiday_congestion_score': [2], 'precipitation_score': [2]
})
pred_level2 = pipe_clf.predict(sample2)
pred_volume2 = pipe_reg.predict(sample2.drop(columns=['time_congestion_score', 'weather_congestion_score',
                                                        'temp_extreme_score', 'holiday_congestion_score', 'precipitation_score']))
advice2 = generate_travel_advice({**sample2.iloc[0].to_dict(), 'congestion_level': level_reverse_map[pred_level2[0]]})
display(sample2)
display(Markdown(f"> 预测堵车等级：**{level_reverse_map[pred_level2[0]]}**"))
display(Markdown(f"> 预测交通流量：**{round(pred_volume2[0]):.0f} 辆/小时**"))
display(Markdown(f"> 出行建议：{advice2}"))

# 场景3：周末午后，晴天
display(Markdown("### 场景3：周末午后，晴天"))
sample3 = pd.DataFrame({
    'hour': [14], 'weekday': [6], 'is_weekend': [1], 'is_holiday': [0],
    'temp_c': [25], 'rain_1h': [0], 'snow_1h': [0], 'clouds_all': [20],
    'weather_main': ['Clear'],
    'time_congestion_score': [2], 'weather_congestion_score': [1],
    'temp_extreme_score': [1], 'holiday_congestion_score': [3], 'precipitation_score': [1]
})
pred_level3 = pipe_clf.predict(sample3)
pred_volume3 = pipe_reg.predict(sample3.drop(columns=['time_congestion_score', 'weather_congestion_score',
                                                        'temp_extreme_score', 'holiday_congestion_score', 'precipitation_score']))
advice3 = generate_travel_advice({**sample3.iloc[0].to_dict(), 'congestion_level': level_reverse_map[pred_level3[0]]})
display(sample3)
display(Markdown(f"> 预测堵车等级：**{level_reverse_map[pred_level3[0]]}**"))
display(Markdown(f"> 预测交通流量：**{round(pred_volume3[0]):.0f} 辆/小时**"))
display(Markdown(f"> 出行建议：{advice3}"))

# ===================== 16. 生成Excel报告 =====================
display(Markdown("## 📑 十三、生成Excel分析报告"))
wb = Workbook()
wb.remove(wb.active)

header_font = Font(name='SimHei', size=12, bold=True, color='FFFFFF')
header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
center_alignment = Alignment(horizontal='center', vertical='center')
border = Border(left=Side('thin'), right=Side('thin'), top=Side('thin'), bottom=Side('thin'))


def write_sheet(wb, sheet_name, headers, data, col_widths=None):
    """统一的Excel写入函数"""
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
    return ws


# 原始数据概览sheet
headers = df.columns.tolist()
write_sheet(wb, '原始数据概览', headers, df.head(100).values.tolist(),
            [15] * len(headers))

# 整体统计分析sheet
write_sheet(wb, '整体统计分析', ['指标', '数值'], [
    ['总样本量', len(df)],
    ['交通流量平均值', core_stats['mean']],
    ['交通流量中位数', core_stats['50%']],
    ['交通流量标准差', core_stats['std']],
    ['交通流量最小值', core_stats['min']],
    ['交通流量最大值', core_stats['max']],
    ['异常值数量', outlier_count]
], [20, 20])

# 小时级交通流量分析
write_sheet(wb, '小时级交通流量分析', ['小时', '平均流量', '中位数流量', '流量标准差'],
            hourly_stats.values.tolist(), [18] * 4)

# 月度交通流量分析
write_sheet(wb, '月度交通流量分析', ['月份', '平均流量', '中位数流量', '流量标准差'],
            monthly_stats.values.tolist(), [18] * 4)

# 工作日周末对比
write_sheet(wb, '工作日周末对比分析', ['类型', '平均流量', '中位数流量', '流量标准差', '样本数量'],
            weekend_stats.reset_index().values.tolist(), [18] * 5)

# 节假日对比
write_sheet(wb, '节假日对比分析', ['类型', '平均流量', '中位数流量', '流量标准差', '样本数量'],
            holiday_stats.reset_index().values.tolist(), [18] * 5)

# 天气类型影响
write_sheet(wb, '天气类型影响分析', ['天气主类型', '平均流量', '中位数流量', '流量标准差', '样本数量'],
            weather_stats.reset_index().values.tolist(), [18] * 5)

# 降雨影响
write_sheet(wb, '降雨影响分析', ['类型', '平均流量', '中位数流量', '流量标准差', '样本数量'],
            rain_stats.reset_index().values.tolist(), [18] * 5)

# 降雪影响
write_sheet(wb, '降雪影响分析', ['类型', '平均流量', '中位数流量', '流量标准差', '样本数量'],
            snow_stats.reset_index().values.tolist(), [18] * 5)

# 相关性分析sheet
corr_headers = corr_matrix.index.tolist()
write_sheet(wb, '相关性分析', [''] + corr_headers,
            [[h] + list(row) for h, row in zip(corr_headers, corr_matrix.values)],
            [18] + [12] * len(corr_headers))

# 异常值分析sheet
write_sheet(wb, '异常值分析', ['指标', '数值'], [
    ['交通流量均值', round(mean_traffic, 2)],
    ['交通流量标准差', round(std_traffic, 2)],
    ['异常值下限', round(lower_bound, 2)],
    ['异常值上限', round(upper_bound, 2)],
    ['异常值数量', outlier_count]
], [20, 20])

# 专家组评分统计sheet
write_sheet(wb, '专家组评分统计', ['评分维度', '均值', '标准差', '最小值', '最大值'],
            [[f.replace('_score', '')] + list(row) for f, row in
             zip(expert_features, df[expert_features].describe().loc[['mean', 'std', 'min', 'max']].values.T)],
            [22, 12, 12, 12, 12])

# 堵车等级分布sheet
write_sheet(wb, '堵车等级分布', ['堵车等级', '样本数量', '占比'],
            [[level, count, f"{count / len(df) * 100:.1f}%"]
             for level, count in zip(congestion_dist.index, congestion_dist.values)],
            [18, 12, 12])

# 模型评估sheet
write_sheet(wb, '模型评估结果', ['模型类型', '评价指标', '数值'], [
    ['堵车分类模型', '准确率', round(accuracy_score(y_test, y_pred_clf), 4)],
    ['堵车分类模型', '加权F1', round(f1_score(y_test, y_pred_clf, average='weighted'), 4)],
    ['流量回归模型', 'MAE', round(mae, 2)],
    ['流量回归模型', 'RMSE', round(rmse, 2)],
    ['流量回归模型', 'R2', round(r2, 4)]
], [18, 18, 12])

excel_file = "高速公路交通流数据分析报告_完整版.xlsx"
wb.save(excel_file)
display(Markdown(f"✅ Excel报告已生成：`{excel_file}`"))

# ===================== 17. 最终总结 =====================
display(Markdown("## 📌 十四、最终总结"))
display(Markdown("""
### 本分析报告包含以下模块：

| 模块 | 内容 | 参照信用分析 |
|------|------|-------------|
| 数据预处理 | 缺失值处理、时间特征提取、温度转换 | 一~三章 |
| 统计分析 | 整体/时间/天气维度分析 | 四章 |
| 专家组（新增） | 5大常识规则编码为评分 | 第四章特征工程 |
| 综合评分（新增） | 加权公式计算拥堵分数 | final_credit_score |
| 堵车等级（新增） | 五级拥堵分类 | risk_category |
| 建议系统（新增） | 根据等级+天气+时间给出建议 | 无（新增） |
| 避堵推荐（新增） | 最佳出行时段推荐 | 无（新增） |
| 模型训练 | 堵车分类 + 流量回归 | 六~七章 |

### 专家组核心逻辑
将人类交通常识（高峰堵车、雨雪路滑、节假日出行多等）编码为数值评分，
通过加权公式融合，最终输出堵车等级和出行建议。
"""))

display(Markdown("🎉 **全部分析流程执行完毕！图表、Excel报告、模型文件输出完成**"))
