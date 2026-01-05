# -*- coding: utf-8 -*-
"""不使用日期的纯数值序列变点检测"""
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

# ===== 1. 读取数据 =====
df = pd.read_excel("data.xlsx")  # 假设只有Value列
ts = df['Value1'].ffill().values

# ===== 2. 变点检测 =====
"""
可选算法：
- rpt.Pelt: 高效变点检测（适合长序列）
- rpt.BinSeg: 二分法递归分割
- rpt.Dynp: 动态规划（精确但耗内存）
- rpt.Window: 滑动窗口法

可选模型：
- "l2"       : 检测均值变化
- "rbf"      : 检测非线性变化
- "l1"       : 检测稀疏变化
- "normal"   : 检测均值/方差变化
"ar"	自回归系数变化	时间序列预测模型稳定性
"rank"	数据分布秩变化	非参数化场景
"cosine"	特征向量方向变化	高维数据模式切换
"gamma"	Gamma分布参数变化	等待时间分析
"poisson"	Poisson分布强度变化	计数型数据分析
"clinear"	分段线性回归系数变化	趋势转折检测
"""
model = rpt.Dynp(model="clinear", min_size=25)
#model = rpt.Pelt(model="rbf", min_size=30)
model.fit(ts)
break_indices = model.predict(n_bkps=16)[:-1]  # 直接获取有效变点索引
#break_indices = model.predict(pen=10)[:-1]  # pen值越小越敏感
# ===== 3. 结果可视化 =====
plt.figure(figsize=(12, 6))
plt.plot(ts, label='data')

# 标注所有变点
for bp in break_indices:
    plt.axvline(x=bp, color='red', linestyle='--')
    plt.text(bp, ts[bp]*0.95, f'CP@{bp}',
            rotation=45, ha='right')

plt.title("Change Point Detection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

# 保存图形
plt.savefig('change_points_plot.png', bbox_inches='tight')  # 使用bbox_inches='tight'避免标签被截断
plt.show()

# ===== 4. 保存结果 =====
# 创建包含原始序列和变点位置的新DataFrame
output_df = pd.DataFrame({
    "Value": ts  # 原始时间序列列
})

# 添加变点位置列（非变点位置为NaN）
output_df["ChangePoint_Position"] = np.nan
output_df.loc[break_indices, "ChangePoint_Position"] = break_indices

# 保存到Excel
output_df.to_excel("change_points_marked_series.xlsx", index=False)