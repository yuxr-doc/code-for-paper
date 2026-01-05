import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  StratifiedKFold,RepeatedStratifiedKFold,LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix,precision_recall_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
import optuna
from optuna import Study, Trial
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from pandas import json_normalize
import joblib
from pathlib import Path
from datetime import datetime
import platform
import sys
from typing import  Dict, Any
import json
from matplotlib.gridspec import GridSpec
from sklearn.utils import resample
from sklearn.base import clone
###################

#-----------------------------------
def process_scores(scores):
    df = json_normalize(
        scores,
        meta=[
            'fold',
            'accuracy',
            'f1',
            'recall',
            'precision',
            'roc_auc'
        ],
        sep='_'
    )
    column_mapping = {
        'best_params_learning_rate': 'Best_learning_rate',
        'best_params_max_depth': 'Best_max_depth',
        'best_params_min_child_weight': 'Best_min_child_weight',
        'best_params_subsample': 'Best_subsample',
        'best_params_colsample_bytree': 'Best_colsample_bytree',
        'best_params_n_estimators': 'Best_n_estimators',
        'best_params_reg_lambda': 'Best_reg_lambda',
        'best_params_alpha': 'Best_alpha',
        'best_params_gamma': 'Best_gamma'
    }
    return df.rename(columns=column_mapping)

#数据提取
df = pd.read_excel(r"C:\Users\Yxr\Desktop\PythonProject\source data.xlsx", header=None, sheet_name="Sheet7", nrows=275 + 1)# sheet_name="Sheet3", nrows=181 + 1)

header = df.iloc[0]
df = df[1:].astype('float32')
df.columns = header

feature_columns = [

'ANG_10_Mean',

]
feature_names = [

'RFH_Ang_Mean',

]
X = df[feature_columns]
tag = 'SCR Event3'
Y = df[tag].astype("int")
# 原始未处理数据（保持原始状态）
X_raw = df[feature_columns].values  # 转换为numpy数组
Y_raw = df[tag].astype("int").values
#X_raw, Y_raw = remove_outliers(X_raw, Y_raw)

###################
#数据提取 - 外部验证
df_ev = pd.read_excel(r"C:\Users\Yxr\Desktop\PythonProject\source data.xlsx", header=None, sheet_name="Sheet4", nrows=94 + 1)#

header_ev = df_ev.iloc[0]
df_ev = df_ev[1:].astype('float32')
df_ev.columns = header_ev
# 原始未处理数据（保持原始状态）
X_raw_ev = df_ev[feature_columns].values  # 转换为numpy数组
Y_raw_ev = df_ev[tag].astype("int").values
###################

#交叉验证折数
num_outcv = 5
num_inncv = 5
num_repeats = 5
outer_cv = RepeatedStratifiedKFold(n_splits=num_outcv, n_repeats=num_repeats, random_state=42)
# 创建 KFold 对象
inner_cv = StratifiedKFold(n_splits=num_inncv, shuffle=True, random_state=42)
# 初始化SHAP值收集容器
shap_records = {}  # 使用字典存储样本索引与SHAP值集合
#
final_scores = []
#all_test_indices = []
# 用于存储最优模型和性能
final_best_model = None
best_score = 0
# 在代码开头添加全局存储变量
# 绘制平均ROC曲线
mean_fpr = np.linspace(0, 1, 100)
tprs = []
auc_values = []
all_y_true = np.array([])
all_y_proba = np.array([])
#sns.set_style("whitegrid")
#fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
# 在外部交叉验证循环前初始化 阈值调整
optimal_thresholds = []
threshold_analysis_data = []
threshold = 0.5
# 在代码开头初始化
global_clean_indices = np.arange(len(X_raw))
global_negative_samples = np.array([], dtype=int)  # 全局负面样本记录
# 校准数据收集容器
calibrated_probs, uncalibrated_probs, true_labels = [], [], []
calibrated_y_true = np.array([])
calibrated_y_proba = np.array([])

def objective(trial, x_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    metrics = {
        'accuracy': [],
        'f1': [],
        'roc_auc': [],
        'sensitivity': [],
        'specificity': [],
        'ppv': [],
        'npv': []
    }

    for train_index, valid_index in inner_cv.split(x_train, y_train):
        x_train_infold, x_invalid_infold = x_train[train_index], x_train[valid_index]
        y_train_infold = y_train[train_index]  # 使用 NumPy 数组
        y_invalid_infold = y_train[valid_index]  # 使用 NumPy 数组

        # 标准化（仅在训练集拟合）
        scaler = StandardScaler()
        x_train_scaled_infold = scaler.fit_transform(x_train_infold)

        # 第三步：SMOTE过采样
        #smote = SMOTE(sampling_strategy='minority')
        #x_train_infold, y_train_infold = smote.fit_resample(x_train_scaled_infold, y_train_infold)
        x_train_infold, y_train_infold = (x_train_scaled_infold, y_train_infold)

        # 转换验证集数据（不可参与任何拟合）
        x_invalid_infold = scaler.transform(x_invalid_infold)  # 使用训练集的scaler

        model = GradientBoostingClassifier(**param,random_state=42
                                  )

        # 拟合模型
        model.fit(x_train_infold, y_train_infold
                  #, eval_set=[(x_invalid_infold, y_invalid_infold)],verbose=False
                  )
        # 验证模型
        y_proba_infold = model.predict_proba(x_invalid_infold)[:, 1]
        y_pred_infold = (y_proba_infold >= threshold).astype(int)

        # 计算验证集最佳阈值
        fpr, tpr, _ = roc_curve(y_invalid_infold, y_proba_infold)

        # 计算所有指标
        tn, fp, fn, tp = confusion_matrix(y_invalid_infold, y_pred_infold).ravel()

        def calculate_metrics(tn, fp, fn, tp):
            return {
                'accuracy': accuracy_score(y_invalid_infold, y_pred_infold),
                'f1': f1_score(y_invalid_infold, y_pred_infold),
                'roc_auc': roc_auc_score(y_invalid_infold, y_proba_infold),
                'sensitivity': tp / (tp + fn) if (tp + fn) != 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) != 0 else 0,
                'ppv': tp / (tp + fp) if (tp + fp) != 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) != 0 else 0
            }
        current_metrics = calculate_metrics(tn, fp, fn, tp)
        for key in current_metrics:  # 将各指标追加到对应列表
            metrics[key].append(current_metrics[key])

    for metric in metrics:
        mean_val = np.mean(metrics[metric])  # 保持原始比例（0-1范围）
        std_val = np.std(metrics[metric])
        trial.set_user_attr(f"{metric}_mean", mean_val)  # 存储浮点数
        trial.set_user_attr(f"{metric}_std", std_val)

    return np.mean(metrics['roc_auc'])


# 在外部交叉验证中评估模型
for fold, (out_train_index, out_valid_index) in enumerate(outer_cv.split(X_raw, Y_raw)):
    # ================== 数据预处理阶段 ==================
    # 拆分原始数据
    X_train_raw, X_val_raw = X_raw[out_train_index], X_raw[out_valid_index]
    y_train_raw, y_val = Y_raw[out_train_index], Y_raw[out_valid_index]

    study = optuna.create_study(direction='maximize')

    study.optimize(lambda trial: objective(trial, X_train_raw, y_train_raw), n_trials=100,
                   callbacks=[research_stop_callback], show_progress_bar=True)  # 进行 100 次试验

    best_params = study.best_params
    best_model = GradientBoostingClassifier(**best_params,random_state=42
                                   )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    ##################################################
    # 外部验证的数据标准化
    X_val_scaled_ev = scaler.transform(X_raw_ev)
    y_val_ev = Y_raw_ev
    ##################################################

    # 获取最佳试验的评估指标
    best_trial = study.best_trial
    print(f"inner的最佳指标：")
    print('Best hyperparameters: ', best_params)
    print(f"Accuracy: {best_trial.user_attrs['accuracy_mean']:.3f} ± {best_trial.user_attrs['accuracy_std']:.3f}")
    print(f"F1-score: {best_trial.user_attrs['f1_mean']:.3f} ± {best_trial.user_attrs['f1_std']:.3f}")
    print(f"ROC AUC: {best_trial.user_attrs['roc_auc_mean']:.3f} ± {best_trial.user_attrs['roc_auc_std']:.3f}")

    #smote = SMOTE(sampling_strategy='minority', k_neighbors= 5)
    #X_train_scaled, y_train_raw = smote.fit_resample(X_train_scaled, y_train_raw)

    best_model.fit(X_train_scaled, y_train_raw,
                  # eval_set=[(X_train_scaled, y_train_raw)],verbose=False
                   )
    y_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 存储全局数据
    all_y_true = np.concatenate([all_y_true, y_val])
    all_y_proba = np.concatenate([all_y_proba, y_proba])

    test_auc = roc_auc_score(y_val, y_proba)
    test_accuracy = accuracy_score(y_val, y_pred)
    test_f1 = f1_score(y_val, y_pred)
    test_recall = recall_score(y_val, y_pred)
    test_precision = precision_score(y_val, y_pred)

    test_brier_raw = brier_score_loss(y_val, y_proba)

    # 计算ROC数据
    fpr, tpr, thresholds  = roc_curve(y_val, y_proba)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    auc_values.append(auc(fpr, tpr))

    # 校准模型
    '''
    cal_probs = bootstrap_calibration(model=best_model,
                                       X_train=X_train_scaled,
                                       y_train=y_train_raw,
                                       X_test=X_val_scaled
                                       )'''
    calibrated = CalibratedClassifierCV(best_model, method='sigmoid',  cv=inner_cv)
    calibrated.fit(X_train_scaled, y_train_raw)
    cal_probs = calibrated.predict_proba(X_val_scaled)[:, 1]
    cal_pred = (cal_probs >= threshold).astype(int)

    brier_calibrated = brier_score_loss(y_val, cal_probs)

    # 存储结果
    calibrated_probs.extend(cal_probs)
    uncalibrated_probs.extend(y_proba)
    true_labels.extend(y_val)

    # 存储全局数据
    cal_auc = roc_auc_score(y_val, cal_probs)
    cal_accuracy = accuracy_score(y_val, cal_pred)
    cal_f1 = f1_score(y_val, cal_pred)
    cal_recall = recall_score(y_val, cal_pred)
    cal_precision = precision_score(y_val, cal_pred)

    ##################################################
    # 外部验证的性能指标
    ev_cal_probs = best_model.predict_proba(X_val_scaled_ev)[:, 1]
    ev_cal_pred = (ev_cal_probs >= threshold).astype(int)
    # 存储全局数据
    ev_auc = roc_auc_score(y_val_ev, ev_cal_probs)
    ev_accuracy = accuracy_score(y_val_ev, ev_cal_pred)
    ev_f1 = f1_score(y_val_ev, ev_cal_pred)
    ev_recall = recall_score(y_val_ev, ev_cal_pred)
    ev_precision = precision_score(y_val_ev, ev_cal_pred)
    print(f"外部验证 AUC: {ev_auc:.3f} | 外部验证 f1: {ev_f1:.3f}")
    ##################################################

    #shap值计算
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_val_scaled)  # 获取SHAP值矩阵
    for idx, sample_idx in enumerate(out_valid_index):
        if sample_idx not in shap_records:
            shap_records[sample_idx] = []
        # 添加tuple包含验证轮次和SHAP值
        shap_records[sample_idx].append((fold, shap_values[idx]))

    #评价指标
    final_scores.append({
        'fold': fold + 1,
        'test_best_params': study.best_params,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_roc_auc': test_auc,
        'test_brier_raw':test_brier_raw,
        #-----------
        'cal_accuracy': cal_accuracy,
        'cal_f1': cal_f1,
        'cal_recall': cal_recall,
        'cal_precision': cal_precision,
        'cal_roc_auc': cal_auc,
        'cal_brier':brier_calibrated,
        # -----------外部验证
        'ev_accuracy': ev_accuracy,
        'ev_f1': ev_f1,
        'ev_recall': ev_recall,
        'ev_precision': ev_precision,
        'ev_roc_auc': ev_auc
    })
    print(f"Fold {fold + 1} | 外循环的Best AUC: {test_auc:.3f} | 外循环的Best f1: {test_f1:.3f}")

# shap后处理：按样本索引排序并构建矩阵
sorted_indices = sorted(shap_records.keys())
shap_matrix = np.zeros((len(sorted_indices), shap_values.shape[1]))
for i, sample_idx in enumerate(sorted_indices):
    # 提取该样本所有fold的SHAP值
    fold_records = shap_records[sample_idx]
    # 均值模式
    all_shap = np.array([r[1] for r in fold_records])
    mean_shap = np.mean(all_shap, axis=0)
    # 策略3：中位数模式（对异常值鲁棒）
    #median_shap = np.median([r[1] for r in fold_records], axis=0)
    shap_matrix[i] = mean_shap

# 最终输出所有指标
print("\nFinal Model Metrics:")
for metric in ['accuracy','f1', 'recall', 'precision', 'roc_auc', 'brier_raw']:
    # 使用动态键名构建
    key_name = f'test_{metric}'
    values = [m[key_name] for m in final_scores]
    print(f"{metric.upper():<9}: {np.mean(values):.3f} ± {np.std(values):.3f}")
print("----------------")
for metric in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc', 'brier']:
    key_name = f'cal_{metric}'
    values = [m[key_name] for m in final_scores]
    print(f"{metric.upper():<9}: {np.mean(values):.3f} ± {np.std(values):.3f}")
print("----------------外部验证")
for metric in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']:
    key_name = f'ev_{metric}'
    values = [m[key_name] for m in final_scores]
    print(f"{metric.upper():<9}: {np.mean(values):.3f} ± {np.std(values):.3f}")
#保存外折X重复次数的模型性能至excel
# 转换数据
df = process_scores(final_scores)
# 按fold排序并设置索引
df = df.sort_values('fold').set_index('fold')
# 导出到Excel
excel_path = "gbdt_model_performance.xlsx"
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Performance Metrics')
    # 自动调整列宽（可选）
    worksheet = writer.sheets['Performance Metrics']
    for idx, col in enumerate(df.columns):
        max_len = max((
            df[col].astype(str).map(len).max(),  # 数据最大长度
            len(str(col))  # 列名长度
        )) + 2
        worksheet.set_column(idx, idx, max_len)
print(f"数据已成功导出至: {excel_path}")

shap.summary_plot(shap_matrix, X_raw, feature_names=feature_names, max_display=37)
shap.summary_plot(shap_matrix, X_raw, feature_names=feature_names, plot_type='bar', max_display=37)

# 4. 绘制平均ROC曲线
# 计算统计量
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_values)
std_tpr = np.std(tprs, axis=0)
# 5. 绘制最终图形
# 主绘图逻辑
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(mean_fpr, mean_tpr, color='#2b8cbe', lw=3,
        label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
# ax.fill_between(mean_fpr,np.maximum(mean_tpr - std_tpr, 0),np.minimum(mean_tpr + std_tpr, 1),color='#a6bddb', alpha=0.3, label='±1 SD')

# 参考线
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#636363',
        label='Random Chance')

# 样式优化
ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02],
       xlabel='False Positive Rate',
       ylabel='True Positive Rate',
       title='ROC Curve')
ax.grid(True, linestyle='--', alpha=0.6)

# 智能图例定位
ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
# 保存多种格式（满足期刊要求）
plt.savefig('ROC Curve.tif',  # TIFF用于投稿
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
plt.savefig('ROC Curve.pdf')  # PDF用于排版
plt.savefig('ROC Curve.svg')  # SVG用于后期编辑
plt.show()


#------------------------------------------
scaler_forshow = StandardScaler()
X_raw_forshow = scaler.fit_transform(X_raw)
shap_explanation = shap.Explanation(
values=shap_matrix, # SHAP值矩阵
data=X_raw_forshow, # 原始特征数据
feature_names=feature_names # 特征名称
)
# 配置全局绘图参数（符合学术出版标准）
plt.rcParams.update({
    'font.family': 'Arial',        # 优先使用期刊推荐字体
   # 'font.family': 'Times New Roman',
    'font.size': 12,               # 正文字号基准
    'axes.titlesize': 14,          # 标题字号
    'axes.labelsize': 12,          # 轴标签字号
    'xtick.labelsize': 10,         # X轴刻度字号
    'ytick.labelsize': 10,         # Y轴刻度字号
    'figure.dpi': 600,             # 出版级分辨率
    'savefig.bbox': 'tight',       # 自动裁剪白边
    'pdf.fonttype': 42,            # 确保PDF文字可编辑
    'svg.fonttype': 'none'         # 防止SVG文字转路径
})

# 创建高精度画布
fig = plt.figure(figsize=(18, 9),  # 黄金比例调整
                dpi=200,          # 出版级分辨率
                constrained_layout=True)

# 专业级网格布局配置
gs = GridSpec(1, 2, figure=fig,
            width_ratios=[3.5, 1],  # 优化视觉权重分配
            wspace=0.06,           # 精确控制子图间距
            left=0.08, right=0.92,
            top=0.92, bottom=0.1)  # 精确边距控制
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
max_display = 16
# ========================
# 左侧Beeswarm图优化
# ========================
# 专业配色方案（颜色无障碍兼容）
cmap = plt.get_cmap('coolwarm').copy()
cmap.set_bad(color='#808080')  # 设置缺失值颜色

shap.plots.beeswarm(
    shap_explanation,
    ax=ax1,
    show=False,
    color=cmap,               # 使用学术友好的配色
    max_display=max_display,           # 限制显示特征数
    plot_size=None
)

# 增强坐标轴可读性
ax1.set_xlabel("SHAP Value (impact on model output)",
              fontsize=12,
              labelpad=10)
ax1.set_title("A) Feature Importance Analysis",
             fontsize=14,
             pad=18,
             #weight='bold',
             loc='center')  # 符合学术标题格式

# 设置科学记数法坐标轴
ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
ax1.xaxis.get_offset_text().set_size(10)  # 调整科学计数法字号

# ========================
# 右侧Bar图优化
# ========================

shap.plots.bar(
    shap_explanation,
    ax=ax2,
    show=False,
    max_display=max_display,          # 控制显示条目数
)
#--------特征排序Y对齐并且右图颜色字体修改

for bar in ax2.patches:
    bar.set_facecolor('#6086d1')#改颜色

for text in ax2.texts:  # 注意倒序匹配
    text.set_fontsize(8)                     # 设置字体大小
    text.set_color('#333333')                 # 设置字体颜色

# 简化右轴显示
ax2.grid(False) #右图网格取消
ax2.spines[['right', 'top']].set_visible(False)
ax2.set_ylabel('')  # 清除冗余标签
ax2.set_xlabel("Mean |SHAP value|",
              fontsize=12,
              labelpad=10)
ax2.set_title("B) Global Feature Impact",
             fontsize=14,
             pad=18,
             #weight='bold',
             loc='center')
ax2.tick_params(
    axis='y',        # 选择Y轴
    labelleft=False  # 关闭左侧标签（适用于常规垂直坐标系）
)
fig.set_constrained_layout_pads(w_pad=0.08)
# 保存多种格式（满足期刊要求）
plt.savefig('SHAP_Analysis.tif',  # TIFF用于投稿
           dpi=600,
           pil_kwargs={"compression": "tiff_lzw"})
plt.savefig('SHAP_Analysis.pdf')  # PDF用于排版
plt.savefig('SHAP_Analysis.svg')  # SVG用于后期编辑
plt.show()

#校准图
# 转换为数组
true_labels = np.array(true_labels)
calibrated_probs = np.array(calibrated_probs)
uncalibrated_probs = np.array(uncalibrated_probs)
# 绘制校准曲线
def plot_calibration_curve(y_true, probs, label, color):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=5)#,strategy='quantile')
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label, color=color)
###########
plt.figure(figsize=(10, 8))
plot_calibration_curve(true_labels, uncalibrated_probs, "Uncalibrated", color='#6086d1')
plot_calibration_curve(true_labels, calibrated_probs, "Calibrated", color='#cb6451')
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability", fontsize=12)
plt.ylabel("Fraction of positives", fontsize=12)
plt.title(f"Calibration Comparison (Brier: Uncalibrated (blue) vs Calibrated (red))", fontsize=14)
plt.legend()
# 保存多种格式（满足期刊要求）
plt.savefig('Calibration_Analysis.tif',  # TIFF用于投稿
           dpi=600,
           pil_kwargs={"compression": "tiff_lzw"})
plt.savefig('Calibration_Analysis.pdf')  # PDF用于排版
plt.savefig('Calibration_Analysis.svg')  # SVG用于后期编辑
plt.show()
