import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

def load_cmapss_data(subset="FD001", data_dir="CMAPSSData/"):
    cols = ['engine_id', 'cycle'] + [f'op_{i}' for i in range(3)] + [f'sensor_{i}' for i in range(21)]
    
    train_path = f"{data_dir}/train_{subset}.txt"
    test_path = f"{data_dir}/test_{subset}.txt"
    rul_path = f"{data_dir}/RUL_{subset}.txt"
    
    train_df = pd.read_csv(train_path, sep='\\s+', header=None, names=cols)
    test_df = pd.read_csv(test_path, sep='\\s+', header=None, names=cols)
    true_rul = pd.read_csv(rul_path, sep='\\s+', header=None, names=['true_rul'])
    
    # 数据清洗和预处理
    for col in [f'sensor_{i}' for i in range(21)] + [f'op_{i}' for i in range(3)]:
        train_median = train_df[col].median()
        train_mean, train_std = train_df[col].mean(), train_df[col].std()
        
        # 训练集处理
        train_df[col] = train_df[col].fillna(train_median)
        train_df[col] = train_df[col].clip(train_mean - 3 * train_std, train_mean + 3 * train_std)
        
        # 测试集使用训练集的统计量
        test_df[col] = test_df[col].fillna(train_median)
        test_df[col] = test_df[col].clip(train_mean - 3 * train_std, train_mean + 3 * train_std)

    # 计算RUL
    def calculate_rul(train_df, test_df, true_rul):
        """改进的分段线性RUL计算"""
        train_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
        train_cycles.columns = ['engine_id', 'max_cycle']
        train_df = train_df.merge(train_cycles, on='engine_id')
        
        # 计算原始线性RUL
        train_df['rul'] = train_df['max_cycle'] - train_df['cycle']
        
        # 早期健康阶段设置为固定值（推荐125）
        # 原理：发动机早期性能稳定，RUL不应线性递减
        early_rul_threshold = 125
        train_df['rul'] = train_df['rul'].clip(upper=early_rul_threshold)
        
        # 测试集同样处理
        test_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
        test_cycles.columns = ['engine_id', 'max_cycle']
        test_cycles['true_rul'] = true_rul['true_rul'].values
        test_df = test_df.merge(test_cycles, on='engine_id')
        test_df['rul'] = test_df['true_rul'] + (test_df['max_cycle'] - test_df['cycle'])
        test_df['rul'] = test_df['rul'].clip(upper=early_rul_threshold)
        
        print(f"✓ RUL标注改进：使用分段线性策略（阈值={early_rul_threshold}）")
        
        return train_df, test_df

    train_df, test_df = calculate_rul(train_df, test_df, true_rul)
    print(f"加载{subset}完成：训练集{train_df.shape}，测试集{test_df.shape}")


    plot_rul_calculation_visualization(train_df, test_df, subset)
        
    return train_df, test_df

def split_train_val(train_df, test_size=0.3, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df['engine_id']))
    
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    
    return train_data, val_data

def data_quality_check(df, name):
    print(f"{name}数据质量报告：")
    print(f"- 形状：{df.shape}")
    print(f"- NaN总数：{df.isna().sum().sum()}")
    print(f"- Inf总数：{np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    
    const_cols = [col for col in df.select_dtypes(include=[np.number]).columns if df[col].var() < 1e-8]
    if const_cols:
        print(f"- 常数列（需移除）：{const_cols}")
    else:
        print("- 无空值/Inf/常数列，数据质量合格")
    
    return const_cols

def plot_rul_calculation_visualization(train_df, test_df, subset="FD001"):
    """绘制RUL计算过程可视化图"""
    plt.style.use('default')
    sns.set_palette("husl")
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'RUL Calculation Methodology Visualization - {subset}', 
                fontsize=16, fontweight='bold', y=0.98)
    sample_engines = train_df['engine_id'].unique()[:4]
    # 颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # 子图1: 训练集RUL计算过程
    for i, engine_id in enumerate(sample_engines):
        engine_data = train_df[train_df['engine_id'] == engine_id].sort_values('cycle')
        max_cycle = engine_data['max_cycle'].iloc[0]
        cycles = engine_data['cycle'].values
        rul = engine_data['rul'].values
        # 绘制RUL曲线
        ax1.plot(cycles, rul, linewidth=2.5, color=colors[i], 
                label=f'Engine {engine_id}', alpha=0.8)
        # 标记最大周期点
        ax1.scatter(max_cycle, 0, color=colors[i], s=80, zorder=5, 
                   edgecolors='white', linewidth=2)
    # 添加RUL截断线
    ax1.axhline(y=125, color='red', linestyle='--', linewidth=2, 
               label='RUL Truncation (125 cycles)', alpha=0.7)
    ax1.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Training Set: RUL = Max Cycle - Current Cycle', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # 子图2: 测试集RUL计算过程
    test_sample_engines = test_df['engine_id'].unique()[:4]
    for i, engine_id in enumerate(test_sample_engines):
        engine_data = test_df[test_df['engine_id'] == engine_id].sort_values('cycle')
        true_rul = engine_data['true_rul'].iloc[0]
        max_cycle = engine_data['max_cycle'].iloc[0]
        cycles = engine_data['cycle'].values
        rul = engine_data['rul'].values
        # 绘制RUL曲线
        ax2.plot(cycles, rul, linewidth=2.5, color=colors[i], 
                label=f'Engine {engine_id}', alpha=0.8)
        # 标记真实RUL起始点
        ax2.scatter(max_cycle, true_rul, color=colors[i], s=80, zorder=5, 
                   marker='s', edgecolors='white', linewidth=2)
        # 添加标注
        ax2.annotate(f'True RUL: {true_rul}', 
                    xy=(max_cycle, true_rul), 
                    xytext=(max_cycle+5, true_rul+10),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5),
                    fontsize=9, fontweight='bold', color=colors[i])
        
    # 添加RUL截断线
    ax2.axhline(y=125, color='red', linestyle='--', linewidth=2, 
               label='RUL Truncation (125 cycles)', alpha=0.7)
    ax2.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Test Set: RUL = True RUL + (Max Cycle - Current Cycle)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # 保存高质量图片
    os.makedirs('RUL', exist_ok=True)  
    plt.savefig(f"RUL/RUL_Calculation_Methodology_{subset}.png",
               dpi=400, bbox_inches="tight", facecolor='white')
    print(f"RUL计算可视化图已保存: RUL/RUL_Calculation_Methodology_{subset}.png")
    return fig
