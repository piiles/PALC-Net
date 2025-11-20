import numpy as np
import warnings
import matplotlib.pyplot as plt
from data_loader import load_cmapss_data, split_train_val, data_quality_check
from feature_engineering import (
    select_features_multi_modal,
    build_advanced_features,
    build_sequence_features,
)
from my_model.pconv_cnn_lstm_attention import PConv_CNN_LSTM_Attention
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import os

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 12
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 18


def visualizations(
    model,
    X_test_seq,
    X_test_stat,
    y_test,
    test_engine_ids,
    test_cycles,
    test_df,
    subset="FD003",
    save_path="figure/",
):
    os.makedirs(save_path, exist_ok=True)
    # 1. 模型预测
    print("进行测试集预测...")
    y_pred = model.predict(X_test_seq, X_test_stat)

    # 创建结果DataFrame
    results_df = pd.DataFrame(
        {
            "engine_id": test_engine_ids[: len(y_pred)],
            "cycle": test_cycles[: len(y_pred)],
            "true_rul": y_test[: len(y_pred)],
            "predicted_rul": y_pred.flatten() if y_pred.ndim > 1 else y_pred,
        }
    )

    # 计算评估指标
    mae = mean_absolute_error(y_test[: len(y_pred)], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[: len(y_pred)], y_pred))

    print(f"测试集性能指标:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # 创建综合可视化图
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # 1. 预测vs真实值散点图 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        results_df["true_rul"],
        results_df["predicted_rul"],
        c=results_df["engine_id"],
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="white",
        linewidth=0.5,
    )

    # 添加理想线
    max_rul = max(results_df["true_rul"].max(), results_df["predicted_rul"].max())
    ax1.plot(
        [0, max_rul],
        [0, max_rul],
        "r--",
        linewidth=2,
        alpha=0.8,
        label="Ideal Prediction",
    )

    ax1.set_xlabel("True RUL (cycles)", fontweight="bold")
    ax1.set_ylabel("Predicted RUL (cycles)", fontweight="bold")
    ax1.set_title("Prediction vs True RUL", fontweight="bold", pad=20)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Engine ID", fontweight="bold")

    # 添加统计信息文本框
    textstr = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. 误差分布直方图 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    errors = results_df["predicted_rul"] - results_df["true_rul"]

    # 使用KDE绘制误差分布
    sns.histplot(
        errors,
        kde=True,
        ax=ax2,
        color="#2E86AB",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax2.axvline(
        x=errors.mean(),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean Error: {errors.mean():.2f}",
    )

    ax2.set_xlabel("Prediction Error (cycles)", fontweight="bold")
    ax2.set_ylabel("Density", fontweight="bold")
    ax2.set_title("Prediction Error Distribution", fontweight="bold", pad=20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 按发动机的预测轨迹 (左下)
    ax3 = fig.add_subplot(gs[1, :])

    # 选择几个代表性的发动机进行展示
    sample_engines = results_df["engine_id"].unique()[:6]
    colors = plt.cm.Set3(np.linspace(0, 1, len(sample_engines)))

    for i, engine_id in enumerate(sample_engines):
        engine_data = results_df[results_df["engine_id"] == engine_id]
        if len(engine_data) > 10:  # 只展示有足够数据点的发动机
            ax3.plot(
                engine_data["cycle"],
                engine_data["true_rul"],
                color=colors[i],
                linewidth=2.5,
                alpha=0.8,
                label=f"Engine {engine_id} (True)",
            )
            ax3.plot(
                engine_data["cycle"],
                engine_data["predicted_rul"],
                color=colors[i],
                linewidth=2.5,
                alpha=0.8,
                linestyle="--",
                label=f"Engine {engine_id} (Pred)",
            )

    ax3.set_xlabel("Cycle Number", fontweight="bold")
    ax3.set_ylabel("RUL (cycles)", fontweight="bold")
    ax3.set_title(
        "RUL Prediction Trajectories for Sample Engines", fontweight="bold", pad=20
    )
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # 添加整体标题
    fig.suptitle(
        f"PConv-CNN-LSTM-Attention Model Performance on {subset}\n"
        f"Comprehensive RUL Prediction Analysis",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(
        f"{save_path}/results_{subset}.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"结果可视化已保存至: {save_path}/results_{subset}.png")

    return results_df, fig


print("=== 航空发动机RUL预测 ===")

# 1. 数据加载与预处理
train_df, test_df = load_cmapss_data(subset="FD004", data_dir="CMAPSSData/")
train_data, val_data = split_train_val(train_df)
data_quality_check(train_data, "训练集")
data_quality_check(test_df, "测试集")

# 选择传感器
selected_sensors = select_features_multi_modal(train_data, top_k=15)
print(f"选中的传感器数量: {len(selected_sensors)}")
print(f"选中的传感器: {selected_sensors}")

# 2. 特征工程
# 2.1 传统统计特征
X_train, y_train, scaler, pca, feature_cols = build_advanced_features(
    df=train_data,
    selected_sensors=selected_sensors,
    window_sizes=[5, 10, 20],
    is_train=True,
)

X_val, y_val, _, _, _ = build_advanced_features(
    df=val_data,
    selected_sensors=selected_sensors,
    window_sizes=[5, 10, 20],
    is_train=False,
    scaler=scaler,
    pca=pca,
)

X_test, y_test, _, _, _ = build_advanced_features(
    df=test_df,
    selected_sensors=selected_sensors,
    window_sizes=[5, 10, 20],
    is_train=False,
    scaler=scaler,
    pca=pca,
)

print(f"传统特征训练集: {X_train.shape}, 目标值: {y_train.shape}")
print(f"特征列示例: {feature_cols[:10]}...")

# 2.2 时序特征
X_train_seq, y_train_seq, seq_scaler, train_engine_ids, train_cycles = (
    build_sequence_features(
        df=train_data, selected_sensors=selected_sensors, is_train=True
    )
)

X_val_seq, y_val_seq, _, val_engine_ids, val_cycles = build_sequence_features(
    df=val_data, selected_sensors=selected_sensors, is_train=False, scaler=seq_scaler
)

X_test_seq, y_test_seq, _, test_engine_ids, test_cycles = build_sequence_features(
    df=test_df, selected_sensors=selected_sensors, is_train=False, scaler=seq_scaler
)

print(f"时序特征训练集: {X_train_seq.shape}, 目标值: {y_train_seq.shape}")

# 3. 改进模型训练
# 数据对齐
min_samples = min(len(X_train_seq), len(X_train))
X_train_seq_aligned = X_train_seq[:min_samples]
X_train_stat_aligned = X_train[feature_cols].values[:min_samples]
y_train_aligned = y_train_seq[:min_samples]

val_min_samples = min(len(X_val_seq), len(X_val))
val_X_seq_aligned = X_val_seq[:val_min_samples]
val_X_stat_aligned = X_val[feature_cols].values[:val_min_samples]
val_y_aligned = y_val_seq[:val_min_samples]

print(f"时序输入维度: {X_train_seq_aligned.shape}")
print(f"统计输入维度: {X_train_stat_aligned.shape}")

# 训练我们的模型
our_model = PConv_CNN_LSTM_Attention(
    sequence_length=50,
    cnn_filters=32,
    lstm_units=64,
    dense_units=32,
    epochs=80,
    patience=15,
    learning_rate=0.001,  # 降低学习率
    pconv_div=2,  # 添加PConv参数
)

our_model.fit(
    X_seq=X_train_seq_aligned,
    X_stat=X_train_stat_aligned,
    y=y_train_aligned,
    validation_data=([val_X_seq_aligned, val_X_stat_aligned], val_y_aligned),
)

# 4. 改进模型评估
y_pred_our = our_model.predict(val_X_seq_aligned, val_X_stat_aligned)

print(f"\n改进模型验证集性能:")
print(f"MAE: {mean_absolute_error(val_y_aligned, y_pred_our):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(val_y_aligned, y_pred_our)):.2f}")
print(f"R2: {r2_score(val_y_aligned, y_pred_our):.4f}")

# 5. 测试集评估和可视化
test_min_samples = min(len(X_test_seq), len(X_test))
test_X_seq_aligned = X_test_seq[:test_min_samples]
test_X_stat_aligned = X_test[feature_cols].values[:test_min_samples]
test_y_aligned = y_test_seq[:test_min_samples]
test_engine_ids_aligned = test_engine_ids[:test_min_samples]
test_cycles_aligned = test_cycles[:test_min_samples]

print(f"测试集对齐后维度:")
print(f"时序特征: {test_X_seq_aligned.shape}")
print(f"统计特征: {test_X_stat_aligned.shape}")
print(f"目标值: {test_y_aligned.shape}")

# 可视化
results_df, visualization_fig = visualizations(
    model=our_model,
    X_test_seq=test_X_seq_aligned,
    X_test_stat=test_X_stat_aligned,
    y_test=test_y_aligned,
    test_engine_ids=test_engine_ids_aligned,
    test_cycles=test_cycles_aligned,
    test_df=test_df,
    subset="FD004",
    save_path="figure",
)
