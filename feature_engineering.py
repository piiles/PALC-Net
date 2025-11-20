import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def select_features_multi_modal(train_df, top_k=20):
    """多模态特征选择"""
    available_sensors = [col for col in train_df.columns if col.startswith("sensor_")]
    print(f"数据集中可用的传感器: {available_sensors}")

    if not available_sensors:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        available_sensors = [
            col
            for col in numeric_cols
            if col not in ["engine_id", "cycle", "rul", "max_cycle", "true_rul"]
        ]

    sensor_cols = available_sensors

    def fault_phase_score(df, sensors):
        """故障阶段评分函数"""
        if "max_cycle" not in df.columns:
            max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
            max_cycles.columns = ["engine_id", "max_cycle"]
            df = df.merge(max_cycles, on="engine_id")

        fault_phase = df[df["cycle"] >= df["max_cycle"] - 10].copy()
        normal_phase = df[df["cycle"] <= df["max_cycle"] * 0.3].copy()

        scores = {}
        for sensor in sensors:
            if sensor not in df.columns:
                continue
            try:
                fault_mean, fault_std = (
                    fault_phase[sensor].mean(),
                    fault_phase[sensor].std(),
                )
                normal_mean, normal_std = (
                    normal_phase[sensor].mean(),
                    normal_phase[sensor].std(),
                )

                def get_trend(phase_df, sensor_col):
                    phase_df = phase_df.sort_values("cycle")
                    if len(phase_df) < 5:
                        return 0
                    x = phase_df["cycle"].values.reshape(-1, 1)
                    y = phase_df[sensor_col].values
                    lr = LinearRegression().fit(x, y)
                    return lr.coef_[0]

                fault_trend = get_trend(fault_phase, sensor)
                normal_trend = get_trend(normal_phase, sensor)
                trend_diff = abs(fault_trend - normal_trend)
                change_ratio = abs(fault_mean - normal_mean) / (normal_mean + 1e-8)
                cv_ratio = (fault_std / (fault_mean + 1e-8)) / (
                    normal_std / (normal_mean + 1e-8) + 1e-8
                )
                scores[sensor] = change_ratio * cv_ratio * (1 + trend_diff)
            except Exception as e:
                scores[sensor] = 0
        return scores

    def correlation_score(df, sensors):
        """相关性评分函数"""
        scores = {}
        if "rul" not in df.columns or "engine_id" not in df.columns:
            return {sensor: 0 for sensor in sensors}

        for sensor in sensors:
            if sensor not in df.columns:
                continue
            try:
                if df[sensor].isna().all() or df[sensor].var() < 1e-8:
                    scores[sensor] = 0
                    continue

                final_vals = df.groupby("engine_id")[sensor].apply(
                    lambda x: x.iloc[-10:].mean() if len(x) >= 10 else x.iloc[-1]
                )
                final_rul = df.groupby("engine_id")["rul"].first()

                common_engines = final_vals.index.intersection(final_rul.index)
                if len(common_engines) < 2:
                    scores[sensor] = 0
                    continue

                final_vals = final_vals.loc[common_engines]
                final_rul = final_rul.loc[common_engines]

                if len(final_vals) > 1 and final_vals.var() > 1e-8:
                    corr, p_value = pearsonr(final_vals, final_rul)
                    scores[sensor] = abs(corr) * (1 - p_value)
                else:
                    scores[sensor] = 0
            except Exception as e:
                print(f"传感器 {sensor} 相关性计算失败: {str(e)}")
                scores[sensor] = 0

        if all(score == 0 for score in scores.values()):
            for sensor in sensors:
                if sensor in df.columns:
                    scores[sensor] = df[sensor].var()
        return scores

    # 领域知识传感器
    potential_domain_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ]
    domain_sensors = [
        sensor for sensor in potential_domain_sensors if sensor in sensor_cols
    ]

    if len(domain_sensors) < 5:
        variances = {sensor: train_df[sensor].var() for sensor in sensor_cols}
        high_var_sensors = sorted(variances.items(), key=lambda x: x[1], reverse=True)[
            :8
        ]
        domain_sensors.extend([s[0] for s in high_var_sensors])
        domain_sensors = list(set(domain_sensors))

    # 计算各种评分
    try:
        fault_scores = fault_phase_score(train_df, sensor_cols)
        corr_scores = correlation_score(train_df, sensor_cols)
    except Exception as e:
        print(f"特征评分计算失败: {e}")
        variances = {sensor: train_df[sensor].var() for sensor in sensor_cols}
        fault_scores = variances
        corr_scores = variances

    # 特征选择逻辑
    fault_selected = sorted(fault_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    fault_selected = [f[0] for f in fault_selected if f[1] > 0.01]

    corr_selected = sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    corr_selected = [f[0] for f in corr_selected if f[1] > 0.01]

    if not corr_selected and fault_selected:
        corr_selected = fault_selected[: min(5, len(fault_selected))]

    domain_selected = domain_sensors[:top_k]

    # 合并所有特征
    all_features = fault_selected + corr_selected + domain_selected
    feature_votes = pd.Series(all_features).value_counts()
    final_selected = feature_votes[feature_votes >= 1].index.tolist()

    if len(final_selected) < 8:
        print("选中的特征较少，补充高方差传感器...")
        variances = {sensor: train_df[sensor].var() for sensor in sensor_cols}
        high_var_sensors = sorted(variances.items(), key=lambda x: x[1], reverse=True)[
            :12
        ]
        high_var_sensors = [s[0] for s in high_var_sensors]
        final_selected.extend(high_var_sensors[: min(10, len(high_var_sensors))])
        final_selected = list(set(final_selected))

    print(f"多模态选择结果：")
    print(f"最终选中特征: {final_selected}，共{len(final_selected)}个传感器")

    return final_selected


def build_advanced_features(
    df, selected_sensors, window_sizes=[5, 10, 20], is_train=True, scaler=None, pca=None
):
    """特征工程"""
    df_features = df[["engine_id", "cycle", "rul"]].copy()

    # **关键改进**: 添加工况特征
    if 'op_condition' in df.columns:
        df_features['op_condition'] = df['op_condition']
        # One-hot编码工况
        for i in range(df['op_condition'].max() + 1):
            df_features[f'op_cond_{i}'] = (df['op_condition'] == i).astype(int)

    available_sensors = [sensor for sensor in selected_sensors if sensor in df.columns]
    print(
        f"构建特征: 请求{len(selected_sensors)}个传感器，实际可用{len(available_sensors)}个"
    )

    if len(available_sensors) < 5:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_sensors = [
            col
            for col in numeric_cols
            if col not in ["engine_id", "cycle", "rul", "max_cycle", "true_rul"]
        ]

    # 构建特征
    for sensor in available_sensors:
        df_features[f"{sensor}_raw"] = df[sensor]

        if 'op_condition' in df.columns:
            # 在每个工况内分别标准化
            for condition in df['op_condition'].unique():
                mask = df['op_condition'] == condition
                condition_mean = df.loc[mask, sensor].mean()
                condition_std = df.loc[mask, sensor].std() + 1e-8
                df_features.loc[mask, f'{sensor}_raw'] = \
                    (df.loc[mask, sensor] - condition_mean) / condition_std

        for window in window_sizes:
            group = df.groupby("engine_id")[sensor]
            # 均值特征
            mean_feat = (
                group.rolling(window=window, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
            df_features[f"{sensor}_mean_{window}"] = mean_feat
            # 标准差特征
            std_feat = (
                group.rolling(window=window, min_periods=1).std().reset_index(drop=True)
            )
            df_features[f"{sensor}_std_{window}"] = std_feat.fillna(0)
            # 趋势特征
            trend_feat = (
                group.rolling(window=window, min_periods=2)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                    raw=True,
                )
                .reset_index(drop=True)
            )
            df_features[f"{sensor}_trend_{window}"] = trend_feat
            # 偏度特征
            if window >= 3:
                skew_feat = (
                    group.rolling(window=window, min_periods=3)
                    .skew()
                    .reset_index(drop=True)
                )
                df_features[f"{sensor}_skew_{window}"] = skew_feat.fillna(0)

    # 传感器间交互特征
    for i in range(len(available_sensors)):
        for j in range(i + 1, min(i + 3, len(available_sensors))):
            s1, s2 = available_sensors[i], available_sensors[j]
            df_features[f"ratio_{s1}_{s2}"] = df[s1] / (df[s2] + 1e-8)
            df_features[f"diff_{s1}_{s2}"] = df[s1] - df[s2]

    # 操作条件与传感器交互
    op_cols = [f"op_{i}" for i in range(3)]
    for op in op_cols:
        if op in df.columns:
            for sensor in available_sensors[:6]:
                df_features[f"{op}_×_{sensor}"] = df[op] * df[sensor]

    feature_cols = [
        col for col in df_features.columns if col not in ["engine_id", "cycle", "rul"]
    ]

    # 处理无穷值和缺失值
    for col in feature_cols:
        max_val = df_features[col].replace([np.inf, -np.inf], np.nan).max()
        df_features[col] = df_features[col].replace(np.inf, max_val)
        df_features[col] = df_features[col].replace(
            -np.inf, -max_val if not pd.isna(max_val) else 0
        )
        df_features[col] = df_features.groupby("engine_id")[col].transform(
            lambda x: x.fillna(x.mean())
        )
        df_features[col] = df_features[col].fillna(df_features[col].mean())

    X = df_features[feature_cols].values
    y = df_features["rul"].values
    
    # 标准化
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("测试集/验证集必须传入训练集拟合的scaler对象！")
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_features.index)
    id_cols = df_features[["engine_id", "cycle", "rul"]].copy()
    X_scaled_df[["engine_id", "cycle", "rul"]] = id_cols

    # PCA降维
    from sklearn.decomposition import PCA

    if len(feature_cols) > 30:
        if is_train:
            n_components = min(25, len(feature_cols) - 5)
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled_df[feature_cols])
        else:
            if pca is None:
                raise ValueError(
                    "测试集/验证集特征数>30需PCA降维，请传入训练集拟合的pca对象！"
                )
            X_pca = pca.transform(X_scaled_df[feature_cols])

        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"pca_{i}" for i in range(X_pca.shape[1])],
            index=X_scaled_df.index,
        )
        pca_df[["engine_id", "cycle", "rul"]] = id_cols
        X_scaled_df = pca_df
        feature_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]

    # 过滤常数列
    var_cols = list(
        X_scaled_df[feature_cols].columns[X_scaled_df[feature_cols].var() > 1e-8]
    )
    all_cols = var_cols + ["engine_id", "cycle", "rul"]
    X_final = X_scaled_df[all_cols]

    assert not X_final[var_cols].isnull().any().any(), "处理后的特征仍包含缺失值！"
    assert not np.isinf(X_final[var_cols].values).any(), "处理后的特征仍包含无穷大值！"

    print(
        f"特征工程完成：原始{len(available_sensors)}个传感器 → 生成{len(feature_cols)}个特征 → 过滤后{len(var_cols)}个特征"
    )

    return X_final, y, scaler, pca, var_cols


def build_sequence_features(
    df, selected_sensors, sequence_length=30, is_train=True, scaler=None
):
    engine_ids = df["engine_id"].unique()
    X_seq_list, y_list, engine_id_list, cycle_list = [], [], [], []

    for eid in engine_ids:
        engine_data = (
            df[df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        )
        sensor_data = engine_data[selected_sensors].values
        rul_data = engine_data["rul"].values
        cycle_data = engine_data["cycle"].values

        if len(engine_data) < sequence_length:
            padding = np.zeros(
                (sequence_length - len(engine_data), len(selected_sensors))
            )
            sensor_data_padded = np.vstack([padding, sensor_data])
            rul_data_padded = np.hstack(
                [np.full(sequence_length - len(engine_data), rul_data[0]), rul_data]
            )
            cycle_padded = np.hstack(
                [np.full(sequence_length - len(engine_data), cycle_data[0]), cycle_data]
            )
        else:
            sensor_data_padded = sensor_data
            rul_data_padded = rul_data
            cycle_padded = cycle_data

        for i in range(len(sensor_data_padded) - sequence_length + 1):
            X_seq_list.append(sensor_data_padded[i : i + sequence_length])
            y_list.append(rul_data_padded[i + sequence_length - 1])
            engine_id_list.append(eid)
            cycle_list.append(cycle_padded[i + sequence_length - 1])

    X_seq = np.array(X_seq_list)
    y = np.array(y_list)

    if is_train:
        scaler = StandardScaler()
        original_shape = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_2d = scaler.fit_transform(X_seq_2d)
        X_seq = X_seq_2d.reshape(original_shape)
    else:
        if scaler is None:
            raise ValueError("测试集/验证集必须传入训练集拟合的scaler对象！")
        original_shape = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_2d = scaler.transform(X_seq_2d)
        X_seq = X_seq_2d.reshape(original_shape)

    return X_seq, y, scaler, engine_id_list, cycle_list


def align_modal_features(
    X_seq, engine_id_list, cycle_list, X_stat, stat_engine_id, stat_cycle
):
    seq_df = pd.DataFrame(
        {"engine_id": engine_id_list, "cycle": cycle_list, "seq_idx": range(len(X_seq))}
    )
    stat_df = X_stat[[stat_engine_id, stat_cycle]].copy()
    stat_df["stat_idx"] = stat_df.index

    aligned_df = pd.merge(
        seq_df,
        stat_df,
        left_on=["engine_id", "cycle"],
        right_on=[stat_engine_id, stat_cycle],
        how="inner",
    )

    X_seq_aligned = X_seq[aligned_df["seq_idx"].values]
    X_stat_aligned = X_stat.iloc[aligned_df["stat_idx"].values]
    y_aligned = X_stat_aligned["rul"].values
    groups_aligned = X_stat_aligned["engine_id"].values

    print(f"跨模态对齐完成：原始时序样本{len(X_seq)} → 对齐后{len(X_seq_aligned)}")

    return X_seq_aligned, X_stat_aligned, y_aligned, groups_aligned
