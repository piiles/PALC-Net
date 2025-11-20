import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Concatenate, Layer, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    Concatenate, Layer, LayerNormalization, BatchNormalization
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.base import BaseEstimator, RegressorMixin
import os

class PartialConv1D(Layer):
    """Keras实现的Partial Convolution 1D层"""
    def __init__(self, filters, kernel_size, n_div=4, activation='relu', **kwargs):
        super(PartialConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_div = n_div
        self.activation = activation
        
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.conv_channels = max(1, self.in_channels // self.n_div)  

        # 卷积部分：处理1/n_div的输入通道，输出完整的filters通道
        self.partial_conv = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=True,
            kernel_initializer='he_normal'
        )
        
        # 投影部分：将untouched通道投影到相同维度
        self.identity_proj = Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        
        # 批归一化
        self.bn = BatchNormalization()
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # 1. 分割输入通道
        conv_part = inputs[:, :, :self.conv_channels]
        identity_part = inputs[:, :, self.conv_channels:]
        
        # 2. 部分卷积（计算密集型操作）
        conv_output = self.partial_conv(conv_part)
        
        # 3. 轻量级投影（计算高效）
        identity_output = self.identity_proj(identity_part)
        
        # 4. 融合两部分（残差连接）
        output = conv_output + identity_output
        
        # 5. 批归一化
        output = self.bn(output, training=training)
        
        # 6. 激活函数
        if self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'tanh':
            output = tf.nn.tanh(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'n_div': self.n_div,
            'activation': self.activation
        })
        return config

class CrossAttentionLayer(Layer):
    """交叉注意力层"""
    def __init__(self, units, name=None, dropout_rate=0.3):
        super(CrossAttentionLayer, self).__init__(name=name)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Query-Key-Value投影
        self.W_query = Dense(units, use_bias=False, kernel_initializer='glorot_uniform')
        self.W_key = Dense(units, use_bias=False, kernel_initializer='glorot_uniform')
        self.W_value = Dense(units, use_bias=False, kernel_initializer='glorot_uniform')
        
        # 门控机制
        self.gate = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')
        
        # 正则化
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
    
    def call(self, temporal_features, static_features, training=None):
        # 1. 计算Query-Key-Value
        Q = self.W_query(temporal_features)
        K = self.W_key(static_features)
        V = self.W_value(static_features)
        
        # 2. 计算缩放点积注意力
        d_k = tf.cast(self.units, tf.float32)
        attention_scores = tf.reduce_sum(Q * K, axis=-1, keepdims=True) / tf.sqrt(d_k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 3. 加权Value
        context = attention_weights * V
        context = self.dropout(context, training=training)
        
        # 4. 门控融合
        gate_input = Concatenate()([temporal_features, context])
        gate_value = self.gate(gate_input)
        fused = gate_value * temporal_features + (1 - gate_value) * context
        
        # 5. 残差连接 + 层归一化
        output = self.layer_norm(temporal_features + fused)
        
        return output

class PConv_CNN_LSTM_Attention(BaseEstimator, RegressorMixin):
    """
    完全修复的PConv-CNN-LSTM-Attention混合模型
    """
    def __init__(self, 
                 sequence_length=50, 
                 cnn_filters=32,
                 lstm_units=64,
                 dense_units=32,
                 learning_rate=0.001,
                 epochs=80,
                 patience=15,
                 pconv_div=2):
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.pconv_div = pconv_div
        self.model = None
    
    def build_model(self, n_timesteps, n_sensor_features, n_stat_features):
        """构建改进的混合模型架构"""
        print(f"\n{'='*60}")
        print(f"构建PConv-CNN-LSTM-Attention模型")
        print(f"{'='*60}")
        print(f"时序输入: ({n_timesteps}, {n_sensor_features})")
        print(f"统计输入: ({n_stat_features},)")
        print(f"PConv分割比例: 1/{self.pconv_div}")
        
        # ===== 1. 时序特征分支 =====
        sequence_input = Input(shape=(n_timesteps, n_sensor_features), name='sequence_input')
        
        # 第一个PConv块
        x = PartialConv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            n_div=self.pconv_div,
            activation='relu',
            name='pconv1'
        )(sequence_input)
        x = MaxPooling1D(pool_size=2, name='maxpool1')(x)
        x = Dropout(0.2)(x)
        
        # 第二个PConv块
        x = PartialConv1D(
            filters=self.cnn_filters * 2,
            kernel_size=3,
            n_div=self.pconv_div,
            activation='relu',
            name='pconv2'
        )(x)
        x = MaxPooling1D(pool_size=2, name='maxpool2')(x)
        x = Dropout(0.2)(x)
        
        # LSTM时序建模
        x = LSTM(units=self.lstm_units, return_sequences=True, name='lstm1')(x)
        x = Dropout(0.2)(x)
        x = LSTM(units=self.lstm_units // 2, name='lstm2')(x)
        
        # 深度时序特征
        temporal_output = Dense(
            units=self.dense_units,
            activation='relu',
            name='deep_temporal_features'
        )(x)
        
        # ===== 2. 统计特征分支 =====
        stat_input = Input(shape=(n_stat_features,), name='statistical_input')
        
        static_features = Dense(
            units=self.dense_units,
            activation='relu',
            kernel_initializer='he_normal',
            name='deep_static_features'
        )(stat_input)
        static_features = Dropout(0.3)(static_features)
        
        # ===== 3. 改进的交叉注意力融合 =====
        cross_attention = CrossAttentionLayer(
            units=self.dense_units,
            dropout_rate=0.3,
            name='improved_cross_attention'
        )
        fused_features = cross_attention(temporal_output, static_features)
        
        # ===== 4. 预测头 =====
        x = Dense(units=self.dense_units * 2, activation='relu', name='merge_dense1')(fused_features)
        x = Dropout(0.3)(x)
        x = Dense(units=self.dense_units, activation='relu', name='merge_dense2')(x)
        x = Dropout(0.3)(x)
        final_output = Dense(units=1, activation='linear', name='rul_output')(x)
        
        # ===== 5. 构建模型 =====
        model = Model(
            inputs=[sequence_input, stat_input],
            outputs=final_output
        )
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(f"{'='*60}\n")
        
        return model
    
    def fit(self, X_seq, X_stat, y, groups=None, validation_data=None, verbose=1):
        """训练模型"""
        # 构建模型
        n_timesteps = X_seq.shape[1]
        n_sensor_features = X_seq.shape[2]
        n_stat_features = X_stat.shape[1] if len(X_stat.shape) > 1 else 1
        
        self.model = self.build_model(n_timesteps, n_sensor_features, n_stat_features)
        
        if verbose:
            print("模型架构：")
            self.model.summary()
        
        # 准备训练数据
        train_data = [X_seq, X_stat]
        
        # 准备回调函数
        callbacks = []
        
        # 1. 早停
        if validation_data is not None:
            val_X, val_y = validation_data
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                mode='min',
                verbose=1
            )
            callbacks.append(early_stop)
            
            # 2. 学习率衰减
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                mode='min',
                verbose=1
            )
            callbacks.append(lr_scheduler)
            
            # 3. 保存最佳模型
            os.makedirs('result', exist_ok=True) 
            checkpoint = ModelCheckpoint(
                'result/best_pconv_model.h5',  
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0
            )
            callbacks.append(checkpoint)
        
        # 训练模型
        if verbose:
            print(f"\n{'='*60}")
            print("开始训练...")
            print(f"{'='*60}\n")
        
        history = self.model.fit(
            train_data,
            y,
            validation_data=(val_X, val_y) if validation_data else None,
            epochs=self.epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("训练完成！")
            print(f"{'='*60}\n")
        
        return self
    
    def predict(self, X_seq, X_stat, groups=None):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练！")
        
        predictions = self.model.predict([X_seq, X_stat], verbose=0)
        return np.maximum(predictions.flatten(), 0)
    
    def score(self, X_seq, X_stat, y):
        """计算R²分数（sklearn接口）"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X_seq, X_stat)
        return r2_score(y, y_pred)