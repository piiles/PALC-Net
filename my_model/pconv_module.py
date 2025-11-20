import torch
import torch.nn as nn

class PartialConv1D(nn.Module):
    """1D版本的Partial Convolution，适用于时序数据"""

    def __init__(self, dim, n_div=4, forward='split_cat', kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dim_conv = dim // n_div  # 将被卷积处理的维度
        self.dim_untouched = dim - self.dim_conv  # 保持不变的维度
        
        self.partial_conv = nn.Conv1d(
            self.dim_conv, self.dim_conv,
            kernel_size, stride, padding,
            bias=False, groups=self.dim_conv  # 深度可分离卷积
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        # 训练和推理都适用
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        return x

    def forward_slicing(self, x):
        # 仅适用于推理
        x = x.clone()
        x[:, :self.dim_conv, :] = self.partial_conv(x[:, :self.dim_conv, :])
        return x

class PConv_CNN_LSTM(nn.Module):
    """使用PConv的CNN-LSTM混合模型"""

    def __init__(self, input_shape, cnn_filters=64, lstm_units=100, dense_units=50, n_div=4):
        super().__init__()
        n_timesteps, n_features = input_shape

        # 使用PConv替代传统Conv1D
        self.pconv1 = PartialConv1D(n_features, n_div=n_div, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(n_features)
        self.pconv2 = PartialConv1D(n_features, n_div=n_div, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_features)

        # 池化层
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)

        # LSTM层
        self.lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(lstm_units // 2, dense_units)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dense_units, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # PConv分支
        x = self.relu(self.bn1(self.pconv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.pconv2(x)))
        x = self.pool2(x)

        # LSTM分支
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 取最后一个时间步

        # 全连接
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x