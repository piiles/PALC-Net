import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def build_cnn_lstm_model(input_shape):
    """构建CNN-LSTM混合模型"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(units=100, return_sequences=True),
        LSTM(units=50),
        Dense(units=50, activation='relu'),
        Dropout(0.3),
        Dense(units=1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_cnn_lstm(X_train_seq, y_train, X_val_seq, y_val, epochs=50, patience=10):
    """训练CNN-LSTM模型"""
    model = build_cnn_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    return model, history
