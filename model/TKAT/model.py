# model.py
import keras
import tensorflow as tf
from keras.layers import InputLayer, Dense
from keras.models import Sequential

# Импортируйте ваш класс TKAN из соответствующего модуля
# Предполагается, что TKAN уже реализован в другом файле, например, в tkan.py
from tkan import TKAN  # Замените 'tkan' на имя файла, где реализован класс TKAN

def create_model(X_train_seq, y_train_seq,device='cpu'):
    """
    Создает и возвращает модель TKAN.

    :param X_train_seq: Входные данные для обучения
    :param y_train_seq: Целевые данные для обучения
    :return: Скомпилированная модель Keras
    """
    with tf.device(device):
        model = Sequential([
            InputLayer(input_shape=X_train_seq.shape[1:]),
            TKAN(100, sub_kan_configs=[
                {'spline_order': 3, 'grid_size': 10},
                {'spline_order': 1, 'grid_size': 5},
                {'spline_order': 4, 'grid_size': 6}
            ], return_sequences=True, use_bias=True),
            TKAN(100, sub_kan_configs=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True),
            TKAN(100, sub_kan_configs=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True),
            TKAN(100, sub_kan_configs=[None for _ in range(3)], return_sequences=False, use_bias=True),
            Dense(y_train_seq.shape[1]),  # Выходной слой
        ])
    return model