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
            TKAN(84, sub_kan_configs=[
                {'spline_order': 15, 'grid_size': 10},
            ], return_sequences=True, use_bias=True),
            TKAN(200, sub_kan_configs=[
                {'spline_order': 15, 'grid_size': 10},
            ], return_sequences=True, use_bias=True),
            TKAN(21, sub_kan_configs=[
                {'spline_order': 15, 'grid_size': 10},
            ], return_sequences=True, use_bias=True),
            # TKAN(128, sub_kan_configs=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True),
            # TKAN(128, sub_kan_configs=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True),
            #TKAN(128, sub_kan_configs=[None for _ in range(3)], return_sequences=False, use_bias=True),
            Dense(y_train_seq.shape[1]),  # Выходной слой
        ])
    return model



# from tensorflow.keras.layers import RNN, InputLayer, Dense
# from tensorflow.keras.models import Sequential

# class TKANCell(tf.keras.layers.AbstractRNNCell):
#     def __init__(self, units, sub_kan_configs, use_bias, **kwargs):
#         super().__init__(**kwargs)
#         self.units = units
#         self.tkan = TKAN(
#             units, 
#             sub_kan_configs=sub_kan_configs,
#             use_bias=use_bias,
#             return_sequences=False  # Для ячейки возвращаем только последний выход
#         )
        
#     @property
#     def state_size(self):
#         return self.units  # Размер состояния
    
#     @property
#     def output_size(self):
#         return self.units  # Размер выхода
    
#     def build(self, input_shape):
#         self.tkan.build(input_shape)
#         self.built = True
        
#     def call(self, inputs, states):
#         # inputs shape: (batch_size, features)
#         # states: список с предыдущим состоянием
        
#         # Вычисляем новый выход и состояние
#         output = self.tkan(tf.expand_dims(inputs, 1))  # Добавляем временную ось
#         output = tf.squeeze(output, axis=1)  # Убираем временную ось
        
#         # Обновляем состояние
#         new_state = output  # Простое обновление состояния
        
#         return output, [new_state]

# def create_model(X_train_seq, y_train_seq, device='cpu'):
#     """
#     Создает рекуррентную модель с TKAN
#     """
#     with tf.device(device):
#         model = Sequential([
#             InputLayer(input_shape=(None, X_train_seq.shape[-1])),  # (timesteps, features)
#             RNN(TKANCell(84, 
#                 sub_kan_configs=[{'spline_order': 15, 'grid_size': 10}],
#                 use_bias=True),
#                 return_sequences=True),
#             RNN(TKANCell(200,
#                 sub_kan_configs=[{'spline_order': 15, 'grid_size': 10}],
#                 use_bias=True),
#                 return_sequences=True),
#             RNN(TKANCell(21,
#                 sub_kan_configs=[{'spline_order': 15, 'grid_size': 10}],
#                 use_bias=True),
#                 return_sequences=False),
#             Dense(y_train_seq.shape[1])
#         ])
#     return model