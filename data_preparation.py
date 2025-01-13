import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



class PreparationDataset(Dataset):

    def __init__(self, path: str = None,data: torch.tensor = None):
        
        # Загружаем все данные из файла
        self.all_data = torch.tensor([])
        if path is None :
            if data is not None:
                self.all_data = data
            else:
                raise ValueError(f"Ошибка передачи ссылки или данных")

        else:
            try:
                self.all_data = torch.load(path)
            except Exception as e:
                raise ValueError(f"Ошибка загрузки файла: {e}")

        self.data = []  # Список для хранения признаков
        self.output = []  # Список для хранения целевых значений

    def __len__(self):
        return len(self.data)

    def all_data_X_y(self):
        # Возвращаем данные и целевые значения
        return self.data, self.output

    def __getitem__(self, idx):
        # Получаем элемент по индексу
        x = self.data[idx]
        y = self.output[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    
    def cur_to_cur(self, feature_window_size, downsample_step, num_features, target_window_size, windows, device):
        """
        Подготовка данных для текущего окна.

        Параметры:
        - path (str): Путь к файлу с данными
        - feature_window_size (int): Размер окна для признаков
        - downsample_step (int): Шаг downsampling
        - num_features (int): Число признаков в данных
        - target_window_size (int): Размер окна для целевой переменной
        - windows (int): Размер окна для обработки
        - device (str): Устройство для вывода данных (CPU или GPU)
        """

        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")
        
        # Преобразуем данные в DataFrame для удобной обработки
        data_frame = pd.DataFrame(self.all_data)
        for ma in np.arange(0, 1.01, 0.05):
            for att in range(0, 1501, 100):
                filtered_data = data_frame[(data_frame[1] == att) & (data_frame[4] == ma)]
                filtered_data = filtered_data[::downsample_step]

                # Получаем целевые значения
                target_values = np.array(filtered_data.iloc[windows:, -1:]).reshape(-1)
                filtered_data.iloc[:, -1] = filtered_data.iloc[:, -1].shift(1)
                rolling_windows = filtered_data.rolling(window=windows)

                # Формируем набор данных
                dataset = [window.values for window in rolling_windows if len(window) == windows][:-1]
                self.data.extend(dataset)
                self.output.extend(target_values)

        # Преобразуем списки в тензоры
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.output = torch.tensor(self.output, dtype=torch.float32)

    def many_to_many(self, window_size: int, num_features: int = 2, downsample_step: int = 1, target_window_size: int = 1,device: torch.device = 'cpu'):
        """
        Подготовка данных для модели.

        Параметры:
        - window_size (int): Размер окна для обработки
        - num_features (int): Число признаков в данных
        - downsample_step (int): Шаг downsampling
        - target_window_size (int): Размер окна для целевой переменной
        - device (torch.device): Устройство для вывода данных (CPU или GPU)

        Возвращает:
        - X (torch.Tensor): Признаки
        - y (torch.Tensor): Целевая переменная
        - original_data (pd.DataFrame): Оригинальные данные
        """
        
        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")

        # Подготовка данных
        reshaped_data = self.all_data[:, 1:].reshape(-1, 5334, num_features)
        if downsample_step > 1:
            reshaped_data = reshaped_data[:, int(5/0.003)::downsample_step, :]

        # Оптимизированная обработка окон
        all_feature_windows = []
        all_target_values = []

        for i in range(reshaped_data.shape[0]):
            feature_values = reshaped_data[i].numpy()

            # X: окна для признаков
            feature_windows = np.lib.stride_tricks.sliding_window_view(
                feature_values,
                window_shape=(window_size, feature_values.shape[1]))[: -target_window_size]

            # y: окна для целевых значений со смещением
            target_values = np.lib.stride_tricks.sliding_window_view(
                feature_values[:, -1],
                window_shape=target_window_size)[window_size:]

            all_target_values.append(target_values)
            all_feature_windows.append(feature_windows)

        # Объединяем все окна
        combined_feature_windows = np.vstack(all_feature_windows).squeeze(1)

        combined_target_values = np.vstack(all_target_values)

        if combined_feature_windows.size == 0:
            raise ValueError("После обработки данные пусты. Проверьте параметры window_size и downsample_step.")

        # Формирование X и y
        X = torch.tensor(combined_feature_windows).float()
        y = torch.tensor(combined_target_values).float()

        # Проверка соответствия размеров X и y
        if len(X) != len(y):
            raise ValueError(f"Несоответствие размеров X ({len(X)}) и y ({len(y)})")

        X = X.to(device)
        y = y.to(device)
        return X, y, pd.DataFrame(self.all_data)

    def vec_to_vec(self, window_size: int, num_features: int = 2, downsample_step: int = 1, target_window_size: int = 1,device: torch.device = 'cpu'):
        """
        Подготовка данных для модели.

        Параметры:
        - window_size (int): Размер окна для обработки
        - num_features (int): Число признаков в данных
        - device (torch.device): Устройство для вывода данных (CPU или GPU)
        - downsample_step (int): Шаг downsampling
        - target_window_size (int): Размер окна для целевой переменной

        Возвращает:
        - X (torch.Tensor): Признаки
        - y (torch.Tensor): Целевая переменная
        - original_data (pd.DataFrame): Оригинальные данные
        """
        
        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")

        # Подготовка данных
        reshaped_data = self.all_data[:, 1:].reshape(-1, 5334, num_features)
        # убираем первые 5 секунд int(5/0.003)
        if downsample_step > 1:
            reshaped_data = reshaped_data[:, int(5/0.003)::downsample_step, :]

        # Оптимизированная обработка окон
        all_feature_windows = []
        all_target_values = []

        for i in range(reshaped_data.shape[0]):
            feature_values = reshaped_data[i].numpy()

            # X: окна для признаков
            feature_windows = np.lib.stride_tricks.sliding_window_view(
                feature_values,
                window_shape=(window_size, feature_values.shape[1]))[1: -target_window_size]

            # y: окна для целевых значений со смещением
            target_values = np.lib.stride_tricks.sliding_window_view(
                feature_values[:, -1],
                window_shape=target_window_size)[window_size:-1]

            all_target_values.append(target_values)
            all_feature_windows.append(feature_windows)

        # Объединяем все окна
        combined_feature_windows = np.vstack(all_feature_windows)
        combined_feature_windows = combined_feature_windows.reshape(combined_feature_windows.shape[0], -1)[:, :-1]

        combined_target_values = np.vstack(all_target_values)

        if combined_feature_windows.size == 0:
            raise ValueError("После обработки данные пусты. Проверьте параметры window_size и downsample_step.")

        # Формирование X и y
        X = torch.tensor(combined_feature_windows).float()
        y = torch.tensor(combined_target_values).float()

        # Проверка соответствия размеров X и y
        if len(X) != len(y):
            raise ValueError(f"Несоответствие размеров X ({len(X)}) и y ({len(y)})")

        X = X.to(device)
        y = y.to(device)
        return X, y, pd.DataFrame(self.all_data)
    


        

def update_data_with_predictions(model: torch.nn.Module, df: pd.DataFrame, input_x: torch.Tensor) -> tuple:
    """
    Обновление данных с использованием предсказаний модели.

    Параметры:
    - model (torch.nn.Module): Обученная модель для предсказания.
    - df (pd.DataFrame): Исходные данные, которые будут обновлены.
    - input_x (torch.Tensor): Входные данные для модели, содержащие признаки.

    Возвращает:
    - new_input (torch.Tensor): Обновленные входные данные, содержащие предыдущие значения, новые данные и предсказания.
    - predictions (torch.Tensor): Предсказания модели.
    """
    
    # Получение предсказаний от модели
    predictions = model(input_x)
    
    # Извлечение предыдущих значений из входных данных
    previous_values = input_x[:, 9:]
    
    # Преобразование DataFrame в тензор
    new_data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Объединение предыдущих значений, новых данных и предсказаний
    new_input = torch.cat((previous_values, new_data_tensor, predictions), dim=1)
    
    return new_input, predictions




if __name__=='__main__':
    feature_values = torch.randint(low=0, high=100, size=(20, 2))
    window_size = 3
    target_window_size = 2


    feature_windows = np.lib.stride_tricks.sliding_window_view(
                feature_values,
                window_shape=(window_size, feature_values.shape[1]))[: -target_window_size]

            # y: окна для целевых значений со смещением
    target_values = np.lib.stride_tricks.sliding_window_view(
                feature_values[:, -1],
                window_shape=target_window_size)[window_size:]
    
    print(feature_windows)
    print(target_values)