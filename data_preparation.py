from typing import Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class PreparationDataset(Dataset):
    """
    Класс для подготовки данных для обучения моделей.

    Параметры:
    - path (str, опционально): Путь к файлу с данными.
    - data (torch.Tensor, опционально): Тензор данных.
    - train_test_split (bool): Флаг для разделения данных на обучающую и тестовую выборки.

    Исключения:
    - ValueError: Если path и data оба равны None.
    - ValueError: Если возникает ошибка при загрузке файла.
    """

    def __init__(self, path: Optional[str] = None, data: Optional[torch.Tensor] = None, train_test_split: bool = False):
        self.all_data = torch.tensor([])
        if path is None and data is None:
            raise ValueError("Ошибка передачи ссылки или данных")
        elif path is not None:
            try:
                self.all_data = torch.load(path)
            except Exception as e:
                raise ValueError(f"Ошибка загрузки файла: {e}")
        else:
            self.all_data = data

        self.data = []  # Список для хранения признаков
        self.output = []  # Список для хранения целевых значений

    def __len__(self) -> int:
        """Возвращает количество элементов в данных."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает элемент данных по индексу."""
        x = self.data[idx]
        y = self.output[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def all_data_X_y(self) -> Tuple[list, list]:
        """Возвращает данные и целевые значения."""
        return self.data, self.output

    def PDtrain_test_split(
        self, X: torch.Tensor, y: torch.Tensor, test_size: float = 0.33, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Параметры:
        - X (torch.Tensor): Признаки.
        - y (torch.Tensor): Целевые значения.
        - test_size (float): Доля тестовой выборки. По умолчанию 0.33.
        - random_state (int): Seed для воспроизводимости. По умолчанию 42.

        Возвращает:
        - X_train, X_test, y_train, y_test (torch.Tensor): Разделенные данные.
        """
        bin = torch.arange(start=0, end=550, step=10, requires_grad=False) + torch.tensor([1000])
        bin_stratify = torch.bucketize(y.to('cpu'), bin)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=bin_stratify.numpy()
        )
        return X_train, X_test, y_train, y_test

    def cur_to_cur(
        self, feature_window_size: int, downsample_step: int, num_features: int, target_window_size: int, windows: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Подготовка данных для текущего окна.

        Параметры:
        - feature_window_size (int): Размер окна для признаков.
        - downsample_step (int): Шаг downsampling.
        - num_features (int): Число признаков в данных.
        - target_window_size (int): Размер окна для целевой переменной.
        - windows (int): Размер окна для обработки.
        - device (torch.device): Устройство для вывода данных (CPU или GPU).

        Возвращает:
        - X (torch.Tensor): Тензоры признаков.
        - y (torch.Tensor): Тензоры целевых значений.
        - original_data (pd.DataFrame): Оригинальные данные.

        Исключения:
        - ValueError: Если downsample_step < 1.
        - ValueError: Если размеры X и y не совпадают.
        """
        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")

        data_frame = pd.DataFrame(self.all_data)
        for ma in np.arange(0, 1.01, 0.05):
            for att in range(0, 1501, 100):
                filtered_data = data_frame[(data_frame[1] == att) & (data_frame[4] == ma)]
                filtered_data = filtered_data[::downsample_step]

                target_values = np.array(filtered_data.iloc[windows:, -1:]).reshape(-1)
                filtered_data.iloc[:, -1] = filtered_data.iloc[:, -1].shift(1)
                rolling_windows = filtered_data.rolling(window=windows)

                dataset = [window.values for window in rolling_windows if len(window) == windows][:-1]
                self.data.extend(dataset)
                self.output.extend(target_values)

        X = torch.tensor(self.data).float()
        y = torch.tensor(self.output).float()

        if len(X) != len(y):
            raise ValueError(f"Несоответствие размеров X ({len(X)}) и y ({len(y)})")

        X = X.to(device)
        y = y.to(device)
        return X, y, pd.DataFrame(self.all_data)

    def many_to_many(
        self, window_size: int, num_features: int = 2, downsample_step: int = 1, target_window_size: int = 1, device: torch.device = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Подготовка данных для модели many-to-many.

        Параметры:
        - window_size (int): Размер окна для обработки.
        - num_features (int): Число признаков в данных.
        - downsample_step (int): Шаг downsampling.
        - target_window_size (int): Размер окна для целевой переменной.
        - device (torch.device): Устройство для вывода данных (CPU или GPU).

        Возвращает:
        - X (torch.Tensor): Признаки.
        - y (torch.Tensor): Целевая переменная.
        - original_data (pd.DataFrame): Оригинальные данные.
        """
        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")

        reshaped_data = self.all_data[:, 1:].reshape(-1, 5334, num_features)
        if downsample_step > 1:
            reshaped_data = reshaped_data[:, int(5 / 0.003)::downsample_step, :]

        all_feature_windows = []
        all_target_values = []

        for i in range(reshaped_data.shape[0]):
            feature_values = reshaped_data[i].numpy()

            feature_windows = np.lib.stride_tricks.sliding_window_view(
                feature_values, window_shape=(window_size, feature_values.shape[1])
            )[: -target_window_size]

            target_values = np.lib.stride_tricks.sliding_window_view(
                feature_values[:, -1], window_shape=target_window_size
            )[window_size:]

            all_target_values.append(target_values)
            all_feature_windows.append(feature_windows)

        combined_feature_windows = np.vstack(all_feature_windows).squeeze(1)
        combined_target_values = np.vstack(all_target_values)

        if combined_feature_windows.size == 0:
            raise ValueError("После обработки данные пусты. Проверьте параметры window_size и downsample_step.")

        X = torch.tensor(combined_feature_windows).float()
        y = torch.tensor(combined_target_values).float()

        if len(X) != len(y):
            raise ValueError(f"Несоответствие размеров X ({len(X)}) и y ({len(y)})")

        X = X.to(device)
        y = y.to(device)
        return X, y, pd.DataFrame(self.all_data)

    def vec_to_vec(
        self, window_size: int, num_features: int = 2, downsample_step: int = 1, target_window_size: int = 1, device: torch.device = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Подготовка данных для модели vector-to-vector.

        Параметры:
        - window_size (int): Размер окна для обработки.
        - num_features (int): Число признаков в данных.
        - downsample_step (int): Шаг downsampling.
        - target_window_size (int): Размер окна для целевой переменной.
        - device (torch.device): Устройство для вывода данных (CPU или GPU).

        Возвращает:
        - X (torch.Tensor): Признаки.
        - y (torch.Tensor): Целевая переменная.
        - original_data (pd.DataFrame): Оригинальные данные.
        """
        if downsample_step < 1:
            raise ValueError("Параметр downsample_step должен быть >= 1")

        reshaped_data = self.all_data[:, 1:].reshape(-1, 5334, num_features)
        if downsample_step > 1:
            reshaped_data = reshaped_data[:, int(5 / 0.003)::downsample_step, :]

        all_feature_windows = []
        all_target_values = []

        for i in range(reshaped_data.shape[0]):
            feature_values = reshaped_data[i].numpy()

            feature_windows = np.lib.stride_tricks.sliding_window_view(
                feature_values, window_shape=(window_size, feature_values.shape[1])
            )[1: -target_window_size]

            target_values = np.lib.stride_tricks.sliding_window_view(
                feature_values[:, -1], window_shape=target_window_size
            )[window_size:-1]

            all_target_values.append(target_values)
            all_feature_windows.append(feature_windows)

        combined_feature_windows = np.vstack(all_feature_windows)
        combined_feature_windows = combined_feature_windows.reshape(combined_feature_windows.shape[0], -1)[:, :-1]

        combined_target_values = np.vstack(all_target_values)

        if combined_feature_windows.size == 0:
            raise ValueError("После обработки данные пусты. Проверьте параметры window_size и downsample_step.")

        X = torch.tensor(combined_feature_windows).float()
        y = torch.tensor(combined_target_values).float()

        if len(X) != len(y):
            raise ValueError(f"Несоответствие размеров X ({len(X)}) и y ({len(y)})")

        X = X.to(device)
        y = y.to(device)
        return X, y, pd.DataFrame(self.all_data)

def update_data_with_predictions(model: torch.nn.Module, df: pd.DataFrame, input_x: torch.Tensor) -> tuple:
    """
    Обновление данных с использованием предсказаний модели.
    idea:

    - input : [1 2 3 4]

    - shift input [1 2 3 4] -> predict[5] -> update [2 3 4 5]
    
    - output : update[2 3 4 5], predict[5]

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
    

    X = torch.rand((1000,10))*100
    y = torch.rand((1000,1))*100+ torch.rand((1000,1))*10

    PD = PreparationDataset(data=X)

    X_train, X_test, y_train, y_test = PD.PDtrain_test_split(X,y)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

