from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class LossVisualizer:
    """
    Класс для визуализации потерь, метрик и весов модели.

    Параметры:
    - losses (Dict[str, list]): Словарь, содержащий метрики потерь и оценки.
    - data (Optional[any]): Дополнительные данные для визуализации (по умолчанию None).

    Пример:
        losses = {
            'train_total_loss': [],
            'train_main_loss': [],
            'train_quantum_loss': [],
            'train_mape': [],
            'train_alpha': [],
            'test_mape': [],
            'test_tube': []
        }
    """

    def __init__(self, losses: Dict[str, list], data: Optional[any] = None):
        self.losses = losses
        self.data = data
        self.model = None

    def show_training_loss(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 60) -> None:
        """
        Отображает график основной функции потерь во время обучения.

        Параметры:
        - figsize (Tuple[int, int]): Размер графика. По умолчанию (12, 6).
        - dpi (int): Разрешение графика. По умолчанию 60.
        """
        if 'train_main_loss' not in self.losses:
            raise KeyError("Ключ 'train_main_loss' отсутствует в losses.")

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.losses['train_main_loss'], label='Основная потеря', color='blue', alpha=0.7)
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Значение функции потерь')
        plt.title('График основной функции потерь')
        plt.legend()
        plt.show()

    def show_test_tube(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 60) -> None:
        """
        Отображает график отношения количества попаданий данных в 5% барьер на тестовых данных.

        Параметры:
        - figsize (Tuple[int, int]): Размер графика. По умолчанию (12, 6).
        - dpi (int): Разрешение графика. По умолчанию 60.
        """
        if 'test_tube' not in self.losses:
            raise KeyError("Ключ 'test_tube' отсутствует в losses.")

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.losses['test_tube'], label='Отношение попаданий', color='green', alpha=0.7)
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Отношение попаданий в 5% барьер')
        plt.title('График отношения попаданий на тестовых данных')
        plt.legend()
        plt.show()

    def show_mape(
        self,
        show_all: bool = True,
        show_chip: bool = True,
        start_epoch: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 60
    ) -> None:
        """
        Отображает графики MAPE для обучающей и тестовой выборок.

        Параметры:
        - show_all (bool): Отображать ли все эпохи. По умолчанию True.
        - show_chip (bool): Отображать ли первые 150 эпох. По умолчанию True.
        - start_epoch (Optional[int]): Начальная эпоха для отображения. По умолчанию None.
        - figsize (Tuple[int, int]): Размер графика. По умолчанию (12, 6).
        - dpi (int): Разрешение графика. По умолчанию 60.
        """
        if 'train_mape' not in self.losses or 'test_mape' not in self.losses:
            raise KeyError("Ключи 'train_mape' или 'test_mape' отсутствуют в losses.")

        train_mape = self.losses['train_mape']
        test_mape = self.losses['test_mape']

        if show_all:
            self._plot_mape(train_mape, test_mape, "График MAPE для обучающей и тестовой выборок", figsize, dpi)

        if show_chip:
            self._plot_mape(train_mape[:150], test_mape[:150], "График MAPE для первых 150 эпох", figsize, dpi)

        if start_epoch is not None:
            self._plot_mape(
                train_mape[start_epoch:],
                test_mape[start_epoch:],
                f'График MAPE, начиная с эпохи {start_epoch}',
                figsize,
                dpi
            )

    def _plot_mape(
        self,
        train_mape: list,
        test_mape: list,
        title: str,
        figsize: Tuple[int, int],
        dpi: int
    ) -> None:
        """Вспомогательный метод для построения графиков MAPE."""
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(train_mape, label='Ошибка на обучающей выборке', alpha=0.7)
        plt.plot(test_mape, label='Ошибка на тестовой выборке', color='red')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Значение MAPE')
        plt.title(title)
        plt.legend()
        plt.show()

    def histogram_mape(
        self,
        model: nn.Module,
        X: torch.Tensor,
        target: torch.Tensor,
        limit_percel: Optional[float] = None,
        keras: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 80
    ) -> None:
        """
        Отображает гистограмму распределения MAPE на тестовых данных.

        Параметры:
        - model (nn.Module): Модель для предсказаний.
        - X (torch.Tensor): Входные данные.
        - target (torch.Tensor): Целевые значения.
        - limit_percel (Optional[float]): Порог для фильтрации значений MAPE. По умолчанию None.
        - keras (bool): Использовать ли модель Keras. По умолчанию False.
        - figsize (Tuple[int, int]): Размер графика. По умолчанию (12, 6).
        - dpi (int): Разрешение графика. По умолчанию 80.
        """
        if keras:
            pred = model.predict(X)
        else:
            model.eval()
            pred = model.forward(X).to('cpu')
            model.train()

        target = target.to('cpu')
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        per_loss_rd = (mape < 0.01 * limit_percel).sum().item() / (mape.numel()) if limit_percel else 1.0

        mape = mape.view(-1)
        low_mape_values = mape[mape < 0.01 * limit_percel] if limit_percel else mape
        lete = f'< {0.01 * limit_percel}' if limit_percel else 'всех значения'

        plt.figure(figsize=figsize, dpi=dpi)
        plt.hist(low_mape_values.detach().numpy(), bins=100, edgecolor='black')
        plt.xlabel('Значения MAPE')
        plt.ylabel('Количество примеров')
        plt.title(f'Распределение метрики MAPE на тестовых данных {lete}: {per_loss_rd:.4}')
        plt.show()

        print(f'Процент значений MAPE {lete}: {per_loss_rd:.4}')

        
    def show_predictions_rnn(self,
                model: nn.Module, 
                df: pd.DataFrame,
                pd_params: tuple,
                keras: bool = False,
                device:torch.device | str = 'cpu') -> None:
        """
        Отображает графики предсказаний модели по сравнению с истинными значениями.

        Параметры:
        - df (pd.DataFrame): Данные для визуализации.
        - model (nn.Module): Модель для предсказаний.
        - pd_params (tuple): Параметры обработки данных # window_size num_features downsample_step target_window_size
        - device (str): Устройство для выполнения (CPU или GPU).

        Возвращает:
        - dict: Словарь с истинными и предсказанными значениями.
        """
        from data_preparation import PreparationDataset
        #results = {'true_values': [], 'predictions': [], 'x_values': [], 'y_values': []}
        for max_value in np.arange(0, 1.01, 0.25):  
            for attribute in range(0, 1501, 500):  
                filtered_df = df[(df[2] == attribute) & (df[5] == max_value)]
                print('Количество записей:', len(filtered_df))
                if filtered_df.empty:
                    continue
                    
                PD = PreparationDataset(data=torch.tensor(filtered_df.values))  # Убедитесь, что данные преобразованы в тензор

                window_size ,num_features,downsample_step,target_window_size =  pd_params
                # 0.003 * downsample_step = шаг данных в секундах
                  # количество выходных данных для 1 примера
                    
                    # Подготовка данных
                X, y, _ = PD.many_to_many(
                        window_size ,
                        num_features,
                        downsample_step,# 0.003 * downsample_step = шаг данных в секундах
                        target_window_size,
                        device=device  
                    )
                if keras:
                    predictions = model.predict(X)[:, -1].to('cpu').detach().numpy()
                else:
                    model.eval()
                    predictions = model.forward(X)[:, -1].to('cpu').detach().numpy()
                    model.train()
                    
                true_values = X[:,-1,0].to('cpu').detach().numpy() # Преобразуем в numpy сразу
                    
                    # Построение графика
                    # results['input'].append(true_values)
                    # results['predictions'].append(predictions)
                    # results['target'].append(y)  # Обрезаем до той же длины
                
                plt.figure(figsize=(12, 6), dpi=60)
                plt.scatter(true_values,predictions, s=1, label='Предсказания модели') 
                plt.scatter(true_values,y[:,-1].to('cpu').detach().numpy(), color='red', s=0.15, label='Истинные значения') 
                plt.xlabel('Топливо')
                plt.ylabel('Тяга')
                plt.title(f'Max: {max_value}, Attribute: {attribute}\nКрасное: истинные значения\nСинее: предсказанные значения')
                plt.legend()
                plt.show()
        
        #return results
    def show_predictions(self,
                model: nn.Module, 
                df: pd.DataFrame, 
                pd_params: tuple,
                keras: bool = False,
                device:torch.device | str = 'cpu') -> None:
        from data_preparation import PreparationDataset
        #results = {'true_values': [], 'predictions': [], 'x_values': [], 'y_values': []}
        for max_value in np.arange(0, 1.01, 0.5):  
            for attribute in range(0, 1501, 700):  
                filtered_df = df[(df[2] == attribute) & (df[5] == max_value)]
                print('Количество записей:', len(filtered_df))
                if filtered_df.empty:
                    continue
                    
                PD = PreparationDataset(data=torch.tensor(filtered_df.values))  # Убедитесь, что данные преобразованы в тензор

                window_size,num_features,downsample_step,target_window_size = pd_params  # количество выходных данных для 1 примера
                    
                    # Подготовка данных
                X, y, _ = PD.vec_to_vec(
                        window_size=window_size,
                        num_features=num_features,
                        downsample_step=downsample_step,
                        target_window_size=target_window_size,
                        device=device  
                    )
                if keras:
                    predictions = model.predict(X).to('cpu').detach().numpy()
                else:
                    model.eval()
                    predictions = model.forward(X).to('cpu').detach().numpy()
                    model.train()
                    
                true_values = X[:,-9].to('cpu').detach().numpy() 
                    
                plt.figure(figsize=(12, 6), dpi=60)
                plt.scatter(true_values,predictions, s=3, label='Предсказания модели') 
                plt.scatter(true_values,y[:,-1].to('cpu').detach().numpy(), color='red', s=6, label='Истинные значения') 
                plt.xlabel('Топливо')
                plt.ylabel('Тяга')
                plt.title(f'Max: {max_value}, Attribute: {attribute}\nКрасное: истинные значения\nСинее: предсказанные значения')
                plt.legend()
                plt.show()


    def show_weight(self, model: nn.Module, figsize: tuple = (8, 6), dpi: int = 60, cmap: str = 'viridis') -> None:
        """
        Визуализирует все параметры модели (веса).

        Аргументы:
            model (nn.Module): Модель, веса которой нужно визуализировать.
            figsize (tuple, опционально): Размер фигур для визуализации. По умолчанию (8, 6).
            dpi (int, опционально): Разрешение фигур. По умолчанию 60.
            cmap (str, опционально): Цветовая карта для визуализации. По умолчанию 'viridis'.

        Возвращает:
            None
        """
        for name, param in model.named_parameters():
            if 'weight' in name:  # Извлекаем только веса (игнорируем смещения)
                print(f"Слой: {name}, Размер: {param.shape}")
                weights = param.to('cpu').detach().numpy()  # Преобразуем в numpy для визуализации

                if len(weights.shape) == 2:  # Для полносвязных слоев
                    self._visualize_fc_layer(weights, name, figsize, dpi, cmap)
                elif len(weights.shape) == 4:  # Для сверточных слоев
                    self._visualize_conv_layer(weights, name, figsize, dpi, cmap)
                else:
                    print(f"Визуализация для слоя {name} с размерностью {weights.shape} не поддерживается.")


    def _visualize_fc_layer(self, weights: np.ndarray, layer_name: str, figsize: tuple, dpi: int, cmap: str) -> None:
        """
        Визуализирует веса полносвязного слоя.

        Аргументы:
            weights (np.ndarray): Веса слоя.
            layer_name (str): Имя слоя.
            figsize (tuple): Размер фигуры.
            dpi (int): Разрешение фигуры.
            cmap (str): Цветовая карта.

        Возвращает:
            None
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(weights, cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.title(f"Веса слоя: {layer_name}")
        plt.xlabel("Входные нейроны")
        plt.ylabel("Выходные нейроны")
        plt.show()


    def _visualize_conv_layer(self, weights: np.ndarray, layer_name: str, figsize: tuple, dpi: int, cmap: str) -> None:
        """
        Визуализирует веса сверточного слоя.

        Аргументы:
            weights (np.ndarray): Веса слоя.
            layer_name (str): Имя слоя.
            figsize (tuple): Размер фигуры.
            dpi (int): Разрешение фигуры.
            cmap (str): Цветовая карта.

        Возвращает:
            None
        """
        num_filters = weights.shape[0]
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 5), dpi=dpi)
        if num_filters == 1:
            axes = [axes]  # Для случая с одним фильтром

        for i, ax in enumerate(axes):
            ax.imshow(weights[i, 0], cmap=cmap)  # Первый канал (входной)
            ax.set_title(f"Фильтр {i + 1}")
            ax.axis('off')

        plt.suptitle(f"Веса слоя: {layer_name}")
        plt.show()


if __name__=='__main__':
    mape = torch.rand((2,2))
    limit_percel = 5
    per_loss_rd = (mape < 0.01 * limit_percel).sum().item() / (mape.numel())
    print(per_loss_rd)

    low_mape_values =mape[mape<0.01*limit_percel].view(-1).sum().item()
    print(low_mape_values)

    # plt.figure(figsize=(12, 6), dpi=60)
    # plt.scatter(mape,mape2, s=3, label='Предсказания модели') 
    # plt.scatter(mape,mape3, color='red', s=6, label='Истинные значения') 
    # plt.xlabel('Топливо')
    # plt.ylabel('Тяга')
    # plt.legend()
    # plt.show()