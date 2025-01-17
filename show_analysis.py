from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd



class LossVisualizer:
    """
    Класс для визуализации потерь и метрик модели.

    Параметры:
    - losses (dict): Словарь, содержащий метрики потерь и оценки.
    - data (optional): Дополнительные данные для визуализации (по умолчанию None).

    Пример:
        losses: {
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

    def show_training_loss(self) -> None:
        """Отображает график основной функции потерь во время обучения."""
        main_loss = self.losses['train_main_loss']

        plt.figure(figsize=(12, 6), dpi=60)
        plt.plot(main_loss, label='Основная потеря', color='blue', alpha=0.7)
        plt.xlabel('Эпоха обучения')  # Подпись оси X
        plt.ylabel('Значение функции потерь')  # Подпись оси Y
        plt.title('График основной функции потерь')
        plt.legend()
        plt.show()

    def show_test_tube(self) -> None:
        """Отображает график отношения количества попаданий данных в 5% барьер на тестовых данных."""
        test_tube = self.losses['test_tube']

        plt.figure(figsize=(12, 6), dpi=60)
        plt.plot(test_tube, label='Отношение попаданий', color='green', alpha=0.7)
        plt.xlabel('Эпоха обучения')  # Подпись оси X
        plt.ylabel('Отношение попаданий в 5% барьер')  # Подпись оси Y
        plt.title('График отношения попаданий на тестовых данных')
        plt.legend()
        plt.show()

    def show_mape(self, show_all: bool = True, show_chip: bool = True, start_epoch: Optional[int] = None) -> None:
        """
        Отображает графики MAPE для обучающей и тестовой выборок.

        Параметры:
        - show_all (bool): Отображать ли все эпохи (по умолчанию True).
        - show_chip (bool): Отображать ли первые 150 эпох (по умолчанию True).
        - start_epoch (optional): Начальная эпоха для отображения (по умолчанию None).
        """
        train_mape = self.losses['train_mape']
        test_mape = self.losses['test_mape']

        if show_all:
            plt.figure(figsize=(12, 6), dpi=60)
            plt.plot(train_mape, label='Ошибка на обучающей выборке', alpha=0.7)
            plt.plot(test_mape, label='Ошибка на тестовой выборке', color='red')
            plt.xlabel('Эпоха обучения')  
            plt.ylabel('Значение MAPE')  
            plt.title('График MAPE для обучающей и тестовой выборок')
            plt.legend()
            plt.show()

        if show_chip:
            plt.figure(figsize=(12, 6), dpi=60)
            plt.plot(train_mape[:150], label='Ошибка на обучающей выборке', alpha=0.7)
            plt.plot(test_mape[:150], label='Ошибка на тестовой выборке', color='red')
            plt.xlabel('Эпоха обучения')  
            plt.ylabel('Значение MAPE')  
            plt.title('График MAPE для первых 150 эпох')
            plt.legend()
            plt.show()

        if start_epoch is not None:
            plt.figure(figsize=(12, 6), dpi=60)
            plt.plot(train_mape[start_epoch:], label='Ошибка на обучающей выборке', alpha=0.7)
            plt.plot(test_mape[start_epoch:], label='Ошибка на тестовой выборке', color='red')
            plt.xlabel('Эпоха обучения')  
            plt.ylabel('Значение MAPE')  
            plt.title(f'График MAPE, начиная с эпохи {start_epoch}')
            plt.legend()
            plt.show()

    def histogram_mape(self, model: nn.Module, X: torch.Tensor, target: torch.Tensor, limit_percel: float | None = None,keras = False) -> None:
        """Отображает гистограмму распределения MAPE на тестовых данных."""
        
        if keras:
            pred = model.predict(X)
        else:
            model.eval()
            pred = model.forward(X).to('cpu')
            model.train()

        target = target.to('cpu')
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        per_loss_rd = (mape < 0.01 * limit_percel).sum().item() / (mape.numel())
        # Проверка размерности MAPE
        mape = mape.view(-1)
        # Фильтрация значений MAPE
        if limit_percel is not None:
            low_mape_values = mape[mape<0.01*limit_percel]
            lete = f'< {0.01 * limit_percel}' 
        else:
            low_mape_values = mape
            lete = 'всех значения'

        plt.figure(figsize=(12, 6), dpi=80)
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