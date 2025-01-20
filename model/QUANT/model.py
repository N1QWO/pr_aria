import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union

    
class QuantumBandLayer(nn.Module):
    """
    Класс квантового слоя.

    Этот слой использует квантовые энергетические зоны и переходные веса для
    обработки входных данных. Он включает в себя квантовые проекции и
    активацию.

    Параметры:
    - in_features (int): Размер входного слоя.
    - out_features (int): Размер выходного слоя.
    - num_bands (int): Число квантовых полос (по умолчанию 3).
    - temperature (float): Температура для активации (по умолчанию 1.0).
    - device (torch.device | str): Устройство для выполнения (CPU или GPU, по умолчанию 'cpu').
    """

    def __init__(self, in_features: int, out_features: int, num_bands: int = 3, temperature: float = 1.0, device: torch.device | str = 'cpu'):
        super(QuantumBandLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bands = num_bands
        self.temperature = temperature
        self.device = device
        self.training = True
        # Энергетические зоны
        self.energy_bands = nn.ModuleList([
            nn.Linear(in_features, out_features).to(device)
            for _ in range(num_bands)
        ])
        
        # Веса переходов
        self.transition_weights = nn.Parameter(
            torch.randn((num_bands, num_bands, out_features), device=device) * 0.02
        )
        
        # Квантовые проекции
        self.quantum_projections = nn.ModuleList([
            nn.Linear(out_features, out_features).to(device)
            for _ in range(num_bands)
        ])
        
        
        self.band_mixing = nn.Parameter(torch.randn(num_bands, device=device) * 0.02)
        self.outp_w = nn.Parameter(torch.randn(2, device=device))
        self.activation = nn.GELU()

    def to(self, device: torch.device | str) -> 'QuantumBandLayer':
        """
        Перемещение слоя на указанное устройство.

        Параметры:
        - device (torch.device | str): Устройство для перемещения слоя.

        Возвращает:
        - self: Объект слоя.
        """
        super().to(device)
        self.device = device
        return self
        
    def quantum_transition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Квантовый переход с активацией.

        Параметры:
        - x (torch.Tensor): Входные данные.

        Возвращает:
        - (torch.Tensor): Примененные активации.
        """
        safe_x = torch.clamp(x, min=-2, max=2)
        return self.activation(safe_x) * torch.sigmoid(safe_x)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход через слой.

        Параметры:
        - x (torch.Tensor): Входные данные.

        Возвращает:
        - output (torch.Tensor): Выходные данные слоя.
        - quantum_states (torch.Tensor): Квантовые состояния (если модель в режиме обучения).
        """
        # Перемещение входных данных на заданное устройство (CPU или GPU)
        x = x.to(self.device)
        
        # Получение размера батча (количество входных примеров)
        batch_size = x.size(0)
        # print('x',x.shape)
        # Применение каждого квантового слоя к входным данным
        band_states = [band(x) for band in self.energy_bands]
        # print('len(band_states)',len(band_states),'band_states[0].shape',band_states[0].shape)
        # band_states теперь содержит выходы всех квантовых слоев, упакованные в тензор
        band_states = torch.stack(band_states)
        # print('band_states',band_states.shape)
        
        # Применение квантовых проекций к состояниям полос
        quantum_states = [
            self.quantum_transition(proj(state))
            for proj, state in zip(self.quantum_projections, band_states)
        ]
        # print('quantum_states',len(quantum_states),'quantum_states[0]',quantum_states[0].shape)
        # quantum_states теперь содержит квантовые состояния после активации
        quantum_states = torch.stack(quantum_states)
        # print('quantum_states',quantum_states.shape)
        # Вычисление взаимодействий между полосами с использованием матричного умножения
        band_interactions = torch.einsum(
            'nbi,nmf,mbi->bf',
            quantum_states,
            self.transition_weights,
            band_states
        )
        # print('self.transition_weights',self.transition_weights.shape)
        # print('quantum_states * self.transition_weights * band_states = band_interactions')
        # print('band_interactions',band_interactions.shape)
        # Смешивание состояний полос с использованием весов смешивания
        f_s = F.softmax(self.band_mixing, dim=0)
        # print('f_s = F.softmax(self.band_mixing, dim=0)',f_s.shape)
        mixed_state = torch.einsum(
            'n,nbi->bi', 
            f_s,
            band_states
        )
        # print('f_s * band_states = mixed_state')
        # print('mixed_state',mixed_state.shape)
        
        # Вычисление выходных данных как сумма смешанного состояния и взаимодействий полос
        t_outp_w = F.softmax(self.outp_w, dim=0)
        output = t_outp_w[0]*mixed_state + t_outp_w[1] * band_interactions

        # print('output',output.shape)
        # exit()
        # Если модель в режиме обучения, возвращаем выходные данные и квантовые состояния
        if self.training:
            return output, quantum_states
        
        # В противном случае возвращаем только выходные данные
        return output
class QuantumNeuralNetwork(nn.Module):
    """
    Класс квантовой нейронной сети.

    Эта модель использует квантовые слои для обработки входных данных и
    включает в себя нормализацию и дроп-аут для регуляризации.

    Параметры:
    - input_size (int): Размер входного слоя.
    - hidden_size (int): Размер скрытого слоя.
    - output_size (int): Размер выходного слоя.
    - num_bands (int): Число квантовых полос (по умолчанию 1).
    - device (torch.device | str): Устройство для выполнения (CPU или GPU, по умолчанию 'cpu').
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_bands: int = 1, device: torch.device | str = 'cpu'):
        super(QuantumNeuralNetwork, self).__init__()
        self.device = device
        self.training = True
        
        # Первый квантовый слой
        self.quantum_layer1 = QuantumBandLayer(
            in_features=input_size,
            out_features=hidden_size,
            num_bands=num_bands,
            device=device
        )
        self.norm1 = nn.LayerNorm(hidden_size).to(device)
        self.dropout1 = nn.Dropout(0.1)
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход через сеть.

        Параметры:
        - x (torch.Tensor): Входные данные.

        Возвращает:
        - output (torch.Tensor): Выходные данные модели.
        - quantum_states (list): Состояния квантовых слоев (если модель в режиме обучения).
        """
        x = x.to(self.device)
        quantum_states = []
        self.quantum_layer1.training = self.training
        # Первый квантовый блок
        if self.training:
            x, qs1 = self.quantum_layer1(x)
            quantum_states.append(qs1)
        else:
            x = self.quantum_layer1(x)
        
        x = self.norm1(x)
        x = self.dropout1(x)
        
        # Выходной слой
        output = self.output_layer(x)
        
        if self.training:
            return output, quantum_states
        return output
    
    def to(self, device: torch.device | str) -> 'QuantumNeuralNetwork':
        """
        Перемещение модели на указанное устройство.

        Параметры:
        - device (torch.device | str): Устройство для перемещения модели.

        Возвращает:
        - self: Объект модели.
        """
        super().to(device)
        self.device = device
        return self

class AdaptiveLoss(nn.Module):
    """
    Класс адаптивной функции потерь.

    Эта функция потерь сочетает в себе Huber Loss и MSE Loss с
    квантовой регуляризацией. Она также вычисляет MAPE для мониторинга
    точности предсказаний.

    Параметры:
    - delta (float): Параметр для Huber Loss (по умолчанию 0.01).
    - quantum_weight (float): Веса для квантовой регуляризации (по умолчанию 0.1).
    """

    def __init__(self, delta: float = 0.01, quantum_weight: float = 0.1):
        super(AdaptiveLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)  # Инициализация Huber Loss
        self.mse = nn.MSELoss()  # Инициализация MSE Loss
        self.quantum_weight = quantum_weight  # Установка веса для квантовой регуляризации
        self.loss_tube = None
       
    def forward(self, pred: torch.Tensor, target: torch.Tensor, quantum_states: list) -> dict:
        """
        Прямой проход для вычисления функции потерь.

        Параметры:
        - pred (torch.Tensor): Предсказанные значения.
        - target (torch.Tensor): Целевые значения.
        - quantum_states (list): Состояния квантовых слоев.

        Возвращает:
        - dict: Словарь с общей функцией потерь, основной потерей, квантовой потерей,
                 MAPE и значением alpha.
        """
        # Основная ошибка (Huber)
        main_loss = self.huber(pred, target)
        
        # Вычисление MAPE (Mean Absolute Percentage Error) для мониторинга
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-10)
        
        # Квантовая регуляризация
        quantum_losses = []
        for qs in quantum_states:
            # Вычисление среднего состояния
            mean_state = qs.mean(dim=0, keepdim=True)  # [1, batch_size, features]
            
            # Приведение размерностей в соответствие
            current_qs = qs.transpose(0, 1)  # [batch_size, num_bands, features]
            current_mean = mean_state.expand_as(qs).transpose(0, 1)  # [batch_size, num_bands, features]
            
            # Вычисление MSE для квантового состояния
            quantum_losses.append(self.mse(current_qs, current_mean))
        
        # Суммирование квантовых потерь
        quantum_loss = sum(quantum_losses)
        
        # Адаптивное взвешивание
        alpha = torch.sigmoid(main_loss.detach())
        
        # Общая функция потерь
        total_loss = main_loss + self.quantum_weight * alpha * quantum_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'quantum_loss': quantum_loss,
            'mape': torch.mean(mape),
            'tube':(mape < 0.01 * self.loss_tube).sum() / mape.numel(),
            'alpha': alpha
        }    
class QuantumTrainer:
    """
    Класс для обучения квантовой нейронной сети.

    Этот класс управляет процессом обучения модели, включая настройку
    оптимизатора, функции потерь и планировщика обучения.

    Параметры:
    - model (nn.Module): Модель квантовой нейронной сети.
    - learning_rate (float): Скорость обучения (по умолчанию 0.001).
    - device (str): Устройство для выполнения (CPU или GPU, по умолчанию 'cpu').
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.001,inf_per_epoch: int = 20, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = AdaptiveLoss(delta=0.01).to(device)  # Инициализация функции потерь
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Инициализация оптимизатора
        self.inf_per_epoch = inf_per_epoch
        self.history = {
            'train_total_loss': [],
            'train_main_loss': [],
            'train_quantum_loss': [],
            'train_mape': [],
            'train_alpha': [],
            'train_tube': [],
            'test_total_loss': [],
            'test_main_loss': [],
            'test_quantum_loss': [],
            'test_mape': [],
            'test_alpha': [],
            'test_tube': []
        }
    def create_scheduler(self, step_size: int = 50, gamma: float = 0.5):
        """
        Создание планировщика для изменения скорости обучения.

        Параметры:
        - step_size (int): Количество шагов до изменения скорости обучения (по умолчанию 50).
        - gamma (float): Коэффициент уменьшения скорости обучения (по умолчанию 0.5).

        Возвращает:
        - torch.optim.lr_scheduler.StepLR: Планировщик для изменения скорости обучения.
        """
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=step_size, 
            gamma=gamma
        )

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Оценка модели на тестовых данных.

        Параметры:
        - X (torch.Tensor): Входные данные для оценки.
        - y (torch.Tensor): Целевые значения для оценки.
        - loss_tube (float): Параметр для вычисления MAPE (по умолчанию 5).

        Возвращает:
        - tuple: Метрики оценки и процент потерь ниже заданного порога.
        """
        self.model.eval()  # Перевод модели в режим оценки
        with torch.no_grad():
            y_pred = self.model(X)  # Получение предсказаний
            dummy_states = [torch.zeros_like(y_pred).unsqueeze(0)]  # Создание фиктивных квантовых состояний
            metrics = self.criterion(y_pred, y, dummy_states)  # Вычисление метрик
            
        
        self.model.train()  # Возврат модели в режим обучения
        return metrics
    def add_history(self, train_metrics: dict, test_metrics: dict):
        for key in train_metrics:
            self.history['train_' + key].append(train_metrics[key])
        for key in test_metrics:
            self.history['test_' + key].append(test_metrics[key].item())
    def train_epoch(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict:
        """
        Обучение модели на одной эпохе.

        Параметры:
        - X (torch.Tensor): Входные данные для обучения.
        - y (torch.Tensor): Целевые значения для обучения.
        - batch_size (int): Размер батча.

        Возвращает:
        - dict: Средние метрики за эпоху.
        """
        dataset_size = X.shape[0]
        indices = torch.randperm(dataset_size)  # Перемешивание индексов
        X_shuffled = X[indices].to(self.device)  # Перемешанные входные данные
        y_shuffled = y[indices].to(self.device)  # Перемешанные целевые значения
        
        epoch_metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'quantum_loss': 0.0,
            'mape': 0.0,
            'alpha': 0.0,
            'tube': 0.0
        }
        n_batches = 0
        
        for i in range(0, dataset_size, batch_size):
            X_batch = X_shuffled[i:i+batch_size]  # Получение батча входных данных
            y_batch = y_shuffled[i:i+batch_size]  # Получение батча целевых значений
            
            self.optimizer.zero_grad()  # Обнуление градиентов
            
            # Получение предсказаний и квантовых состояний
            predictions, quantum_states = self.model(X_batch)
            
            # Вычисление метрик потерь
            metrics = self.criterion(predictions, y_batch, quantum_states)
            metrics['total_loss'].backward()  # Обратное распространение
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Обрезка градиентов
            self.optimizer.step()  # Обновление параметров модели
            
            
            # Суммирование метрик за батч
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key].item()
            n_batches += 1
            
        return {k: v / n_batches for k, v in epoch_metrics.items()}  # Возврат средних метрик за эпоху

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int, epochs: int, loss_tube: float = 5) -> dict:
        """
        Обучение модели на заданное количество эпох.

        Параметры:
        - X (torch.Tensor): Входные данные для обучения.
        - y (torch.Tensor): Целевые значения для обучения.
        - X_t (torch.Tensor): Входные данные для тестирования.
        - y_t (torch.Tensor): Целевые значения для тестирования.
        - batch_size (int): Размер батча.
        - epochs (int): Количество эпох для обучения.
        - loss_tube (float): Параметр для вычисления MAPE (по умолчанию 5).

        Возвращает:
        - dict: История метрик обучения и тестирования.
        """
        self.model.to(self.device)  # Перемещение модели на устройство
        self.criterion.loss_tube = loss_tube
        X = X.to(self.device)
        y = y.to(self.device)
        X_t = X_t.to(self.device)
        y_t = y_t.to(self.device)
        scheduler = self.create_scheduler()  # Создание планировщика
        best_test_mape = float('inf')  # Лучшее значение MAPE
        best_model_weights = None  # Лучшие веса модели

        
        for epoch in range(epochs):
            # Обучение на эпохе
            train_metrics = self.train_epoch(X, y, batch_size)
            
            # Шаг планировщика
            scheduler.step()
            
            # Сохранение метрик обучения
            # Оценка на тестовых данных
            test_metrics = self.evaluate(X_t, y_t)
            self.add_history(train_metrics,test_metrics)

            if (test_metrics['tube'].item() +  train_metrics['mape'])/2 < best_test_mape:
                best_test_mape = (test_metrics['tube'].item() +  train_metrics['mape'])/2
                best_model_weights = self.model.state_dict().copy()
                
            
            # Вывод прогресса
            if (epoch + 1) % self.inf_per_epoch == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Train - Total: {train_metrics["total_loss"]:.6f}, '
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'Quantum: {train_metrics["quantum_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}, '
                    f'Alpha: {train_metrics["alpha"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_metrics["tube"]:.6f}'
                )
        torch.save(best_model_weights, 'best_model_weights.pth')

        return self.history  # Возврат истории метрик
    



# Пример использования:
"""
model = QuantumNeuralNetwork(...)
trainer = QuantumTrainer(
    model=model,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

history = trainer.fit(
    X=X_train,
    y=y_train,
    X_t=X_test,
    y_t=y_test,
    batch_size=32,
    epochs=700,
    loss_tube=5
)
"""



if __name__=='__main__':

        # Создаем данные
    train_dataset = None
    test_dataset = None 

    # Создаем даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    X_train, y_train = train_dataset.X, train_dataset.y
    X_test, y_test = test_dataset.X, test_dataset.y

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_,out_ = X_train.shape[1],y_train.shape[1]
    model = QuantumNeuralNetwork(
        input_size=in_,
        hidden_size=8,
        output_size = out_,
        num_bands=1,
        device=device
    )
    
    trainer = QuantumTrainer(
        model=model,
        learning_rate=0.001,
        device=device
    )

    history = trainer.fit(
        X=X_train,
        y=y_train,
        X_t=X_test,
        y_t=y_test,
        batch_size=32,
        epochs=700,
        loss_tube=5
    )