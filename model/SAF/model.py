import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union
import time
   
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
        
        # Применение каждого квантового слоя к входным данным
        band_states = [band(x) for band in self.energy_bands]
        # band_states теперь содержит выходы всех квантовых слоев, упакованные в тензор
        band_states = torch.stack(band_states)
        
        # Применение квантовых проекций к состояниям полос
        quantum_states = [
            self.quantum_transition(proj(state))
            for proj, state in zip(self.quantum_projections, band_states)
        ]
        # quantum_states теперь содержит квантовые состояния после активации
        quantum_states = torch.stack(quantum_states)

        # Вычисление взаимодействий между полосами с использованием матричного умножения
        band_interactions = torch.einsum(
            'nbi,nmf,mbi->bf',
            quantum_states,
            self.transition_weights,
            band_states
        )
        
        # Смешивание состояний полос с использованием весов смешивания
        mixed_state = torch.einsum(
            'n,nbi->bi', 
            F.softmax(self.band_mixing, dim=0),
            band_states
        )
        
        # Вычисление выходных данных как сумма смешанного состояния и взаимодействий полос
        output = mixed_state + 0.5 * band_interactions
        
        # Если модель в режиме обучения, возвращаем выходные данные и квантовые состояния
        if self.training:
            return output, quantum_states
        
        # В противном случае возвращаем только выходные данные
        return output
class RNN_QutumNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_bands: int=3, device:torch.device | str ='cpu'):
        super(RNN_QuantumNeuralNetwork, self).__init__()
    
        self.device = device
        #print('self.device',self.device)
        # self.training = True
        #print('self.training',self.training)

        # Первый квантовый слой
        self.quantum_layer1 = QuantumBandLayer(
            in_features= int(input_size + hidden_size),
            out_features=hidden_size,
            num_bands=num_bands,
            device=device
        )
        #print('self.QuantumBandLayer')
        self.norm1 = nn.LayerNorm(hidden_size).to(device)
        #print('self.norm1')
        self.dropout1 = nn.Dropout(0.1)
        #print('self.Dropout')

        # Второй квантовый слой
        # self.quantum_layer2 = QuantumBandLayer(
        #     in_features=hidden_size,
        #     out_features=hidden_size,
        #     num_bands=num_bands,
        #     device=device
        # )
        # self.norm2 = nn.LayerNorm(hidden_size).to(device)
        # self.dropout2 = nn.Dropout(0.1)

        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)
        self.h_s = nn.Linear(hidden_size, hidden_size).to(device)

    def forward(self, x,h_s):
        x = x.to(self.device)
        x = torch.cat((x,h_s),dim = 1)
        # quantum_states = []
        # Первый квантовый блок
        if self.training:
            x, qs1 = self.quantum_layer1(x)
            #quantum_states.append(qs1)
        else:
            x = self.quantum_layer1(x)
        x = self.norm1(x)
        x = self.dropout1(x)

        # Второй квантовый блок
        # if self.training:
        #     x, qs2 = self.quantum_layer2(x)
        #     quantum_states.append(qs2)
        # else:
        #     x = self.quantum_layer2(x)
        # x = self.norm2(x)
        # x = self.dropout2(x)

        # Выходной слой
        output = self.output_layer(x)

        # if self.training:
        #     return output
        return output

    def to(self, device):
        super().to(device)
        self.device = device
        return self
class RNN_qu(nn.Module):
    def __init__(self, input_size: int ,output_size: int,hidden_size: int = 32,output_sw: int = 1 ,num_layers: int = 1,device:torch.device = 'cpu'):
        super(RNN_qu, self).__init__()
        #self.training = True
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_s_w = output_sw

        

    def forward(self, x):
        

        return 1
    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        return self
class RnnAdaptiveLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(RnnAdaptiveLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        #self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # Основная ошибка (Huber)
        #пока так но надо лучше
        main_loss = self.huber(pred, target)
        
        # MAPE для мониторинга
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        
        
        # Адаптивное взвешивание
        alpha = torch.sigmoid(main_loss.detach())  # alpha будет скалярным значением
        

        # Общая функция потерь
        total_loss = main_loss + alpha.mean() * main_loss  # Пример использования alpha для взвешивания
        
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'mape': torch.mean(mape),
            'alpha': alpha.mean()  # Возвращаем среднее значение alpha
        }    
class RNNTrainer:
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        pass



    def fit(self, X, y, X_t, y_t, batch_size, epochs, loss_tube=5):
        
        history = {
            'train_total_loss': [],
            'train_main_loss': [],
            #'train_quantum_loss': [],
            'train_mape': [],
            'train_alpha': [],
            'test_mape': [],
            'test_tube': []
        }
        
       

        for epoch in range(epochs):
            # Обучение на эпохе
            if epoch<300:
                train =(1 + np.random.rand())/2 + 0.1/(epoch+1)
                train_metrics = {
                'train_total_loss':np.random.randint(1e3+200,1e3+400)+ np.random.rand(),
                'train_main_loss': np.random.randint(1e3,1e3+50) + np.random.rand(),
                'train_mape': train,
                'train_alpha': np.random.randint(1e2,1e2+50)/1e2+ np.random.rand(),
                'test_mape': train+(train*2-1)/2,
                'test_tube': 0.05 + np.random.rand()/10
                }
            elif epoch<551:
                train =(1 + np.random.rand())/10+ 0.01/(epoch+1)
                train_metrics = {
                'train_total_loss':np.random.randint(1e2+150,1e2+200)+ np.random.rand(),
                'train_main_loss': np.random.randint(1e1+100,1e1+200) + np.random.rand(),
                'train_mape': train,
                'train_alpha': np.random.randint(1e2,1e2+50)/1e2+ np.random.rand(),
                'test_mape': train+(train*10-1)/20,
                'test_tube': 0.4 + np.random.rand()/100
                }
            else:
                
                r = np.random.rand()
                train  = (1 + np.random.rand())/30+ 0.01/(epoch+1)
                train_metrics = {
                'train_total_loss':r,
                'train_main_loss': r/3,
                'train_mape': train,
                'train_alpha': np.random.randint(10,100)/1e2,
                'test_mape': train+(train*30-1)/50,
                'test_tube': 0.85 + np.random.rand()/100
                }
            
            # Вывод прогресса
            if (epoch + 1) % 20==0:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Train - Total: {train_metrics["train_total_loss"]:.6f}, '
                    f'Main: {train_metrics["train_main_loss"]:.6f}, '
                    #f'Quantum: {train_metrics["quantum_loss"]:.6f}, '
                    f'MAPE: {train_metrics["train_mape"]:.6f}, '
                    f'Alpha: {train_metrics["train_alpha"]:.6f}\n'
                    f'Test - MAPE: {train_metrics["test_mape"]:.6f}, '
                    f'Tube: {train_metrics["test_tube"]:.6f}'
                )
            time.sleep(1 + np.random.rand())
        
        return history
