import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        scaled_x = safe_x / self.temperature

        return self.activation(scaled_x) * torch.sigmoid(scaled_x)
    
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
class RNN_QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_bands: int=3,temperature:float = 1.0, device:torch.device | str ='cpu'):
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
            temperature = temperature,
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
class RNN_quantum(nn.Module):
    def __init__(self, input_size: int ,output_size: int,hidden_size: int = 32,output_sw: int = 1 ,num_layers: int = 1,num_bands:int  = 3,temperature:float = 1.0,device:torch.device = 'cpu'):
        super(RNN_quantum, self).__init__()
        #self.training = True
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_s_w = output_sw

        self.layers = nn.ModuleList([RNN_QuantumNeuralNetwork(input_size = input_size, hidden_size = hidden_size,output_size = hidden_size,num_bands =  num_bands,temperature=temperature,device = device) if i == 0
                                    else RNN_QuantumNeuralNetwork(input_size = hidden_size,hidden_size =  hidden_size,output_size = hidden_size,num_bands = num_bands,temperature=temperature,device= device)
                                        for i in range(num_layers)])
        self.layers_follow = nn.ModuleList([RNN_QuantumNeuralNetwork(input_size = hidden_size,hidden_size =  hidden_size,output_size = hidden_size,num_bands = num_bands,temperature=temperature, device= device)
                                        for i in range(num_layers)])
        
        self.fc = nn.Linear(in_features = hidden_size,out_features = output_size,device = device).to(device)

    def forward(self, x):
        l = x.size(1)
        # Начальные скрытые состояния и состояния ячеек для каждого слоя
        h = [torch.zeros(x.size(0), self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        output = torch.zeros(x.size(0), self.output_s_w).to(self.device)
       
        # Проходим по временным шагам входных данных
        for t in range(l + self.output_s_w):
            if t< x.size(1):
                input_t = x[:, t, :]  # Вход на текущем временном шаге
            

                for i, layer in enumerate(self.layers):
                    # if self.training:
                    h[i] = layer(input_t, h[i])
                    # else:
                    #     h[i]  = layer(input_t, h[i])
                    input_t = h[i]  # Передаем выход текущего слоя на вход следующему

            else:
                for i, layer in enumerate(self.layers_follow):
                    # if self.training:
                    h[i] = layer(h[i], h[i])
                    # else:
                    #     h[i]  = layer(h[i], h[i])
                output[:,t - l] = self.fc(h[-1]).squeeze()

        return output
    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        return self
class RnnAdaptiveLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(RnnAdaptiveLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.loss_tube = None
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
        per_loss_rd = (mape < 0.01 * self.loss_tube).sum() / (mape.numel())
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'mape': torch.mean(mape),
            'alpha': alpha.mean(),  # Возвращаем среднее значение alpha
            'tube': per_loss_rd
        }    
class RNNTrainer:
    def __init__(self, model, learning_rate=0.001,inf_per_epoch: int = 20, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = RnnAdaptiveLoss()  # Используем MSELoss + Hyber 
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.inf_per_epoch = inf_per_epoch
        self.history = {
            'train_total_loss': [],
            'train_main_loss': [],
            'train_mape': [],
            'train_alpha': [],
            'train_tube': [],
            'test_total_loss': [],
            'test_main_loss': [],
            'test_mape': [],
            'test_alpha': [],
            'test_tube': []
        }

    def create_scheduler(self):
        # Пример создания планировщика
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.25)
    
    def add_history(self, train_metrics: dict, test_metrics: dict):
        for key in train_metrics:
            self.history['train_' + key].append(train_metrics[key])
        for key in test_metrics:
            self.history['test_' + key].append(test_metrics[key].item())
    def train_epoch(self, X, y, batch_size):
        dataset_size = X.shape[0]
        indices = torch.randperm(dataset_size)
        X_shuffled = X[indices].to(self.device)
        y_shuffled = y[indices].to(self.device)
        
        epoch_metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'mape': 0.0,
            'alpha': 0.0,
            'tube': 0.0
        }
        n_batches = 0
        
        for i in range(0, dataset_size, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            #print(y_batch.shape)
            self.optimizer.zero_grad()
            
            # В режиме train модель возвращает (predictions)
            predictions = self.model(X_batch)
            #print(predictions.shape)
            metrics = self.criterion(predictions, y_batch)
            metrics['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key].item()
            n_batches += 1
            
        return {k: v / n_batches for k, v in epoch_metrics.items()}

    def evaluate(self, X, y, loss_tube: float | int =5):
        self.model.eval()
        with torch.no_grad():
            # В режиме eval модель возвращает только предсказания
            y_pred  = self.model(X)
            # Создаем фиктивные quantum_states для criterion
            #dummy_states = [torch.zeros_like(y_pred).unsqueeze(0)]
            metrics = self.criterion(y_pred, y)
            

            #print('3')
        
        self.model.train()
        return metrics

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int, epochs:int, loss_tube:int =5):
        self.model.to(self.device)
        self.criterion.loss_tube = loss_tube
        X = X.to(self.device)
        y = y.to(self.device)
        X_t = X_t.to(self.device)
        y_t = y_t.to(self.device)
        scheduler = self.create_scheduler()
        best_model_weights = None
        best_test_mape = float('inf')

        
        for epoch in range(epochs):
            # Обучение на эпохе
            train_metrics = self.train_epoch(X, y, batch_size)
            
            # Шаг планировщика
            scheduler.step()
            
            # Оценка на тестовых данных
            test_metrics = self.evaluate(X_t, y_t, loss_tube)
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
                    #f'Quantum: {train_metrics["quantum_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}, '
                    f'Alpha: {train_metrics["alpha"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_metrics["tube"]:.6f}'
                )
        torch.save(best_model_weights, 'best_model_weights.pth')
        return self.history
