import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor



class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, device: torch.device | str ='cpu'):
        """
        Инициализация слоя самовнимания (Self-Attention).

        Этот слой реализует механизм самовнимания, который позволяет модели фокусироваться на различных частях входных данных.

        Параметры:
        - embed_dim (int): Размерность векторного представления (embedding dimension).
        - num_heads (int): Количество голов (heads) в механизме внимания.
        - dropout (float): Вероятность обнуления (dropout rate) для регуляризации (по умолчанию 0.1).
        - device (torch.device | str): Устройство для размещения тензоров (CPU или GPU, по умолчанию 'cpu').

        Исключения:
        - ValueError: Если размерность векторного представления не делится на количество голов.
        """
        super(SelfAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        self.query = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.key = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.value = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.fc_out = nn.Linear(embed_dim, embed_dim, device=self.device)

        # Инициализация весов нормальным распределением
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Инициализация весов слоев нормальным распределением.
        """
        for layer in [self.query, self.key, self.value, self.fc_out]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход через слой самовнимания.

        Параметры:
        - x (Tensor): Входные данные с размерностью (batch_size, seq_len, embed_dim).

        Возвращает:
        - Tensor: Выходные данные с размерностью (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("bhqd,bhkd->bhqk", Q, K)
        scaling_factor = self.head_dim ** 0.5
        attention = F.softmax(energy / scaling_factor, dim=-1)

        out = torch.einsum("bhqk,bhvd->bhqd", attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        out = self.fc_out(out)
        out = self.dropout(out)  # Применяем Dropout

        return out

class FeedForwardRegression(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, device: torch.device | str ='cpu'):
        """
        Инициализация полносвязного слоя для регрессии.

        Этот слой состоит из двух полносвязных слоев с активацией ReLU и регуляризацией Dropout.

        Параметры:
        - input_dim (int): Размерность входных данных.
        - hidden_dim (int): Размерность скрытого слоя.
        - output_dim (int): Размерность выходных данных.
        - dropout (float): Вероятность обнуления (dropout rate) для регуляризации (по умолчанию 0.1).
        - device (torch.device | str): Устройство для размещения тензоров (CPU или GPU, по умолчанию 'cpu').
        """
        super(FeedForwardRegression, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход через полносвязный слой.

        Параметры:
        - x (Tensor): Входные данные с размерностью (batch_size, input_dim).

        Возвращает:
        - Tensor: Выходные данные с размерностью (batch_size, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Применяем Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Применяем Dropout
        x = self.fc3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1, device: torch.device | str ='cpu'):
        """
        Инициализация трансформера.

        Этот класс реализует архитектуру трансформера, состоящую из слоев самовнимания и полносвязных слоев.

        Параметры:
        - input_dim (int): Размерность входных данных.
        - hidden_dim (int): Размерность скрытого слоя.
        - output_dim (int): Размерность выходных данных.
        - num_heads (int): Количество голов в механизме внимания.
        - num_layers (int): Количество слоев самовнимания.
        - dropout (float): Вероятность обнуления (dropout rate) для регуляризации (по умолчанию 0.1).
        - device (torch.device | str): Устройство для размещения тензоров (CPU или GPU, по умолчанию 'cpu').
        """
        super(Transformer, self).__init__()
        self.device = device
        self.input_projection = nn.Linear(input_dim, hidden_dim, device=self.device)
        self.self_attention_layers = nn.ModuleList(
            [SelfAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, device=self.device) for _ in range(num_layers)]
        )
        self.feed_forward = FeedForwardRegression(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, device=self.device)
        self.layer_norms = nn.LayerNorm(hidden_dim, device=self.device)  # Нормализация после каждого слоя

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход через трансформер.

        Параметры:
        - x (Tensor): Входные данные с размерностью (batch_size, seq_len, input_dim).

        Возвращает:
        - Tensor: Выходные данные с размерностью (batch_size, output_dim).
        """
        # Преобразуем входные данные
        x = self.input_projection(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)

        for i, attention_layer in enumerate(self.self_attention_layers):
            x = attention_layer(x)  # Применяем механизм внимания
            # x = self.layer_norms[i](x)  # Применяем нормализацию после слоя внимания

        # Применяем полносвязный слой для регрессии
        x = self.layer_norms(x)  # Применяем нормализацию после полносвязного слоя
        x = self.feed_forward(x)

        # Агрегируем выходные данные по временным шагам
        x = x.mean(dim=1)  # Получаем среднее значение по временным шагам

        return x
    

class AdaptiveLoss(nn.Module):
    def __init__(self, delta: float = 0.01):
        super(AdaptiveLoss, self).__init__()
        self.loss = nn.HuberLoss(delta=delta)
        self.loss_tube = None
        #self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.tensor, target: torch.tensor) -> dict:
        #пока так но надо лучше
        main_loss = self.loss(pred, target)
        
        # MAPE для мониторинга
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        per_loss_rd = (mape < 0.01 * self.loss_tube).sum() / (mape.numel())
        return {
            'main_loss': main_loss,
            'mape': torch.mean(mape),
            'tube': per_loss_rd
        }    
  

class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float=0.001, inf_per_epoch: int = 20, device: torch.device | str ='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = AdaptiveLoss()  # Используем MSELoss + Hyber 
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.inf_per_epoch = inf_per_epoch
        self.history = {
            'train_main_loss': [],
            'train_mape': [],
            'train_tube': [],
            'test_main_loss': [],
            'test_mape': [],
            'test_tube': []
        }
    def create_scheduler(self):
        # Пример создания планировщика
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict:
        dataset_size = X.shape[0]
        indices = torch.randperm(dataset_size)
        X_shuffled = X[indices].to(self.device)
        y_shuffled = y[indices].to(self.device)
        
        epoch_metrics = {
            'main_loss': 0.0,
            'mape': 0.0,
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
            metrics['main_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key].item()
            n_batches += 1
            
        return {k: v / n_batches for k, v in epoch_metrics.items()}

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, loss_tube: float | int = 5):
        self.model.eval()
        with torch.no_grad():
            # В режиме eval модель возвращает только предсказания
            y_pred  = self.model(X)
            metrics = self.criterion(y_pred, y)
            
        
        self.model.train()
        return metrics
    
    def add_history(self, train_metrics: dict, test_metrics: dict):
        for key in train_metrics:
            self.history['train_' + key].append(train_metrics[key])
        for key in test_metrics:
            self.history['test_' + key].append(test_metrics[key].item())

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int = 512, epochs:int = 100, loss_tube: int = 5) -> dict:
        self.model.to(self.device)
        self.criterion.loss_tube = loss_tube
        X = X.to(self.device)
        y = y.to(self.device)
        X_t = X_t.to(self.device)
        y_t = y_t.to(self.device)
        scheduler = self.create_scheduler()
        best_test_mape = float('inf')
        best_model_weights = None
        
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
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_metrics["tube"]:.6f}'
                )
        torch.save(best_model_weights, 'best_model_weights.pth')
        return self.history