import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, device: torch.device | str ='cpu'):
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
        for layer in [self.query, self.key, self.value, self.fc_out]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
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
        super(FeedForwardRegression, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Применяем Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Применяем Dropout
        x = self.fc3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1, device: torch.device | str ='cpu'):
        super(Transformer, self).__init__()
        self.device = device
        self.input_projection = nn.Linear(input_dim, hidden_dim, device=self.device)
        self.self_attention_layers = nn.ModuleList(
            [SelfAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, device=self.device) for _ in range(num_layers)]
        )
        self.feed_forward = FeedForwardRegression(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, device=self.device)
        self.layer_norms = nn.LayerNorm(hidden_dim, device=self.device)  # Нормализация после каждого слоя

    def forward(self, x: Tensor) -> Tensor:
        # Преобразуем входные данные
        x = self.input_projection(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)

        for i, attention_layer in enumerate(self.self_attention_layers):
            x = attention_layer(x)  # Применяем механизм внимания
            #x = self.layer_norms[i](x)  # Применяем нормализацию после слоя внимания

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
        #self.loss = nn.MSELoss()
        #self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.tensor, target: torch.tensor) -> dict:
        #пока так но надо лучше
        main_loss = self.loss(pred, target)
        
        # MAPE для мониторинга
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        
        return {
            'main_loss': main_loss,
            'mape': torch.mean(mape),
        }    

class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float=0.001, device: torch.device | str ='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = AdaptiveLoss(delta = 1)  # Используем MSELoss + Hyber 
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
            # Создаем фиктивные quantum_states для criterion
            #dummy_states = [torch.zeros_like(y_pred).unsqueeze(0)]
            metrics = self.criterion(y_pred, y)
            
            loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
            #print('2')
            per_loss_rd = (loss_rd < 0.01 * loss_tube).sum().item() / (loss_rd.numel())
            #print('3')
        
        self.model.train()
        return metrics, per_loss_rd

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int = 512, epochs:int = 100, loss_tube: int = 5) -> dict:
        self.model.to(self.device)
        X = X.to(self.device)
        y = y.to(self.device)
        X_t = X_t.to(self.device)
        y_t = y_t.to(self.device)
        scheduler = self.create_scheduler()
        
        history = {
            'train_main_loss': [],
            'train_mape': [],
            'test_mape': [],
            'test_tube': []
        }
        
        for epoch in range(epochs):
            # Обучение на эпохе
            train_metrics = self.train_epoch(X, y, batch_size)
            
            # Шаг планировщика
            scheduler.step()
            
            # Сохранение метрик обучения
            history['train_main_loss'].append(train_metrics['main_loss'])
            history['train_mape'].append(train_metrics['mape'])
            
            # Оценка на тестовых данных
            test_metrics, test_tube = self.evaluate(X_t, y_t, loss_tube)
            history['test_mape'].append(test_metrics['mape'].item())
            history['test_tube'].append(test_tube)
            
            # Вывод прогресса
            if (epoch + 1) % 20 == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_tube:.6f}'
                )
        
        return history
