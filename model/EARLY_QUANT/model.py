import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class QuantumLayer(nn.Module):
    def __init__(self, in_features, out_features, device='cpu', dropout_rate=0.1):
        super(QuantumLayer, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        
        # Инициализация весов с правильным масштабированием
        self.scale = 1 / np.sqrt(out_features)
        
        # Линейные преобразования для A и B
        self.W_a = nn.Linear(in_features, out_features)
        self.W_b = nn.Linear(in_features, out_features)
        
        # Квантовые преобразования
        self.W_qa = nn.Linear(out_features, out_features)
        self.W_qb = nn.Linear(out_features, out_features)
        
        # Key преобразования
        self.W_ka = nn.Linear(out_features, out_features)
        self.W_kb = nn.Linear(out_features, out_features)
        
        # Нормализация и регуляризация
        self.norm_a = nn.LayerNorm(out_features)
        self.norm_b = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Активации
        self.fa2 = nn.Tanh()
        self.fa = nn.SELU()
        
        # Смещение
        self.f_b = nn.Parameter(torch.zeros(out_features))
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for module in [self.W_a, self.W_b, self.W_qa, self.W_qb, self.W_ka, self.W_kb]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def quan(self, x):
        """Квантовая функция активации с защитой от численной нестабильности"""
        # Добавляем ограничение на входные значения для стабильности
        x = torch.clamp(x, min=-5, max=5)
        return 1 / (torch.exp(-x + 1e-12) - 1)
    
    def forward(self, input):
        # Проверка размерности входа
        if input.dim() != 2:
            raise ValueError(f"Expected 2D input, got {input.dim()}D")
            
        # Основные преобразования с нормализацией
        A = self.norm_a(self.W_a(input))  # (batch_size, in_features) -> (batch_size, out_features)
        B = self.norm_b(self.W_b(input))  # (batch_size, in_features) -> (batch_size, out_features)

        A = torch.clamp(A, min=-100, max=100)
        B = torch.clamp(B, min=-100, max=100)
        
        # Квантовые преобразования с dropout
        qa = self.dropout(self.quan(self.W_qa(A)))  # (batch_size, out_features)
        qb = self.dropout(self.quan(self.W_qb(B)))  # (batch_size, out_features)
        
        # Key преобразования
        ka = self.dropout(self.fa2(self.W_ka(A)))  # (batch_size, out_features)
        kb = self.dropout(self.fa2(self.W_kb(B)))  # (batch_size, out_features)
        
        # Финальное вычисление с масштабированием
        quantum_interaction = self.scale * (A * qb * ka - B * qa * kb)
        quer = B + 0.5 * quantum_interaction + self.f_b
        
        return quer
    
    def get_quantum_states(self, input):
        """Метод для анализа квантовых состояний"""
        with torch.no_grad():
            A = self.W_a(input)
            B = self.W_b(input)
            qa = self.quan(self.W_qa(A))
            qb = self.quan(self.W_qb(B))
            return {
                'qa': qa,
                'qb': qb,
                'A': A,
                'B': B
            }
    
class QuantumModel(nn.Module):
    def __init__(self, in_features, out_features, head=1, hid_q=64, hid_l=128, dropout_rate=0.1, device='cpu'):
        super(QuantumModel, self).__init__()
        self.device = device
        self.head = head
        
        # Создаем ModuleList для корректной работы с параметрами
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(in_features, hid_q, device=device) for _ in range(head)
        ])
        
        # Вычисляем размер после конкатенации
        self.concat_size = hid_q * head
        
        # Слои после конкатенации
        self.norm = nn.LayerNorm(self.concat_size).to(device)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Линейные преобразования
        self.fc1 = nn.Linear(self.concat_size, hid_l)
        self.fc2 = nn.Linear(hid_l, out_features)
        
        # Активации
        self.activation = nn.GELU()  # Можно использовать GELU вместо ReLU
        
    def forward(self, input):
        # Получаем выходы от всех квантовых слоев
        q_outputs = [quantum_layer(input) for quantum_layer in self.quantum_layers]
        
        # Конкатенация по последней размерности
        q_cat = torch.cat(q_outputs, dim=-1)
        
        # Нормализация и dropout
        x = self.norm(q_cat)
        x = self.dropout(x)
        
        # Линейные преобразования с активацией
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
    
    def get_attention_weights(self, input):
        """Метод для получения весов внимания от каждой головы"""
        attention_weights = []
        for quantum_layer in self.quantum_layers:
            if hasattr(quantum_layer, 'get_attention_weights'):
                attention_weights.append(quantum_layer.get_attention_weights(input))
        return attention_weights


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
    def __init__(self, model: nn.Module, learning_rate: float=0.001, device: torch.device | str ='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = AdaptiveLoss()  # Используем MSELoss + Hyber 
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
            metrics = self.criterion(y_pred, y)
            
        
        self.model.train()
        return metrics

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
            test_metrics = self.evaluate(X_t, y_t, loss_tube)
            history['test_mape'].append(test_metrics['mape'].item())
            history['test_tube'].append(test_metrics['tube'].item())

            if (test_metrics['tube'].item() +  train_metrics['mape'])/2 < best_test_mape:
                best_test_mape = (test_metrics['tube'].item() +  train_metrics['mape'])/2
                best_model_weights = self.model.state_dict().copy()
            
            # Вывод прогресса
            if (epoch + 1) % 40 == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_metrics["tube"]:.6f}'
                )
        torch.save(best_model_weights, 'best_model_weights.pth')
        return history
