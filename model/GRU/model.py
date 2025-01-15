import torch
import torch.nn as nn
import torch.optim as optim

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, device: torch.device | str ='cpu'):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        # Объявляем веса для обновления и сброса
        self.Wz = nn.Linear(input_size, hidden_size, device=self.device)  # Веса для обновления
        self.Uz = nn.Linear(hidden_size, hidden_size, device=self.device)  # Веса для обновления скрытого состояния
        self.Wr = nn.Linear(input_size, hidden_size, device=self.device)  # Веса для сброса
        self.Ur = nn.Linear(hidden_size, hidden_size, device=self.device)     # Веса для сброса скрытого состояния
        self.Wh = nn.Linear(input_size, hidden_size, device=self.device)  # Веса для нового состояния
        self.Uh = nn.Linear(hidden_size, hidden_size, device=self.device)  # Веса для нового состояния скрытого состояния

    def forward(self, x, h_prev):
        # Обновление
        z_t = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        # Сброс
        r_t = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))
        # Новое состояние
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r_t * h_prev))
        # Обновленное состояние
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, device: torch.device | str ='cpu'):
        super(GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList([GRUCell(input_size if i == 0 else hidden_size, hidden_size, device=self.device) for i in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)  # Инициализация скрытого состояния

        for t in range(seq_len):
            for layer in range(self.num_layers):
                h = self.gru_cells[layer](x[:, t, :], h)  # Применяем GRU ячейку

        return h  # Возвращаем последнее скрытое состояние

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device: torch.device | str ='cpu'):
        super(GRURegressor, self).__init__()
        self.device = device
        self.gru = GRU(input_size, hidden_size, num_layers, device=self.device)
        self.fc = nn.Linear(hidden_size, output_size, device=self.device)  # Полносвязный слой для регрессии

    def forward(self, x):
        h = self.gru(x)  # Получаем последнее скрытое состояние
        out = self.fc(h)  # Применяем полносвязный слой
        return out

class RnnAdaptiveLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(RnnAdaptiveLoss, self).__init__()
        self.loss = nn.HuberLoss(delta=delta)
        self.loss_tube = None
        #self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
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

class RNNTrainer:
    def __init__(self, model, learning_rate=0.001, inf_per_epoch=100,device='cpu',):
        self.model = model.to(device)
        self.device = device
        self.criterion = RnnAdaptiveLoss()  # Используем MSELoss + Hyber 
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.inf_per_epoch = 100
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

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        dataset_size = X.shape[0]
        indices = torch.randperm(dataset_size)
        X_shuffled = X[indices].to(self.device)
        y_shuffled = y[indices].to(self.device)
        
        epoch_metrics = {
            'main_loss': 0.0,
            'mape': 0.0,
            'tube': 0.0,
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
            metrics = self.criterion(y_pred, y)
            
        self.model.train()
        return metrics
    def add_history(self, train_metrics: dict, test_metrics: dict):
        for key in train_metrics:
            self.history['train_' + key].append(train_metrics[key])
        for key in test_metrics:
            self.history['test_' + key].append(test_metrics[key].item())
            
    def fit(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int = 512, epochs:int = 100, loss_tube: int = 5):
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
            
            # Сохранение метрик обучения
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

# Пример использования
if __name__ == "__main__":
    input_size = 10  # Размер входных данных
    hidden_size = 20  # Размер скрытого состояния
    output_size = 1  # Для регрессии
    num_layers = 2  # Количество слоев GRU
    seq_len = 5  # Длина последовательности
    batch_size = 3  # Размер батча

    model = GRURegressor(input_size, hidden_size, output_size, num_layers)
    input_data = torch.randn(batch_size, seq_len, input_size)  # Генерация случайных входных данных
    output = model(input_data)  # Прямой проход через модель
    print("Output shape:", output.shape)  # Ожидаемый вывод: (batch_size, output_size)