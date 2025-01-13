import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


# Кастомная LSTM-ячейка с KANLayer вместо tanh
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,device = 'cpu'):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sqrt_h = np.sqrt(hidden_size+input_size)
        # Определим линейные слои для вычислений LSTM
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        
        #self.KAN = KAN(width=[hidden_size,3,hidden_size], grid=3, k=3, seed=0,device=device)
        #self.KAN = KAN(width=[hidden_size,hidden_size], grid=5, k=5, seed=0,device=device)
        # Инициализируем KANLayer

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # Объединяем входные данные x и предыдущее скрытое состояние
        combined = torch.cat((x, h_prev), dim=1)
        # print('2.1')
        # print(combined.shape)
        f_t = torch.sigmoid(self.W_f(combined)/self.sqrt_h)
        #print(f_t.shape)
        i_t = torch.sigmoid(self.W_i(combined)/self.sqrt_h)
        #print(i_t.shape)
        o_t = torch.sigmoid(self.W_o(combined)/self.sqrt_h)
        
        # Вместо tanh используем KANLayer для вычисления кандидата скрытого состояния
        #c_tilde = self.KAN(self.W_c(combined))
        c_tilde = torch.tanh(self.W_c(combined)/self.sqrt_h)

        c_t = f_t * c_prev + i_t * c_tilde
        #print('2.3')
        h_t = o_t * torch.tanh(c_t)  # Если нужно, здесь тоже можно заменить tanh на KANLayer

        return h_t, (h_t, c_t)
    
# Класс RNN с использованием кастомной LSTM-ячейки
class VanilaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(VanilaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Создаем нужное количество слоев с кастомной LSTM-ячейкой
        self.layers = nn.ModuleList([CustomLSTMCell(input_size, hidden_size, device) if i == 0 
                                     else CustomLSTMCell(hidden_size, hidden_size, device) 
                                     for i in range(num_layers)])
        
        # Выходной линейный слой для преобразования скрытого состояния в выход
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Начальные скрытые состояния и состояния ячеек для каждого слоя
        h, c = [torch.zeros(x.size(0), self.hidden_size).to(self.device) for _ in range(self.num_layers)], \
               [torch.zeros(x.size(0), self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        #print('1')
        # Проходим по временным шагам входных данных
        for t in range(x.size(1)):
            input_t = x[:, t, :]  # Вход на текущем временном шаге
            #print('2')
            # Проходим через каждый слой
            for i, layer in enumerate(self.layers):
                h[i], (h[i], c[i]) = layer(input_t, (h[i], c[i]))
                input_t = h[i]  # Передаем выход текущего слоя на вход следующему

            #print('3')
        #print('4')
        # Преобразуем скрытое состояние последнего слоя в итоговый выход
        output = self.fc(h[-1])
        #print('5')
        return output

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
            
            
            #print('3')
        
        self.model.train()
        return metrics

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
            if (epoch + 1) % self.inf_per_epoch == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_metrics["tube"]:.6f}'
                )
        torch.save(best_model_weights, 'best_model_weights.pth')
        return history