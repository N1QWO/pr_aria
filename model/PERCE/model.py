import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class three_layer(nn.Module):
    
    def __init__(self, input_size: int, output_size: int,hid: int = 64,hid2: int = 32,hid3: int =64,device: torch.device | str='cpu'):
        
        super(three_layer, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(input_size, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))

        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        
        self.fc14 = nn.Linear(hid3, output_size) 
        self.fc14_b = nn.Parameter(torch.zeros(output_size))
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input: torch.tensor) -> torch.tensor:
        f1 = F.relu(self.fc11(input) + self.fc11_b)
        f2 = F.relu(self.fc12(f1) + self.fc12_b)

        f3 = F.relu(self.fc13(f2) + self.fc13_b)

        f4 = self.fc14(f3) + self.fc14_b
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f4

class up_three_layer(nn.Module):
    def __init__(self, input_size:int, output_size:int, hid:int=64, hid2:int=32, hid3:int=64, device: torch.device | str='cpu', dropout_rate: float =0.2):
        super(up_three_layer, self).__init__()
        self.device = device

        # Определяем слои
        self.fa = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout_rate)  # Один объект Dropout для всех слоев

        # Первый блок
        self.fc11 = nn.Linear(input_size, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))
        self.layer_norm1 = nn.LayerNorm(hid).to(self.device)

        # Второй блок
        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))
        self.layer_norm3 = nn.LayerNorm(hid2).to(self.device)

        # Третий блок
        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))

        # Выходной слой
        self.fc14 = nn.Linear(hid3, output_size)
        self.fc14_b = nn.Parameter(torch.zeros(output_size))

    def forward(self, input: torch.tensor) -> torch.tensor:
        # Первый блок
        f1 = self.fa(self.fc11(input) + self.fc11_b)
        f1 = self.layer_norm1(f1)
        f1 = self.dropout(f1)  

        # Второй блок
        f2 = self.fa(self.fc12(f1) + self.fc12_b)
        f2 = self.layer_norm3(f2)
        f2 = self.dropout(f2)  

        # Третий блок
        f3 = self.fa(self.fc13(f2) + self.fc13_b)
        f3 = self.dropout(f3)  

        f4 = self.fc14(f3) + self.fc14_b

        return f4
    
class up_three_layer_multi(nn.Module):
    
    def __init__(self, input_size: int, output_size: int,hid: int = 64,hid2: int = 32,hid3: int=64,hid4: int= 32,device: torch.device | str ='cpu'):
        
        super(up_three_layer_multi, self).__init__()
        self.device = device

        self.fa = nn.LeakyReLU(negative_slope=0.1)

        self.fc11 = nn.Linear(input_size, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.05)
        # self.dp3 = nn.Dropout(0.05)

        
        self.fa2  = nn.Tanh()
        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))

        self.layer_norm2 = nn.LayerNorm(hid2).to(self.device)

        
        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        

        self.fc14 = nn.Linear(hid3, hid4) 
        self.fc14_b = nn.Parameter(torch.zeros(hid4))

        self.layer_norm4 = nn.LayerNorm(hid4).to(self.device)
        self.fc15 = nn.Linear(hid4, output_size) 
        self.fc15_b = nn.Parameter(torch.zeros(output_size))
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input: torch.tensor) -> torch.tensor:
        f1 =  self.fa(self.fc11(input) + self.fc11_b)
        #f1 = self.layer_norm1(f1)
        #dp1 = self.dp1(f1)

        f2 = self.fa(self.fc12(f1) + self.fc12_b)
        #dp2 = self.dp2(f2) 
        
        f3 = self.layer_norm2(f2)

        f3 =  self.fa2(self.fc13(f3) + self.fc13_b)
        #dp3 = self.dp3(f3)

        f4 = self.fa(self.fc14(f3) + self.fc14_b)
        f4 = self.layer_norm4(f4)

        f5 = self.fc15(f4) + self.fc15_b
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f5

class ImprovedRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size,device: torch.device | str ='cpu'):
        super(ImprovedRegressionNN, self).__init__()
        self.device = device

        # Полносвязные слои
        self.fc1 = nn.Linear(input_size, hidden_size1).to(self.device)
        self.bn1 = nn.BatchNorm1d(hidden_size1).to(self.device)  # Batch Normalization
        self.fc2 = nn.Linear(hidden_size1, hidden_size2).to(self.device)
        self.bn2 = nn.BatchNorm1d(hidden_size2).to(self.device)  # Batch Normalization
        self.fc3 = nn.Linear(hidden_size2, hidden_size3).to(self.device)
        self.bn3 = nn.BatchNorm1d(hidden_size3).to(self.device)  # Batch Normalization
        self.fc4 = nn.Linear(hidden_size3, output_size).to(self.device)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Первый скрытый слой
        x = F.leaky_relu(self.bn1(self.fc1(x)))  # LeakyReLU + BatchNorm
        x = self.dropout(x)
        
        # Второй скрытый слой
        x = F.leaky_relu(self.bn2(self.fc2(x)))  # LeakyReLU + BatchNorm
        x = self.dropout(x)
        
        # Третий скрытый слой
        x = F.leaky_relu(self.bn3(self.fc3(x)))  # LeakyReLU + BatchNorm
        x = self.dropout(x)
        
        # Выходной слой (без активации, так как это регрессия)
        x = self.fc4(x)
        
        return x

class AdaptiveLoss(nn.Module):
    def __init__(self, delta: float = 0.01):
        super(AdaptiveLoss, self).__init__()
        self.loss = nn.HuberLoss(delta=delta)
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
            if (epoch + 1) % 40 == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_tube:.6f}'
                )
        
        return history
