import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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
            if (epoch + 1) % 1 == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}\n'
                    f'Test - MAPE: {test_metrics["mape"]:.6f}, '
                    f'Tube: {test_tube:.6f}'
                )
        
        return history




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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        X = torch.rand(1000,5).cuda()
        y = (X.sum(dim = 1)**2).cuda()
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        dataset = {'train_input':X_train, 'test_input':X_test, 'train_label':y_train, 'test_label':y_test}
        # Создаем данные
        model = KAN(width=[3,2,3,2,3], grid=7, k=5, noise_scale=0.3, seed=2,device = device)

        trainer = KANTrainer(
            model=model,
            learning_rate=0.001,
            device= device
        )

        history = trainer.fit(
            X=X_train,
            y=y_train,
            X_t=X_test,
            y_t=y_test,
            batch_size=64,
            epochs=100,
            loss_tube=5
        )

        print(history['mape'])
        print(history['tube'])