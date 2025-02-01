import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from kan.KANLayer import *
import torch.nn.functional as F

def mape(y_pred,y):
    loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
    return torch.mean(loss_rd)

def tube(y_pred,y):
    loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
    rt = (loss_rd < 0.05).sum() / loss_rd.numel()
    return rt
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
class KANPERCE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hid: int,
                 dropout: float = 0.1,
                 num: int = 5,
                 k: int = 3,
                 noise_scale: float = 0.5,
                 scale_base_mu: float = 0,
                 scale_base_sigma: float = 1,
                 scale_sp: float = 1,
                 base_fun=torch.nn.functional.silu,  # Используем функцию, а не модуль
                 grid_eps: float = 0.02,
                 grid_range=[-1, 1],
                 device: torch.device | str = 'cpu'):
        super(KANPERCE, self).__init__()

        # Убедимся, что device - это torch.device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Преобразуем grid_range в тензор
       # grid_range = torch.tensor(grid_range, device=self.device)

        # Определяем слои
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hid, device=self.device)
        self.kanl1 = KANLayer(
            in_dim=hid,
            out_dim=hid ,
            k=k,
            num=num,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            scale_sp=scale_sp,
            base_fun=base_fun,
            grid_eps=grid_eps,
            grid_range=grid_range,
            device=self.device
        )
        # self.kanl2 = KANLayer(
        #     in_dim=hid * 2,
        #     out_dim=hid,
        #     k=k,
        #     num=num,
        #     noise_scale=noise_scale,
        #     scale_base_mu=scale_base_mu,
        #     scale_base_sigma=scale_base_sigma,
        #     scale_sp=scale_sp,
        #     base_fun=base_fun,
        #     grid_eps=grid_eps,
        #     grid_range=grid_range,
        #     device=self.device
        # )
        self.kanl2 = KANLayer(
            in_dim=hid,
            out_dim=out_dim,
            num=num,
            k=k,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            scale_sp=scale_sp,
            base_fun=base_fun,
            grid_eps=grid_eps,
            grid_range=grid_range,
            device=self.device
        )

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, _, _, _ = self.kanl1(x)
        x, _, _, _ = self.kanl2(x)

        return x
class KANTrainer:
    """
    Класс для обучения квантовой нейронной сети.

    Этот класс управляет процессом обучения модели, включая настройку
    оптимизатора, функции потерь и планировщика обучения.

    Параметры:
    - model (nn.Module): Модель квантовой нейронной сети.
    - learning_rate (float): Скорость обучения (по умолчанию 0.001).
    - device (str): Устройство для выполнения (CPU или GPU, по умолчанию 'cpu').
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.001,fit2:bool = False,inf_per_epoch:int = 10,device: torch.device | str = 'cpu'):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        if fit2:
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

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, loss_tube: float = 5) -> tuple:
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
            
            loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)  # Вычисление относительной потери
            per_loss_rd = (loss_rd < 0.01 * loss_tube).sum().item() / (loss_rd.numel())  # Процент потерь ниже порога
        
        self.model.train()  # Возврат модели в режим обучения
        return metrics, per_loss_rd


    def create_scheduler(self):
        # Пример создания планировщика
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

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
    
    def fit2(self, X: torch.Tensor, y: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int = 512, epochs:int = 100, loss_tube: int = 5) -> dict:
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
        
        # history = {
        #     'train_total_loss': [],
        #     'train_main_loss': [],
        #     'train_quantum_loss': [],
        #     'train_mape': [],
        #     'train_alpha': [],
        #     'test_mape': [],
        #     'test_tube': []
        # }
        dataset = {'train_input':X, 'test_input':X_t, 'train_label':y, 'test_label':y_t}
        history = self.model.fit(dataset, opt='LBFGS', steps=epochs, lamb=self.learning_rate,metrics = [mape,tube] ,batch = batch_size)
        
        # for epoch in range(epochs):
        #     # Обучение на эпохе
        #     train_metrics = self.train_epoch(X, y, batch_size)
            
        #     # Шаг планировщика
        #     scheduler.step()
            
        #     # Сохранение метрик обучения
        #     history['train_total_loss'].append(train_metrics['total_loss'])
        #     history['train_main_loss'].append(train_metrics['main_loss'])
        #     history['train_quantum_loss'].append(train_metrics['quantum_loss'])
        #     history['train_mape'].append(train_metrics['mape'])
        #     history['train_alpha'].append(train_metrics['alpha'])
            
        #     # Оценка на тестовых данных
        #     test_metrics, test_tube = self.evaluate(X_t, y_t, loss_tube)
        #     history['test_mape'].append(test_metrics['mape'].item())
        #     history['test_tube'].append(test_tube)
            
        #     # Вывод прогресса
        #     if (epoch + 1) % 100 == 0 or epoch == 9:
        #         print(
        #             f'Epoch {epoch + 1}\n'
        #             f'Train - Total: {train_metrics["total_loss"]:.6f}, '
        #             f'Main: {train_metrics["main_loss"]:.6f}, '
        #             f'Quantum: {train_metrics["quantum_loss"]:.6f}, '
        #             f'MAPE: {train_metrics["mape"]:.6f}, '
        #             f'Alpha: {train_metrics["alpha"]:.6f}\n'
        #             f'Test - MAPE: {test_metrics["mape"]:.6f}, '
        #             f'Tube: {test_tube:.6f}'
        #         )
        
        return history  # Возврат истории метрик
    



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