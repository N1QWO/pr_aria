import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def mape(y_pred,y):
    loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
    return torch.mean(loss_rd)

def tube(y_pred,y):
    loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
    rt = (loss_rd < 0.05).sum() / loss_rd.numel()
    return rt

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

    def __init__(self, model: nn.Module, learning_rate: float = 0.001, device: torch.device | str = 'cpu'):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate


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