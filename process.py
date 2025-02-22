import torch
from typing import Optional

class ESProcessing:
    """
    Класс для обработки потока событий с использованием модели.

    :param model: Модель для обработки данных.
    :param path_weight: Путь к файлу с весами модели.
    :param size_window: Размер окна данных (количество временных шагов).
    :param input_size: Размерность входных данных (количество признаков).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        path_weight: Optional[str] = None,
        size_window: Optional[int] = None,
        input_size: Optional[int] = None
    ):
        self.model = model
        self.path_weight = path_weight
        self.size_window = size_window
        self.input_size = input_size

        # Инициализация reference_data
        if size_window is not None and input_size is not None:
            self.reference_data = torch.zeros((size_window, input_size))
        else:
            self.reference_data = None

        # Загрузка весов, если путь указан
        if path_weight:
            self.init_weight()
            print(f'Successfully loaded weights from {path_weight}')
        else:
            print('Warning: Model weights not loaded. Use .init_weight() to load weights.')

    def init_weight(self, path_weight: Optional[str] = None):
        """
        Загружает веса модели.

        :param path_weight: Путь к файлу с весами. Если не указан, используется self.path_weight.
        """
        load_path = path_weight if path_weight else self.path_weight
        if load_path:
            try:
                self.model.load_state_dict(torch.load(load_path))
                self.model.eval()  # Переводим модель в режим inference
            except FileNotFoundError:
                raise FileNotFoundError(f"Weights file not found at {load_path}")
        else:
            raise ValueError("No path to weights provided.")

    def cat_data(self, cur_data: torch.Tensor):
        """
        Добавляет новые данные к существующим с проверкой на совместимость.

        :param cur_data: Новые данные (тензор).
        """
        if self.reference_data is None:
            raise ValueError("Reference data not initialized. Set size_window and input_size in __init__.")

        # Проверка на совместимость размерностей
        if cur_data.dim() != self.reference_data.dim():
            raise ValueError(f"Dimensionality mismatch: cur_data has {cur_data.dim()} dimensions, "
                             f"reference_data has {self.reference_data.dim()}.")

        for i in range(cur_data.dim()):
            if i != 0 and cur_data.size(i) != self.reference_data.size(i):
                raise ValueError(f"Size mismatch along axis {i}: cur_data has size {cur_data.size(i)}, "
                                 f"reference_data has size {self.reference_data.size(i)}.")

        # Конкатенация данных вдоль оси 0 (временной оси)
        self.reference_data = torch.cat([self.reference_data, cur_data], dim=0)

        # Обрезка данных до размера окна
        if self.size_window:
            self.reference_data = self.reference_data[-self.size_window:, :]

    def event(self, cur_data: torch.Tensor) -> torch.Tensor:
        """
        Обрабатывает текущий поток данных.

        :param cur_data: Новые данные (тензор).
        :return: Выход модели.
        """
        self.cat_data(cur_data)

        # Добавляем batch dimension, если необходимо
        if self.reference_data.dim() == 2:
            input_data = self.reference_data.unsqueeze(0)  # Добавляем batch dimension
        else:
            input_data = self.reference_data

        # Прогон данных через модель
        with torch.no_grad():  # Отключаем вычисление градиентов
            output = self.model(input_data)

        return output
    
    def event_online(self):
        """
        Принимает данные в реальном времени, обрабатывает их и выводит результат на экран.
        Результат выводится как последовательность чисел.
        """
        print("Онлайн обработка данных. Введите данные (числа, разделённые пробелами):")
        
        while True:
            try:
                # Ввод данных
                user_input = input("Введите данные: ")
                
                # Преобразование ввода в список чисел
                cur_input = list(map(float, user_input.split()))
                
                # Проверка на соответствие размерности входных данных
                if len(cur_input) != self.input_size:
                    print(f"Ошибка: ожидается {self.input_size} значений, получено {len(cur_input)}.")
                    continue
                
                # Преобразование в тензор
                cur_data = torch.tensor(cur_input).unsqueeze(0)  # Добавляем batch dimension
                
                # Обработка данных
                res = self.event(cur_data)
                
                # Преобразование результата в последовательность чисел
                res_list = res.squeeze().tolist()  # Убираем batch dimension и преобразуем в список
                if isinstance(res_list, float):  # Если результат — одно число
                    res_list = [res_list]
                
                # Вывод результата
                print("Результат обработки:", ", ".join(map(str, res_list)))
            
            except ValueError:
                print("Ошибка: введите числа, разделённые пробелами.")
            except KeyboardInterrupt:
                print("\nОнлайн обработка завершена.")
                break