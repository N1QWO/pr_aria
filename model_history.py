import os
import json
import uuid
from typing import Dict, Any, Optional, Tuple


class ModelHistory:
    def __init__(self, history: Optional[Dict[str, Any]] = None):
        """
        Инициализирует объект ModelHistory.

        Аргументы:
            history (Dict[str, Any], опционально): История модели в виде словаря.
        """
        self.history = history if history is not None else {}
        self.key = None 
    def save_history_json(self, filename: str, model_name: str = 'unknown') -> None:
        """
        Сохраняет историю модели в JSON-файл.
        Если файл уже существует, данные добавляются в него.

        Аргументы:
            filename (str): Путь к файлу для сохранения.
            model_name (str, опционально): Имя модели. По умолчанию 'unknown'.
        """
        data = {}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

        data[model_name] = self.history

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_history_json(filename: str) -> Dict[str, Any]:
        """
        Загружает историю из JSON-файла.

        Аргументы:
            filename (str): Путь к файлу для загрузки.

        Возвращает:
            Dict[str, Any]: Загруженные данные.

        Исключения:
            FileNotFoundError: Если файл не существует.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден.")

        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_best_cur_result(self) -> Tuple[Dict[str, Any], int]:
        """
        Находит индекс максимального среднего значения между train, test и tube в истории модели.

        Возвращает:
            Tuple[Dict[str, Any], int]: Лучшие результаты и их индекс.

        Исключения:
            ValueError: Если история пуста или отсутствуют необходимые ключи.
        """
        if not self.history or 'train_tube' not in self.history or 'test_tube' not in self.history:
            raise ValueError("История модели пуста или отсутствуют необходимые ключи.")

        max_avg = float('-inf')
        best_index = -1

        for index, (train, test) in enumerate(zip(self.history['train_tube'], self.history['test_tube'])):
            avg = (train + test) / 2
            if avg > max_avg:
                max_avg = avg
                best_index = index
                
        best_result = {key: values[best_index] for key, values in self.history.items()}
        return best_result, best_index

    def add_result(
        self,
        filename: str,
        model_name: str,
        result: Optional[Dict[str, Any]] | str = 'last',
        message: str = '',
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Добавляет результаты в историю модели.

        Аргументы:
            filename (str): Путь к файлу для сохранения результатов.
            model_name (str): Имя модели.
            result (Dict[str, Any], опционально): Результаты для добавления. По умолчанию None.
            message (str, опционально): Дополнительное сообщение. По умолчанию ''.
            params (Dict[str, Any], опционально): Дополнительные параметры. По умолчанию None.

        Возвращает:
            str: Уникальный ключ, под которым сохранены результаты.

        Исключения:
            ValueError: Если результат не указан и история пуста.
        """

        data = {}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

        index = -1

        if result == 'last':
            result = {key: value[index] for key, value in self.history.items()}    
            index = len(self.history['train_tube']) - 1

        if result == 'best':
            result, index = self.find_best_cur_result()


        add_data = {
            'result': result,
            'message': message,
            'params': params if params is not None else {},
            'epoch': index+1  # Индекс последней эпохи
        }

        unique_key = str(uuid.uuid4())
        
        if model_name not in data:
            data[model_name] = {}

        if unique_key in data[model_name].keys():
            raise ValueError("Коллизия ключей")
        data[model_name][unique_key] = add_data

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.key = unique_key
        return self.key

    def get_result_key(self, filename: str, model_name: str, key: str) -> Any:
        """
        Получает результаты обучения модели по заданному ключу.

        Аргументы:
            filename (str): Путь к JSON-файлу с данными.
            model_name (str): Название модели.
            key (str): Ключ, под которым сохранены результаты.

        Возвращает:
            Any: Результаты обучения.

        Исключения:
            ValueError: Если модель или ключ не найдены.
        """
        hiss = self.load_history_json(filename)

        if model_name not in hiss:
            raise ValueError(f"Модель '{model_name}' не найдена в данных.")

        if key not in hiss[model_name]:
            raise ValueError(f"Ключ '{key}' не найден в данных для модели '{model_name}'.")

        return hiss[model_name][key]

    def get_all_result(self, filename: str, model_name: str) -> Dict[str, Any]:
        """
        Получает все результаты обучения для указанной модели.

        Аргументы:
            filename (str): Путь к JSON-файлу с данными.
            model_name (str): Название модели.

        Возвращает:
            Dict[str, Any]: Все результаты обучения.

        Исключения:
            ValueError: Если модель не найдена.
        """
        hiss = self.load_history_json(filename)

        if model_name not in hiss:
            raise ValueError(f"Модель '{model_name}' не найдена в данных.")

        return hiss[model_name]
    

if __name__ == '__main__':
    import os
    dir = os.path.dirname(__file__)
    history = ModelHistory.load_history_json(os.path.join(dir,'loss_history/all_loss.json'))
    a = history['QUANT']
    MH = ModelHistory(history['QUANT'])

    key = MH.add_result(
        filename = os.path.join(dir,'resualt/main.json'),
        model_name = 'QUANT',
        result = 'best',
        message = 'привет',
        params = {'test':1}
        )
    



