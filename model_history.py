import json
import os

class ModelHistory:
    def __init__(self,history: dict):
        self.history = history  # История текущей модели

    def save_history_json(self, filename: str, model_name: str = 'unknown'):
        """
        Сохраняет историю модели в JSON-файл.
        Если файл уже существует, данные добавляются в него.
        """
        data = {}
        
        # Если файл существует, загружаем его содержимое
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        
        # Добавляем или обновляем историю для текущей модели
        data[model_name] = self.history
        
        # Сохраняем обновлённые данные в файл
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_history_json(filename: str):
        """
        Загружает историю из JSON-файла.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден.")
        
        with open(filename, 'r') as f:
            return json.load(f)