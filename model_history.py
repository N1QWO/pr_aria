import json
import numpy as np
import os
import uuid


class ModelHistory:
    def __init__(self,history: dict | None = None):
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
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_history_json(filename: str):
        """
        Загружает историю из JSON-файла.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден.")
        
        with open(filename, 'r') as f:
            return json.load(f)



    def find_best_cur_result(self):
        """
        Находит индекс максимального среднего значения между test, train и tube в истории модели.
        """

        max_avg = float('-inf')
        best_index = -1
        # Проходим по истории и вычисляем средние значения
        for index in range(len(self.history['train_tube'])):
            # Предполагаем, что entry содержит 'train', 'test' и 'tube'
            train = self.history['train_tube'][index]
            test = self.history['test_tube'][index]

            # Вычисляем среднее значение
            avg = (train + test) / 2

            # Проверяем, является ли текущее среднее максимальным
            if avg > max_avg:
                max_avg = avg
                best_index = index


        data = {}
        for key in self.history:
            data[key] = self.history[key][best_index]
        return data,best_index

    def add_result(self, filename: str, model_name: str, result: dict | str = 'last', message: str = '', params: dict = {}):
        """
        Добавляет результаты в историю модели.

        Параметры:
        ----------
        filename : str
            Путь к файлу для сохранения результатов. Содержимое будет загружено, если файл существует.
        
        model_name : str
            Имя модели, для которой добавляются результаты.

        result : dict | str, по умолчанию 'last'
            Результаты для добавления. 
            'last' использует последние результаты из history, 
            'best' — лучшие результаты из find_best_cur_result.

        message : str, по умолчанию ''
            Дополнительное сообщение, связанное с результатами.

        params : dict, по умолчанию {}
            Дополнительные параметры, связанные с результатами.

        Возвращает:
        ----------
        None
            Метод обновляет файл с результатами.

        Исключения:
        -----------
        ValueError
            Если указано 'best', но в истории нет результатов.
        """
        data = {}
        
        # Если файл существует, загружаем его содержимое
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        index = -1
        if result == 'last':
            result = self.history[-1]    
            index = len(self.history['train_tube']) - 1

        if result == 'best':
            result, index = self.find_best_cur_result()   

        add_data = {
            'result': result,
            'message': message,
            'params': params,
            'epoch': index + 1
        }
        unique_key = str(uuid.uuid4())
        # Проверяем, существует ли ключ model_name, и инициализируем его, если нет
        if model_name not in data:
            data[model_name] = {}

        # Добавляем или обновляем историю для текущей модели
        data[model_name][unique_key] = add_data
        
        # Сохраняем обновлённые данные в файл
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return unique_key
    

    # def get_result_key(self,filename: str,model_name: str,key: str):
    #     hiss = self.load_history_json(filename)
    #     return hiss[model_name][key]
    
    # def get_all_result(self,filename: str,model_name: str):
    #     hiss = self.load_history_json(filename)
    #     return hiss[model_name]
    

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
    



