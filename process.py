import torch




class ESProcessing:
    #event stream processing
    def __init__(self,model,path_weight: str = None):
        self.model = model
        self.path_weight = path_weight
        if path_weight: 
            self.init_weight() 
            print(f'success init weight, path_weight: {path_weight}')
    def init_weight(self,path_weight: str = None):

        if path_weight:   
            self.model.load_state_dict(torch.load(path_weight))
        else:
            self.model.load_state_dict(torch.load(self.path_weight))
        
    #метод полной обработки событий онлайн потока информации
    def event(self,):
        
        
        