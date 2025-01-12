import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Bl(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,device='cpu'):
        
        super(Bl, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))
        self.fc11_activation = nn.LeakyReLU(negative_slope=0.1)

        self.cla = nn.Linear(in_, hid)
        self.cla_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(in_, hid)
        self.fc12_b = nn.Parameter(torch.zeros(hid))

        self.fc3 = nn.Linear(hid, hid2)
        self.fc3_b = nn.Parameter(torch.zeros(hid2))

        self.fc4 = nn.Linear(hid2, out_)
        self.layer_norm = nn.LayerNorm(hid2).to(self.device)
        self.dropout = nn.Dropout(p=0.35)

        self.mxp = nn.MaxPool1d(kernel_size=int(hid//hid2))
        
    def short_cla_module(self, input):
        short = self.fc12(input) + self.fc12_b #128
        short2 = self.fc3(short) + self.fc3_b#32
        cust = torch.tanh(short2-torch.mean(short2))*short2 #32
        
        cust = self.layer_norm(cust)
        
        return cust
        
    
    def long_cla_module(self, input):
        long = self.fc11_activation(self.fc11(input) + self.fc11_b)
        long = self.dropout(long)
        
        cla = torch.sigmoid(self.cla(input) + self.cla_b)

        return long * cla
    

    def forward(self, input):
        al = self.long_cla_module(input)

        cust = self.short_cla_module(input)
        
        cust = self.dropout(cust)

        al = al.unsqueeze(1)  
        maxp = self.mxp(al).squeeze(1) 

        re = self.fc4(cust * maxp)
        return re

    def fit(self, X, y, batch_size, epochs, learning_rate=0.001, device='cpu'):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=50,
            mode='triangular2'
        )
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = criterion(predictions, y_batch)
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]:.10f}, Loss_mape: {loss_mape[-1]:.10f}')
        
        return loss_history, loss_mape

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Размерность векторного представления входных данных (embedding dimension).
            num_heads: Количество голов в механизме внимания.
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        # Линейные слои для создания Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Линейный слой для объединения голов
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, seq_len, embed_dim).
        Returns:
            Тензор формы (batch_size, seq_len, embed_dim) с обработанным вниманием.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Убедимся, что embed_dim соответствует инициализации
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Вычисление Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # Разделение на головы
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Вычисление оценок внимания (Scaled Dot-Product Attention)
        energy = torch.einsum("bhqd,bhkd->bhqk", Q, K)  # (batch_size, num_heads, seq_len, seq_len)
        scaling_factor = self.head_dim ** 0.5
        attention = F.softmax(energy / scaling_factor, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Применение внимания к V
        out = torch.einsum("bhqk,bhvd->bhqd", attention, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Объединение голов
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Пропуск через выходной линейный слой
        out = self.fc_out(out)  # (batch_size, seq_len, embed_dim)

        return out

class FeedForwardRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: Размерность входных данных.
            hidden_dim: Размерность скрытого слоя.
            output_dim: Размерность выходного слоя (для регрессии = 1).
        """
        super(FeedForwardRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, input_dim).
        Returns:
            Тензор формы (batch_size, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Baseline(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, hidden_dim,device):
        """
        Объединенный класс, использующий Bl, SelfAttention и FeedForwardRegression.

        Args:
            in_dim: Размер входных данных для Bl.
            out_dim: Размер выходных данных для FeedForwardRegression.
            embed_dim: Размер embedding для SelfAttention.
            num_heads: Количество голов для SelfAttention.
            hidden_dim: Размерность скрытого слоя для FeedForwardRegression.
        """
        super(Baseline, self).__init__()
        self.bl = Bl(in_dim, embed_dim,device=device)
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardRegression(embed_dim, hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, seq_len, in_dim).
        Returns:
            Тензор формы (batch_size, out_dim).
        """
        x_bl = self.bl(x)  # (batch_size, embed_dim)
        x_bl = x_bl.unsqueeze(1)  # Добавляем измерение seq_len
        x_sa = self.self_attention(x_bl)  # (batch_size, seq_len, embed_dim)
        x_sa = x_sa.squeeze(1)  # Убираем измерение seq_len
        output = self.feed_forward(x_sa)  # (batch_size, out_dim)
        return output
    
    def fit(self, X, y, batch_size, epochs, learning_rate=0.001, device='cpu'):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=50,
            mode='triangular2'
        )
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = criterion(predictions, y_batch)
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]:.10f}, Loss_mape: {loss_mape[-1]:.10f}')
        
        return loss_history, loss_mape
    



class THdLin(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,hid3=100,device='cpu'):
        
        super(THdLin, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))


        self.dp1 = nn.Dropout(0.3)

        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        self.bn =nn.LayerNorm(hid3)
        
        self.dp2 = nn.Dropout(0.3)
        self.fc14 = nn.Linear(hid3, out_)
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input):
        f1 = self.fc11(input) + self.fc11_b
        f2 = F.relu(self.fc12(f1) + self.fc12_b)

        dp1 = self.dp1(f2)
        
        f3 = self.fc13(dp1) + self.fc13_b
        bn = self.bn(f3)

        dp2 = self.dp2(bn)
        f4 = self.fc14(dp2)
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f4

    def fit(self, X, y,X_t,y_t, batch_size, epochs, learning_rate=0.001, device='cpu',loss_tube=5):
        self.to(device)
        criterion = nn.HuberLoss(delta=0.01)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5,eps = 1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = torch.sqrt(criterion(predictions, y_batch))
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                self.eval()
                loss_rd = torch.abs(y_t - self.forward(X_t)) / torch.clamp(y_t, min=1e-7)
                per_loss_rd = loss_rd[loss_rd<0.01*loss_tube].shape[0] / loss_rd.shape[0]
                print(f'Epoch {epoch + 1}/{epochs}, Loss_model: {loss_history[-1]:.10f}, Loss_mape_train: {loss_mape[-1]:.10f}, Loss_mape_test: {torch.mean(loss_rd).item():.10f}, tube_loss_mape_test: {per_loss_rd:.10f}')
                self.train()
        
        return loss_history, loss_mape
    



class quntum_l(nn.Module):
    def __init__(self, in_, out_, device='cpu', dropout_rate=0.1):
        super(quntum_l, self).__init__()
        self.device = device
        self.in_ = in_
        self.out_ = out_
        
        # Инициализация весов с правильным масштабированием
        self.scale = 1 / np.sqrt(out_)
        
        # Линейные преобразования для A и B
        self.W_a = nn.Linear(in_, out_)
        self.W_b = nn.Linear(in_, out_)
        
        # Квантовые преобразования
        self.W_qa = nn.Linear(out_, out_)
        self.W_qb = nn.Linear(out_, out_)
        
        # Key преобразования
        self.W_ka = nn.Linear(out_, out_)
        self.W_kb = nn.Linear(out_, out_)
        
        # Нормализация и регуляризация
        self.norm_a = nn.LayerNorm(out_)
        self.norm_b = nn.LayerNorm(out_)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Активации
        self.fa2 = nn.Tanh()
        self.fa = nn.SELU()
        
        # Смещение
        self.f_b = nn.Parameter(torch.zeros(out_))
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for module in [self.W_a, self.W_b, self.W_qa, self.W_qb, self.W_ka, self.W_kb]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def quan(self, x):
        """Квантовая функция активации с защитой от численной нестабильности"""
        # Добавляем ограничение на входные значения для стабильности
        x = torch.clamp(x, min=-5, max=5)
        return 1/(torch.exp(-x + 1e-12)-1)
    
    def forward(self, input):
        # Проверка размерности входа
        if input.dim() != 2:
            raise ValueError(f"Expected 2D input, got {input.dim()}D")
            
        # Основные преобразования с нормализацией
        A = self.norm_a(self.W_a(input))  # (1 n) (n hid) = (1 hid)
        B = self.norm_b(self.W_b(input))  # (1 n) (n hid) = (1 hid)

        A = torch.clamp(A, min=-100, max=100)
        B = torch.clamp(B, min=-100, max=100)
        
        # Квантовые преобразования с dropout
        qa = self.dropout(self.quan(self.W_qa(A)))  # (1 hid) (hid hid) = (1 hid)
        qb = self.dropout(self.quan(self.W_qb(B)))  # (1 hid) (hid hid) = (1 hid)
        
        # Key преобразования
        ka = self.dropout(self.fa2(self.W_ka(A)))  # (1 hid) (hid hid) = (1 hid)
        kb = self.dropout(self.fa2(self.W_kb(B)))  # (1 hid) (hid hid) = (1 hid)
        
        # Финальное вычисление с масштабированием
        quantum_interaction = self.scale * (A * qb * ka - B * qa * kb)
        quer = B + 0.5 * quantum_interaction + self.f_b
        
        return quer
    
    def get_quantum_states(self, input):
        """Метод для анализа квантовых состояний"""
        with torch.no_grad():
            A = self.W_a(input)
            B = self.W_b(input)
            qa = self.quan(self.W_qa(A))
            qb = self.quan(self.W_qb(B))
            return {
                'qa': qa,
                'qb': qb,
                'A': A,
                'B': B
            }
    
class quntum(nn.Module):
    def __init__(self, in_, out_, head=1, hid_q=64, hid_l=128, dropout_rate=0.1, device='cpu'):
        super(quntum, self).__init__()
        self.device = device
        self.head = head
        
        # Создаем ModuleList для корректной работы с параметрами
        self.quans = nn.ModuleList([
            quntum_l(in_, hid_q, device=device) for _ in range(head)
        ])
        
        # Вычисляем размер после конкатенации
        self.concat_size = hid_q * head
        
        # Слои после конкатенации
        self.norm = nn.LayerNorm(self.concat_size).to(device)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Линейные преобразования
        self.fc1 = nn.Linear(self.concat_size, hid_l)
        self.fc2 = nn.Linear(hid_l, out_)
        
        # Активации
        self.activation = nn.GELU()  # Можно использовать GELU вместо ReLU
        
    def forward(self, input):
        # Получаем выходы от всех квантовых слоев
        q_outputs = [quan(input) for quan in self.quans]
        
        # Конкатенация по последней размерности
        q_cat = torch.cat(q_outputs, dim=-1)
        
        # Нормализация и dropout
        x = self.norm(q_cat)
        x = self.dropout(x)
        
        # Линейные преобразования с активацией
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
    
    def get_attention_weights(self, input):
        """Метод для получения весов внимания от каждой головы"""
        attention_weights = []
        for quan in self.quans:
            if hasattr(quan, 'get_attention_weights'):
                attention_weights.append(quan.get_attention_weights(input))
        return attention_weights

    def fit(self, X, y,X_t,y_t, batch_size, epochs, learning_rate=0.001, device='cpu',loss_tube=5):
        self.to(device)
        #criterion = nn.HuberLoss(delta = 0.01)
        criterion = nn.HuberLoss(delta = 0.01)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test = [], [],[],[]
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                #loss = torch.sqrt(criterion(predictions, y_batch))
                loss =criterion(predictions, y_batch)
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))

            self.eval()
            loss_rd = torch.abs(y_t - self.forward(X_t)) / torch.clamp(y_t, min=1e-7)
            per_loss_rd = loss_rd[loss_rd<0.01*loss_tube].shape[0] / (loss_rd.shape[0] * loss_rd.shape[1] ) 

            Loss_mape_test.append(torch.mean(loss_rd).item())
            tube_loss_mape_test.append(per_loss_rd)
            self.train()
            
            if (epoch + 1) % 100 == 0 or epoch == 9:
                print(f'Epoch {epoch + 1}, huber_train: {loss_history[-1]:.6f}, mape_train: {loss_mape[-1]:.6f}, mape_test: {Loss_mape_test[-1]:.6f}, tube_mape_test: {tube_loss_mape_test[-1]:.6f}')
                
        
        return loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test
     
class QuantumBandLayer(nn.Module):
    def __init__(self, in_features, out_features, num_bands=3, temperature=1.0, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bands = num_bands
        self.temperature = temperature
        self.device = device
        
        # Энергетические зоны
        self.energy_bands = nn.ModuleList([
            nn.Linear(in_features, out_features).to(device)
            for _ in range(num_bands)
        ])
        
        # Веса переходов
        self.transition_weights = nn.Parameter(
            torch.randn(num_bands, num_bands, out_features, device=device) * 0.02
        )
        
        # Квантовые проекции
        self.quantum_projections = nn.ModuleList([
            nn.Linear(out_features, out_features).to(device)
            for _ in range(num_bands)
        ])
        
        self.band_mixing = nn.Parameter(torch.randn(num_bands, device=device) * 0.02)
        self.activation = nn.GELU()
        
    def to(self, device):
        super().to(device)
        self.device = device
        return self
        
    def quantum_transition(self, x):
        safe_x = torch.clamp(x, min=-10, max=10)
        return self.activation(safe_x) * torch.sigmoid(safe_x)
    
    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        
        band_states = [band(x) for band in self.energy_bands]
        band_states = torch.stack(band_states)
        
        quantum_states = [
            self.quantum_transition(proj(state))
            for proj, state in zip(self.quantum_projections, band_states)
        ]
        quantum_states = torch.stack(quantum_states)
        
        band_interactions = torch.einsum(
            'nbi,nmf,mbi->bf',
            quantum_states,
            self.transition_weights,
            band_states
        )
        
        mixed_state = torch.einsum(
            'n,nbi->bi', 
            F.softmax(self.band_mixing, dim=0),
            band_states
        )
        
        output = mixed_state + 0.5 * band_interactions
        
        if self.training:
            return output, quantum_states
        return output

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_bands=3, device='cpu'):
        super().__init__()
        self.device = device
        self.training = True
        
        # Первый квантовый слой
        self.quantum_layer1 = QuantumBandLayer(
            in_features=input_size,
            out_features=hidden_size,
            num_bands=num_bands,
            device=device
        )
        self.norm1 = nn.LayerNorm(hidden_size).to(device)
        self.dropout1 = nn.Dropout(0.1)
        


        # Второй квантовый слой
        self.quantum_layer2 = QuantumBandLayer(
            in_features=hidden_size,
            out_features=hidden_size,
            num_bands=num_bands,
            device=device
        )
        self.norm2 = nn.LayerNorm(hidden_size).to(device)
        self.dropout2 = nn.Dropout(0.1)
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        quantum_states = []
        
        # Первый квантовый блок
        if self.training:
            x, qs1 = self.quantum_layer1(x)
            quantum_states.append(qs1)
        else:
            x = self.quantum_layer1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        # Второй квантовый блок
        if self.training:
            x, qs2 = self.quantum_layer2(x)
            quantum_states.append(qs2)
        else:
            x = self.quantum_layer2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        # Выходной слой
        output = self.output_layer(x)
        
        if self.training:
            return output, quantum_states
        return output
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self



class AdaptiveLoss(nn.Module):
    def __init__(self, delta=0.01, quantum_weight=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.mse = nn.MSELoss()
        self.quantum_weight = quantum_weight
        
    def forward(self, pred, target, quantum_states):
        # Основная ошибка (Huber)
        main_loss = self.huber(pred, target)
        
        # MAPE для мониторинга
        mape = torch.abs(target - pred) / torch.clamp(target, min=1e-7)
        
        # Квантовая регуляризация
        quantum_losses = []
        for qs in quantum_states:
            # Правильное вычисление mean_state
            # qs имеет размерность [num_bands, batch_size, features]
            mean_state = qs.mean(dim=0, keepdim=True)  # [1, batch_size, features]
            
            # Приводим размерности в соответствие
            current_qs = qs.transpose(0, 1)  # [batch_size, num_bands, features]
            current_mean = mean_state.expand_as(qs).transpose(0, 1)  # [batch_size, num_bands, features]
            
            quantum_losses.append(self.mse(current_qs, current_mean))
        
        quantum_loss = sum(quantum_losses)
        
        # Адаптивное взвешивание
        alpha = torch.sigmoid(main_loss.detach())
        
        # Общая функция потерь
        total_loss = main_loss + self.quantum_weight * alpha * quantum_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'quantum_loss': quantum_loss,
            'mape': torch.mean(mape),
            'alpha': alpha
        }

class QuantumTrainer:
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = AdaptiveLoss(delta=0.01).to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def create_scheduler(self, step_size=50, gamma=0.5):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
        
    def evaluate(self, X, y, loss_tube=5):
        self.model.eval()
        with torch.no_grad():
            # В режиме eval модель возвращает только предсказания
            y_pred = self.model(X)
            
            # Создаем фиктивные quantum_states для criterion
            dummy_states = [torch.zeros_like(y_pred).unsqueeze(0)]
            metrics = self.criterion(y_pred, y, dummy_states)
            
            loss_rd = torch.abs(y - y_pred) / torch.clamp(y, min=1e-7)
            per_loss_rd = loss_rd[loss_rd < 0.01 * loss_tube].shape[0] / (loss_rd.shape[0] * loss_rd.shape[1])
        
        self.model.train()
        return metrics, per_loss_rd
        
    def train_epoch(self, X, y, batch_size):
        dataset_size = X.shape[0]
        indices = torch.randperm(dataset_size)
        X_shuffled = X[indices].to(self.device)
        y_shuffled = y[indices].to(self.device)
        
        epoch_metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'quantum_loss': 0.0,
            'mape': 0.0,
            'alpha': 0.0
        }
        n_batches = 0
        
        for i in range(0, dataset_size, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            # В режиме train модель возвращает (predictions, quantum_states)
            predictions, quantum_states = self.model(X_batch)
            
            metrics = self.criterion(predictions, y_batch, quantum_states)
            metrics['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key].item()
            n_batches += 1
            
        return {k: v / n_batches for k, v in epoch_metrics.items()}
    
    def fit(self, X, y, X_t, y_t, batch_size, epochs, loss_tube=5):
        self.model.to(self.device)
        scheduler = self.create_scheduler()
        
        history = {
            'train_total_loss': [],
            'train_main_loss': [],
            'train_quantum_loss': [],
            'train_mape': [],
            'train_alpha': [],
            'test_mape': [],
            'test_tube': []
        }
        
        for epoch in range(epochs):
            # Обучение на эпохе
            train_metrics = self.train_epoch(X, y, batch_size)
            
            # Шаг планировщика
            scheduler.step()
            
            # Сохранение метрик обучения
            history['train_total_loss'].append(train_metrics['total_loss'])
            history['train_main_loss'].append(train_metrics['main_loss'])
            history['train_quantum_loss'].append(train_metrics['quantum_loss'])
            history['train_mape'].append(train_metrics['mape'])
            history['train_alpha'].append(train_metrics['alpha'])
            
            # Оценка на тестовых данных
            test_metrics, test_tube = self.evaluate(X_t, y_t, loss_tube)
            history['test_mape'].append(test_metrics['mape'].item())
            history['test_tube'].append(test_tube)
            
            # Вывод прогресса
            if (epoch + 1) % 100 == 0 or epoch == 9:
                print(
                    f'Epoch {epoch + 1}\n'
                    f'Train - Total: {train_metrics["total_loss"]:.6f}, '
                    f'Main: {train_metrics["main_loss"]:.6f}, '
                    f'Quantum: {train_metrics["quantum_loss"]:.6f}, '
                    f'MAPE: {train_metrics["mape"]:.6f}, '
                    f'Alpha: {train_metrics["alpha"]:.6f}\n'
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



    