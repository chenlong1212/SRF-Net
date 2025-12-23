import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_classes=6, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  
        return self.fc(last_hidden)


# ===== 2. GRU =====
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6, num_layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (N, H)
        return self.fc(last_hidden)


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_classes=6, hidden_dim=64, num_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [hidden_dim, hidden_dim]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                     dilation=dilation, padding=(kernel_size-1)*dilation,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.mean(dim=2)  
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(2) < self.pe.size(2):
            pad = torch.zeros(x.size(0), x.size(1), self.pe.size(2) - x.size(2), device=x.device)
            x = torch.cat([x, pad], dim=2)
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, num_classes=6, d_model=8, num_heads=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model  
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        x = self.input_proj(x)   
        x = self.pos_encoder(x)
        out = self.transformer(x)  # (batch, seq_len, d_model)
        out = out.mean(dim=1)     
        return self.fc(out)

# ===== 5. ST-GCN =====
class STGCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        # x: (N, T, D)
        res = self.residual(x)
        x = self.fc1(x)
        x = self.bn1(x.transpose(1,2)).transpose(1,2)  # BatchNorm1d 
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x.transpose(1,2)).transpose(1,2)
        x = self.dropout(x)
        x = x + res
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.layer1 = STGCNBlock(input_dim, hidden_dim)
        self.layer2 = STGCNBlock(hidden_dim, hidden_dim*2)
        self.layer3 = STGCNBlock(hidden_dim*2, hidden_dim*4)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*4, num_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x.transpose(1,2)).squeeze(-1) 
        x = self.fc_dropout(x)
        out = self.fc(x)
        return out


class STGCNBlock_AG(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim)) 

    def forward(self, x):

        res = self.residual(x)

        x = self.fc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)


        x = torch.matmul(x, self.A)

        x = self.fc2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = x + res
        x = self.relu(x)
        return x

class STGCN_AG(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.layer1 = STGCNBlock_AG(input_dim, hidden_dim)
        self.layer2 = STGCNBlock_AG(hidden_dim, hidden_dim * 2)
        self.layer3 = STGCNBlock_AG(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
   
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

 
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (N, hidden_dim*4)
        x = self.fc_dropout(x)
        out = self.fc(x)
        return out


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
   
        Q = self.query(x)  
        K = self.key(x)   
        V = self.value(x) 

  
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)

 
        out = torch.matmul(attn, V)
        return out

class STGCNBlock_TA(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()


    
        self.temporal_attn = TemporalAttention(hidden_dim)

    def forward(self, x):
   
        res = self.residual(x)

        x = self.fc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)
 
        x = self.temporal_attn(x)

        x = self.fc2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = x + res
        x = self.relu(x)
        return x

class STGCN_TA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.layer1 = STGCNBlock_TA(input_dim, hidden_dim)
        self.layer2 = STGCNBlock_TA(hidden_dim, hidden_dim * 2)
        self.layer3 = STGCNBlock_TA(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

 
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  
        x = self.fc_dropout(x)
        out = self.fc(x)
        return out



class STGCNBlock_AG_TA(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()


        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

     
        self.temporal_attn = TemporalAttention(hidden_dim)

    def forward(self, x):
  
        res = self.residual(x)

        x = self.fc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)

     
        x = torch.matmul(x, self.A)

  
        x = self.temporal_attn(x)

        x = self.fc2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = x + res
        x = self.relu(x)
        return x


class STGCN_AG_TA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.layer1 = STGCNBlock_AG_TA(input_dim, hidden_dim)
        self.layer2 = STGCNBlock_AG_TA(hidden_dim, hidden_dim * 2)
        self.layer3 = STGCNBlock_AG_TA(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
     
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

   
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  
        x = self.fc_dropout(x)
        out = self.fc(x)
        return out



model_dict = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "TCN": TemporalConvNet,
    "Transformer": TransformerEncoderModel,
    "STGCN": STGCN,
    "STGCN_AG": STGCN_AG,
    "STGCN_TA": STGCN_TA,
    "STGCN_AG_TA": STGCN_AG_TA,
}
