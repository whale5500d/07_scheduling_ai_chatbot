import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (max_len, d_model) 크기의 positional encoding 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)   # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)   # 홀수 인덱스

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)으로 차원 확장
        self.register_buffer('pe', pe)  # 학습되지 않는 버퍼로 등록

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.get_buffer('pe')[:, :x.size(1)]
        return self.dropout(x)