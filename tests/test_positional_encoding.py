import torch
from model.positional_encoding import PositionalEncoding

# 테스트
pe = PositionalEncoding(d_model=64, max_len=100)

# Positional Encoding 값 일부 확인
print("Positional Encoding (처음 5개 위치, 처음 8개 차원):")
print(pe.get_buffer('pe')[0, :5, :8])

x = torch.randn(2, 10, 64)          # (batch, seq_len, d_model)

output = pe(x)
print("Input shape :", x.shape) # 기대: (2, 10, 64)
print("Output shape:", output.shape) # 기대: (2, 10, 64)
print("Positional Encoding 적용 완료")