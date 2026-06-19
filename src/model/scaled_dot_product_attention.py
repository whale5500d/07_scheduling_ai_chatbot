from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention
    - Attention 메커니즘의 가장 기본 단위
    - Q, K, V를 받아서 Attention을 계산
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor, # test shape: (2, 5, 64)
                key: torch.Tensor, # test shape: (2, 5, 64)
                value: torch.Tensor, # test shape: (2, 5, 64)
                mask: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        
        d_k = query.size(-1) # Key의 마지막 차원 크기

        # 1. Q와 K의 유사도 계산 (dot product)
        # (batch, seq, d_k) @ (batch, d_k, seq) -> (batch, seq, seq)
        # test shape: (2, 5, 64) @ (2, 64, 5) -> (2, 5, 5)
        scores = torch.matmul(query, key.transpose(-2, -1)) # test shape: (2, 5, 5)

        # 2. Scaling (Scores 안정화)
        scores = scores / math.sqrt(d_k) # test shape: (2, 5, 5)

        # 3. Mask(Causal Mask) 적용 (미래 단어 차단)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # test shape: (2, 5, 5)

        # 4. Softmax로 Attention 가중치 계산
        attn_weights = F.softmax(scores, dim=-1) # test shape: (2, 5, 5)
        attn_weights = self.dropout(attn_weights) # test shape: (2, 5, 5)

        # 5. Softmax 결과 * V -> 가중 평균
        # (batch, seq, seq) @ (batch, seq, d_v) -> (batch, seq, d_v)
        # test shape: (2, 5, 5) @ (2, 5, 64) -> (2, 5, 64)
        output = torch.matmul(attn_weights, value) # test shape: (2, 5, 64)

        return output, attn_weights # test shape: (2, 5, 64), (2, 5, 5)
