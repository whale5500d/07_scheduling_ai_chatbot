import torch.nn as nn
import torch.nn.functional as F
from model.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # assert: 실행 시점에 조건을 검사하는 Assertion(단언문)
        # 필요성: Head 차원 분할 시, 정확히 나누어 떨어지지 않으면 view가 동작하지 않음.
        # Head를 깔끔하게 분할해야만 행렬이 정상 계산 가능
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads   # 한 개의 head가 다루는 차원

        # Q, K, V를 생성하는 Linear layer (Projection)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 여러 Head의 결과를 합친 후 최종 출력하는 Linear
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Single Head Attention 모듈 (재사용)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Q, K, V 생성
        Q = self.q_linear(query)   # (batch, seq, d_model)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 2. Head 차원으로 쪼개기 (Split)
        # (batch, seq, d_model) → (batch, seq, num_heads, head_dim)
        # → (batch, num_heads, seq, head_dim)으로 차원 순서 변경
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention 수행
        # (여러 Head를 한 번에) 앞에서 view로 Q, K, V 값을 쪼갰기 때문에 비록 attention은 하나지만 병렬처리
        # TODO: mask 차원도 필요에 따라 조정해야 함 (나중에 다룰 부분)
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 4. Head 차원 다시 합치기 (Concatenate)
        # (batch, num_heads, seq, head_dim) → (batch, seq, num_heads, head_dim)
        # → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 최종 Linear 통과 (Output Projection)
        output = self.out_linear(attn_output)

        return output, attn_weights