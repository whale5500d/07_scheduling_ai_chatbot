import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # === Post-LN 구조 (원본 논문 스타일) ===
        # TODO: 추후 Post-LN에서 Pre-LN 방식으로 변경 예정
        # Post-LN 방식은 Pre-LN 방식에 비해 상대적으로 학습 불안정

        # 1. Self-Attention + Residual + LayerNorm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Feed Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x