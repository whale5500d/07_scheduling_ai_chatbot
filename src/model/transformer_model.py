import torch
import torch.nn as nn
import torch.nn.functional as F
from model.positional_encoding import PositionalEncoding
from model.transformer_decoder import TransformerDecoder
from model.decoder_layer import DecoderLayer

class TransformerLanguageModel(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 3. Decoder Layer (Post-LN 스타일)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        # 4. TransformerDecoder (여러 층 쌓기)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # 5. 출력층 (다음 토큰 예측)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        """
        input_ids: (batch_size, seq_len)

        TODO: Pre-LN vs Post-LN 구조 변경 검토 (DecoderLayer 내부에 Post-LN 주석 있음)
        """
        # Embedding
        x = self.embedding(input_ids) * (self.d_model ** 0.5)  # 원본 논문에서 사용한 scaling 방법 (선택)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Transformer Decoder
        x = self.decoder(x, mask)

        # 출력 (다음 토큰 확률 분포)
        logits = self.output_linear(x)

        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        텍스트 생성 함수 (Autoregressive Decoding)

        Args:
            input_ids: (batch_size, seq_len) 형태의 입력 토큰
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 시 다양성 조절 (1.0 = 기본, 낮을수록 보수적)
            top_k: 상위 k개 토큰 중에서만 샘플링 (None이면 전체 사용)

        Known Limitation (2026.06.19 기준):
            - EOS 토큰 기반 early stopping이 구현되어 있지 않음.
            - 따라서 max_new_tokens만큼 무조건 생성함.
            - BOS 토큰을 자동으로 붙여주지 않음 (호출하는 쪽에서 처리 필요).
            - 현재는 가중치가 학습되지 않은 상태(random init)에서는 의미 있는 생성이 어려움.
            - temperature / top_k는 지원되지만, repetition_penalty 등 고급 샘플링 기법은 미구현.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 현재 시퀀스로 모델에 넣음
            logits = self(input_ids)[:, -1, :]   # 마지막 토큰의 logits만 사용

            # Temperature 적용
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k 샘플링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 확률 분포로 변환 후 다음 토큰 샘플링
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 생성된 토큰을 기존 시퀀스 뒤에 붙임
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids