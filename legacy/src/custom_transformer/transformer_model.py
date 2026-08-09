import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_transformer.model.positional_encoding import PositionalEncoding
from custom_transformer.model.transformer_decoder import TransformerDecoder
from custom_transformer.model.decoder_layer import DecoderLayer
from custom_transformer.model.utils.generation_utils import trim_after_eos

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


        # Weight tying: output_linear이 embedding과 동일한 가중치를 공유하도록 묶음.
        # (Press & Wolf, 2016 - "Using the Output Embedding to Improve Language Models")
        # nn.Embedding.weight shape: (vocab_size, d_model)
        # nn.Linear.weight shape:    (out_features, in_features) = (vocab_size, d_model)
        # 두 shape가 동일하므로 그대로 공유 가능하다.

        # embedding의 기본 초기화(std≈1.0)가 output_linear의 기본 초기화(std≈0.036)보다
        # 훨씬 커서, 그대로 tying하면 logits 스케일이 폭발해 학습이 불안정해진다
        # (Epoch 1 loss가 tying 전 대비 약 10배로 뛰고, 이후 loss가 진동하며 발산하는
        # 현상으로 실제 확인됨). tying 전에 embedding을 output_linear와 비슷한
        # 스케일(std = d_model ** -0.5)로 재초기화해 이 문제를 해소한다.
        #
        # 효과: 정답으로 한 번도 등장하지 않은 토큰(예: held-out activity)도,
        # 입력으로 등장해 embedding이 학습되면 그 학습된 표현이 output_linear에도
        # 즉시 반영되어, "출력 후보에서 완전히 배제되는" 구조적 문제가 해소된다.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        self.output_linear.weight = self.embedding.weight

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
    
    @torch.no_grad() # Transformer는 모델이 이미 훈련되어 있다고 가정하기 떄문에, 필요 시 별도의 메서드(train, 아직 미구현 상태) 필요
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=None):
        """
        텍스트 생성 함수 (Autoregressive Decoding)

        Args:
            input_ids: (batch_size, seq_len) 형태의 입력 토큰
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 시 다양성 조절 (1.0 = 기본, 낮을수록 보수적)
            top_k: 상위 k개 토큰 중에서만 샘플링 (None이면 전체 사용)
            eos_token_id: 종료 토큰 ID. None이면 조기 종료를 적용하지 않고
                          기존과 동일하게 max_new_tokens만큼 무조건 생성한다.
                          값이 주어지면, 생성된 토큰이 eos_token_id와 일치하는 순간
                          생성을 멈춘다 (조기 종료).

        Returns:
            생성된 전체 토큰 시퀀스 (batch_size, seq_len + 생성된 길이).
            eos_token_id가 주어졌고 생성 중 eos가 나온 경우, 반환되는 시퀀스에는
            eos 토큰 자체가 포함된다 (decode 시점에 trim_after_eos로 제거하는 것을 권장).

        Known Limitation (2026.06.25 기준, EOS 조기 종료 추가):
            - BOS 토큰을 자동으로 붙여주지 않음 (호출하는 쪽에서 처리 필요).
            - 현재는 가중치가 학습되지 않은 상태(random init)에서는 의미 있는 생성이 어려움.
            - temperature / top_k는 지원되지만, repetition_penalty 등 고급 샘플링 기법은 미구현.
            - (주의 사항) eos_token_id 기반 조기 종료는 batch_size=1을 전제로 구현됨.
              batch_size > 1인 경우, 시퀀스마다 종료 시점이 다를 수 있어
              현재 구현은 배치 내 모든 시퀀스가 동시에 멈추는 방식으로 동작하지 않는다.
              (개별 시퀀스별 조기 종료는 추가 구현이 필요한 확장 과제)
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
            probs = F.softmax(logits, dim=-1) # 마지막 차원에 softmax를 적용하라는 의미(vocab 전체에 대한 확률 계산을 위해)
            next_token = torch.multinomial(probs, num_samples=1) # 확률 분포의 확률에 따라 랜덤하게 하나를 샘플링

            # 생성된 토큰을 기존 시퀀스 뒤에 붙임
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # EOS 조기 종료 (batch_size=1 전제)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids