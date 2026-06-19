import torch
from model.transformer_model import TransformerLanguageModel

def test_transformer_language_model():
    print("=== TransformerLanguageModel 테스트 시작 ===\n")

    # 하이퍼파라미터 설정
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    max_len = 256

    print(f"설정: vocab_size={vocab_size}, d_model={d_model}, "
          f"num_heads={num_heads}, num_layers={num_layers}, d_ff={d_ff}\n")

    # 모델 생성
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1
    )

    # === 테스트 1: Forward Pass ===
    print("[테스트 1] Forward Pass")
    batch_size = 2
    seq_len = 12
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)  # mask=None

    print(f"Input shape : {input_ids.shape}")   # (2, 12)
    print(f"Logits shape: {logits.shape}")      # 기대: (2, 12, 1000)
    print()

    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Logits shape이 예상과 다릅니다. 예상: {(batch_size, seq_len, vocab_size)}, 실제: {logits.shape}"

    print("✅ 테스트 1 통과: Forward Pass 정상 동작\n")

    # === 테스트 2: 텍스트 생성 (Generate) ===
    print("[테스트 2] 텍스트 생성 (Generate)")
    start_token = torch.tensor([[1]])  # 시작 토큰

    generated = model.generate(
        input_ids=start_token,
        max_new_tokens=15,
        temperature=0.8,
        top_k=30
    )

    print(f"시작 토큰 shape     : {start_token.shape}")
    print(f"생성된 시퀀스 shape : {generated.shape}")  # 기대: (1, 16)
    print(f"생성된 토큰 ID     : {generated.tolist()[0]}")
    print()

    assert generated.shape[1] == 1 + 15, "생성된 시퀀스 길이가 예상과 다릅니다."

    print("✅ 테스트 2 통과: Generate 정상 동작\n")

    # === 테스트 3: Causal Mask 적용 ===
    print("[테스트 3] Causal Mask 적용")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    logits_with_mask = model(input_ids, mask=causal_mask)
    print(f"Masked Logits shape: {logits_with_mask.shape}")
    print()

    assert logits_with_mask.shape == (batch_size, seq_len, vocab_size), \
        "Mask 적용 시 Logits shape이 예상과 다릅니다."

    print("✅ 테스트 3 통과: Causal Mask 적용 정상 동작\n")

    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_transformer_language_model()