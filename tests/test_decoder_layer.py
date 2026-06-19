import torch
from model.decoder_layer import DecoderLayer

def test_decoder_layer():
    print("=== DecoderLayer (Post-LN) 테스트 시작 ===\n")

    # 하이퍼파라미터 설정
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    d_ff = 256          # Feed Forward Network의 hidden dimension

    print(f"설정: batch_size={batch_size}, seq_len={seq_len}, "
          f"d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}\n")

    # 더미 입력 데이터 생성 (Self-Attention이므로 query = key = value)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}\n")

    # DecoderLayer 생성 (Post-LN 스타일)
    decoder_layer = DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0
    )

    # === 테스트 1: Mask 없이 실행 ===
    print("[테스트 1] Mask 없이 실행")
    output = decoder_layer(x)   # mask=None

    print(f"Output shape: {output.shape}")   # 기대: (2, 6, 64)
    print()

    # Shape 검증
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape이 예상과 다릅니다. 예상: {(batch_size, seq_len, d_model)}, 실제: {output.shape}"

    print("✅ 테스트 1 통과: Mask 없이 정상 동작\n")

    # === 테스트 2: Causal Mask 적용 ===
    print("[테스트 2] Causal Mask 적용")

    # Causal Mask 생성 (batch, 1, seq, seq)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    output_masked = decoder_layer(x, mask=causal_mask)
    print(f"Output shape (with mask): {output_masked.shape}")
    print()

    assert output_masked.shape == (batch_size, seq_len, d_model), \
        "Mask 적용 시 Output shape이 예상과 다릅니다."

    print("✅ 테스트 2 통과: Causal Mask 적용 정상 동작\n")

    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_decoder_layer()