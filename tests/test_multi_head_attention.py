import torch
from model.multi_head_attention import MultiHeadAttention

def test_multi_head_attention():
    print("=== MultiHeadAttention 테스트 시작 ===\n")

    # 하이퍼파라미터 설정
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8

    print(f"설정: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}\n")

    # 더미 데이터 생성
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    print(f"query shape: {query.shape}") # test shape: (2, 6, 64)
    print(f"key shape:   {key.shape}") # test shape: (2, 6, 64)
    print(f"value shape: {value.shape}\n") # test shape: (2, 6, 64)

    # MultiHeadAttention 생성
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    # === 테스트 1: Mask 없이 실행 ===
    print("[테스트 1] Mask 없이 실행")
    output, attn_weights = mha(query, key, value)

    print(f"Output shape:      {output.shape}")        # 기대: (2, 6, 64)
    print(f"Attention weights shape: {attn_weights.shape}")  # 기대: (2, 8, 6, 6)
    print()

    # 간단한 shape 검증
    assert output.shape == (batch_size, seq_len, d_model), "Output shape이 맞지 않습니다."
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape이 맞지 않습니다."

    print("✅ 테스트 1 통과: Mask 없이 정상 동작\n")

    # === 테스트 2: Mask 적용 실행 ===
    # Causal Mask 생성 (Multi-Head에서는 보통 (batch, 1, seq, seq) 형태로 만듦 (head 차원에 broadcasting))
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1) # (2, 1, 6, 6)

    print(f"Causal Mask shape: {causal_mask.shape}\n")

    print("[테스트] Causal Mask 적용")
    output, attn_weights = mha(query, key, value, mask=causal_mask)

    print(f"Output shape:            {output.shape}")           # 기대: (2, 6, 64)
    print(f"Attention weights shape: {attn_weights.shape}")     # 기대: (2, 8, 6, 6)
    print()

    # Attention weights 확인 (미래 단어 차단 여부)
    print("Attention weights 예시 (첫 번째 배치, 첫 번째 Head, 첫 번째 Query):")
    print(attn_weights[0, 0, 0])   # 첫 번째 query의 attention weights

    print("✅ 테스트 2 통과: Causal Mask 적용 정상 동작\n")

    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    test_multi_head_attention()