import torch
from model.scaled_dot_product_attention import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    print("=== ScaledDotProductAttention 테스트 시작 ===")

    # 하이퍼파라미터 설정
    batch_size = 2
    seq_len = 5
    d_k = 64
    d_v = 64

    # 더미 데이터 생성
    query = torch.randn(batch_size, seq_len, d_k) # test shape: (2, 5, 64)
    key = torch.randn(batch_size, seq_len, d_k) # test shape: (2, 5, 64)
    value = torch.randn(batch_size, seq_len, d_v) # test shape: (2, 5, 64)

    # 모델 생성
    attention = ScaledDotProductAttention(dropout=0.0)
    # === 테스트 1: Mask 없이 ===
    print("\n[테스트 1] Mask 없이 실행")
    output, attn_weights = attention(query, key, value)
    print(f"Output shape: {output.shape}")           # 기대: (2, 5, 64)
    print(f"Attention weights shape: {attn_weights.shape}")  # 기대: (2, 5, 5)

    # === 테스트 2: Causal Mask 적용 ===
    # TODO: 현재는 Single Head 기준으로 설계되었으므로, 3차원으로 결과가 반환됨.
    # 추후 Multi-Head로 변경 시 4차원으로 Scaled Dot-Product Attention의 소스 코드 변환 필요
    print("\n[테스트 2] Causal Mask 적용")
    mask = torch.tril(torch.ones(seq_len, seq_len)) # test shape: (5, 5)
    mask = mask.unsqueeze(0) # test shape: (1, 5, 5)
    mask = mask.expand(batch_size, -1, -1)  # test shape: (2, 5, 5)

    output_masked, attn_weights_masked = attention(query, key, value, mask=mask) # 기대: (2, 5, 64), (2, 5, 5)
    print(f"Output shape (masked): {output_masked.shape}")
    print(f"Attention weights shape (masked): {attn_weights_masked.shape}")

    # Attention weights 확인 (미래 단어에 대한 가중치가 0에 가까운지)
    print("\nAttention weights 예시 (첫 번째 배치, 첫 번째 query):")
    print(attn_weights_masked[0])

    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_scaled_dot_product_attention()