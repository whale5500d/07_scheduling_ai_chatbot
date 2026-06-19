# 테스트 코드 실행 결과

## Scaled Dot-Product Attention 테스트 결과

### 1. 테스트 목적

- `ScaledDotProductAttention` 클래스가 정상적으로 동작하는지 검증
- Attention 연산의 기본 로직(유사도 계산, Scaling, Masking, Softmax, 가중 평균)이 올바르게 구현되었는지 확인
- 특히 **Causal Mask** 적용 시 미래 단어를 제대로 차단하는지 검증

### 2. 테스트 환경

- Python 버전: 3.14.3
- PyTorch 버전: 2.12.0
- 테스트 파일: `test_attention.py`
- 실행 명령어: `python test_attention.py`

### 3. 테스트 케이스

1. 테스트 케이스 1 (Mask 없이 Attention 실행)
   - 목적
     - 기본적인 Scaled Dot-Product Attention 연산이 에러 없이 동작하는지 확인
     - 출력 텐서의 shape이 의도한 대로 나오는지 검증

   - 테스트 코드

     ```python
     # Mask 없이 실행
     output, attn_weights = attention(query, key, value)
     ```

   - 기대 결과
     - output.shape: `(2, 5, 64)` (batch_size, seq_len, d_v)
     - attn_weights.shape: `(2, 5, 5)` (batch_size, seq_len, seq_len)

   - 실제 결과
     - Output shape: `torch.Size([2, 5, 64])`
     - Attention weights shape: `torch.Size([2, 5, 5])`
     - 에러 없이 정상 실행

   - 해석 및 결론
     - 기본적인 Attention 연산(Q, K 유사도 계산 -> Scaling -> Softmax -> V 가중 평균)이 정상적으로 동작함
     - 텐서 shape도 예상과 일치하여 구현이 올바른 것으로 판단

2. 테스트 케이스 2 (Causal Mask 적용)
   - 목적
     - Causal Mask가 제대로 적용되는지 확인
     - 미래 단어(자신보다 뒤에 있는 단어)에 대한 Attention 가중치가 0이 되는지 검증
     - Decoder-only 모델에서 필수적인 자동회귀 특성이 잘 동작하는지 확인

   - 테스트 코드

     ```python
     # Causal Mask 생성 및 적용
     mask = torch.tril(torch.ones(seq_len, seq_len))
     mask = mask.unsqueeze(0)
     mask = mask.expand(batch_size, -1, -1)
     output_masked, attn_weights_masked = attention(query, key, value, mask=mask)
     ```

   - 기대 결과
     - Mask 적용 후에도 에러 없이 실행
     - Attention weights에서 **상삼각 영역(미래 단어 위치)의 값이 0**에 가까워야 함

   - 실제 결과

     ```python
     Output shape (masked): torch.Size([2, 5, 64])
     Attention weights shape (masked): torch.Size([2, 5, 5])

     Attention weights 예시 (첫 번째 배치, 첫 번째 query):
     tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.2929, 0.7071, 0.0000, 0.0000, 0.0000],
             [0.0358, 0.2070, 0.7572, 0.0000, 0.0000],
             [0.8696, 0.0549, 0.0272, 0.0483, 0.0000],
             [0.0206, 0.9253, 0.0059, 0.0275, 0.0207]])
     ```

   - 해석 및 결론
     - Causal Mask 적용 시, 각 Query Position에서 자신보다 뒤에 있는 key position의 Attention weight가 정확히 0으로 차단됨을 확인
     - 테스트 결과에서 첫 번째 Query(0번 위치)는 1번 이후 모든 위치가 0
     - 이는 Decoder-only Transformer에서 필요한 **자동회귀(Autoregressive) 특성이 정상적으로 동작하고 있음**을 의미

### 4. 전체 테스트 결론

- ✅ 기본 Attention 연산 정상 동작 확인
- ✅ Causal Mask 정상 동작 확인
- ✅ 텐서 Shape 일치 확인

### 5. 인사이트

- Causal Mask를 적용할 때 차원 관리가 중요함을 학습
  - 현재 Single Head 단계에서는 3차원 Mask가 적합
  - ⚠️ 추후 Multi Head 단계로 확장할 경우, mask 차원이 달라질 수 있으므로 추가적인 Attention 코드 개선이 필요
- 작은 단위부터 테스트하면서 shape과 동작을 검증하는 방식이 이후 구현에서 디버깅 시간을 줄이는 데 효과적
