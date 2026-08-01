import torch
from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer
from custom_transformer.transformer_model import TransformerLanguageModel
# TODO: generate() 래퍼 함수가 유실된 상태 (02_RAG_INTEGRATION.md 재배치 과정에서 누락 추정).
# Track A 종료로 보류 중. 아래 두 테스트는 generate 함수 복구 전까지 비활성화.
# from generate import generate


def test_bpe_with_transformer():
    """
    BPE Tokenizer + TransformerLanguageModel 연결 테스트

    Current Status (2026.06.19)
        - 연결 자체는 기술적으로 동작함 (에러 없이 encode → generate → decode 흐름 완성).
        - 다만 tokenizer vocabulary 품질이 낮고, 모델이 학습되지 않은 상태이기 때문에
          생성 결과가 의미 없는 문자열로 나오는 것이 정상적인 현상임.
        - 이 테스트는 '연결이 되는가'를 검증하는 용도이며,
          '의미 있는 생성이 되는가'는 아직 의미가 없음.
    """
    print("=== BPE + Transformer 연결 테스트 시작 ===\n")
    # Note: 위 결과가 garbage로 나오는 것은 예상된 현상입니다.
    #       - BPE vocabulary 품질 부족
    #       - Transformer 가중치가 random init 상태
    #       두 가지가 주요 원인. (자세한 내용은 bpe_tokenizer.py, transformer_model.py 주석 참조)

    # 1. Tokenizer 학습
    corpus = [
        "low lower lowest",
        "new newer newest",
        "hello world hello"
    ]
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)

    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")

    # 2. 모델 생성
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.token_to_id),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256
    )

    # 3. 텍스트 생성 테스트
    test_text = "hello"
    input_ids = tokenizer.encode(test_text)
    print(f"Input text: {test_text}")
    print(f"Encoded IDs: {input_ids}")

    input_tensor = torch.tensor([input_ids])

    generated = model.generate(
        input_ids=input_tensor,
        max_new_tokens=15,
        temperature=0.8
    )

    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated text: {generated_text}")

    print("\n=== 테스트 완료 ===")


# def test_generate_text():
#     result = generate("hello how are you", max_new_tokens=15)
    
#     assert isinstance(result, str)
#     assert len(result) > 0
#     print(f"생성 결과: {result}")
#     print("✅ 전체 파이프라인 테스트 통과")

# def test_dependency_injection():
#     # 모델과 토크나이저를 직접 생성해서 주입
#     tokenizer = BPETokenizer(vocab_size=50)
#     tokenizer.train(["hello world"])
    
#     model = TransformerLanguageModel(vocab_size=len(tokenizer.token_to_id))
    
#     result = generate("hello", max_new_tokens=10, model=model, tokenizer=tokenizer)
    
#     assert isinstance(result, str)
#     print("✅ 의존성 주입 테스트 통과")

if __name__ == "__main__":
    test_bpe_with_transformer()
    # test_generate_text()
    # test_dependency_injection()