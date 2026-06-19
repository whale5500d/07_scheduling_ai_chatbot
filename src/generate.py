# generate.py
from typing import Optional
import torch

from model.transformer_model import TransformerLanguageModel
from tokenizer.bpe_tokenizer import BPETokenizer


# ============================================================
# 전역 모델/토크나이저 (Lazy Loading)
# ============================================================
_model: Optional[TransformerLanguageModel] = None
_tokenizer: Optional[BPETokenizer] = None


def _load_model_and_tokenizer():
    """
    모델과 토크나이저를 한 번만 로딩하는 함수.
    
    TODO (2026.06.19):
        - 현재는 dummy_corpus로 BPE를 학습하고 random init 모델을 사용 중.
        - 실제로는 미리 학습된 BPE Tokenizer와 학습된 TransformerLanguageModel의
          state_dict를 로드하는 방식으로 변경해야 함.
        - vocab_size도 학습된 tokenizer의 실제 크기로 맞춰야 함.
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    # === BPE Tokenizer 초기화 (임시) ===
    _tokenizer = BPETokenizer(vocab_size=1000)

    # TODO: 실제 학습 데이터로 BPE를 학습하거나, 저장된 tokenizer를 로드해야 함
    dummy_corpus = [
        "hello world hello",
        "low lower lowest",
        "new newer newest",
        "hello nice to meet you",
        "weather is good today"
    ]
    _tokenizer.train(dummy_corpus)

    # === TransformerLanguageModel 초기화 (random weights) ===
    vocab_size = len(_tokenizer.token_to_id)
    _model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_len=512,
        dropout=0.1
    )
    _model.eval()  # 추론 모드로 설정

    return _model, _tokenizer


def generate(
    prompt: str,
    max_new_tokens: int = 30,
    stop_sequences: Optional[list[str]] = None,
    model: Optional[TransformerLanguageModel] = None,
    tokenizer: Optional[BPETokenizer] = None
) -> str:
    """
    TransformerLanguageModel을 사용하는 실제 텍스트 생성 함수.
    
    단계 2의 더미(규칙 기반) 생성기를 완전히 대체함.
    
    Args:
        prompt: 사용자 입력 문장
        max_new_tokens: 생성할 최대 토큰 수
        stop_sequences: 생성 중단 조건 (현재는 미지원, TODO)
        model: 주입할 모델 (None이면 전역 인스턴스 사용)
        tokenizer: 주입할 토크나이저 (None이면 전역 인스턴스 사용)
    
    Returns:
        생성된 전체 텍스트
    """
    if stop_sequences is None:
        stop_sequences = []

    # 모델과 토크나이저 결정 (의존성 주입 또는 전역 로딩)
    if model is None or tokenizer is None:
        model, tokenizer = _load_model_and_tokenizer()

    # 1. Prompt를 token_ids로 변환
    input_ids = tokenizer.encode(prompt)

    # ============================================================
    # 임시 워크어라운드 (2026.06.19)
    # - BPE가 학습하지 못한 토큰은 -1로 반환되는데, 이는 모델의
    #   Embedding에서 IndexError를 발생시킴.
    # - Phase 1(연결 안정화)에서는 일단 -1을 제거하여 테스트를
    #   이어갈 수 있도록 함.
    # - 장기적으로는 bpe_tokenizer.py에서 <unk> 토큰을 제대로
    #   도입하고 처리하는 방식으로 변경해야 함.
    # ============================================================
    original_length = len(input_ids)
    input_ids = [tid for tid in input_ids if tid >= 0]

    if len(input_ids) == 0:
        # 모든 토큰이 OOV인 극단적인 경우 (거의 발생하지 않음)
        print("[경고] 모든 토큰이 OOV입니다. 빈 프롬프트로 대체합니다.")
        input_ids = [0]  # 첫 번째 토큰 ID로 대체 (임시)

    if len(input_ids) < original_length:
        print(f"[워크어라운드] OOV 토큰 {original_length - len(input_ids)}개 제거됨")
        
    input_tensor = torch.tensor([input_ids])  # (batch=1, seq_len)

    # 2. 모델을 통한 생성
    # TODO: 현재 model.generate()는 EOS 기반 early stopping을 지원하지 않음.
    #       따라서 max_new_tokens만큼 무조건 생성함.
    generated_ids = model.generate(
        input_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50
    )

    # 3. 생성된 token_ids를 텍스트로 복원
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    # TODO: stop_sequences에 해당하는 문자열이 생성되면 잘라내는 후처리 로직 추가 필요
    # 현재는 model.generate() 단계에서 제어하지 못하므로 후처리로 처리해야 함.

    # TODO: 학습 필요 - 비동기 모델 추론 최적화
    #
    # 현재 generate()는 동기 방식으로 모델을 호출하고 있음.
    # FastAPI는 비동기 프레임워크이므로, 다량의 동시 요청 시
    # 이벤트 루프가 블로킹되어 성능 저하가 발생할 수 있음.
    #
    # 고려할 수 있는 최적화 방향:
    # 1. fastapi.concurrency.run_in_threadpool 사용
    # 2. torch.jit / torch.compile 적용
    # 3. vLLM, TensorRT 등 별도 추론 엔진 도입
    # 4. 요청 큐 + Background Task 구조
    #
    # 지금은 모델 규모가 작아서 큰 문제가 없지만,
    # 추후 서비스화할 때 반드시 학습해야 할 주제.
    return generated_text


# ============================================================
# 직접 실행 테스트용
# ============================================================
if __name__ == "__main__":
    result = generate("오늘 날씨가", max_new_tokens=20)
    print("생성 결과:", result)