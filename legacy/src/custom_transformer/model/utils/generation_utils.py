"""
생성된 토큰 시퀀스에서 <eos> 처리를 담당하는 유틸리티.

생성(generate) 로직과 분리한 이유:
    - TransformerLanguageModel.generate()는 실제 학습된 가중치가 있어야
      의미 있는 단위 테스트가 가능하다.
    - "eos가 나오면 그 이후를 잘라낸다"는 규칙 자체는 모델과 무관한
      순수 로직이므로, 별도 함수로 분리해서 모델 없이 테스트 가능하게 한다.
"""


def trim_after_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    """
    token_ids 안에서 eos_token_id가 처음 등장하는 위치를 찾아,
    그 이전까지의 토큰만 남기고 잘라낸다 (eos 토큰 자체는 결과에 포함하지 않음).

    eos_token_id가 없으면 원본 리스트를 그대로 반환한다.

    Args:
        token_ids: 생성된 토큰 ID 시퀀스 (질문 + 답변 전체)
        eos_token_id: 종료를 의미하는 토큰 ID

    Returns:
        eos 이전까지 잘린 토큰 ID 리스트
    """
    if eos_token_id in token_ids:
        eos_position = token_ids.index(eos_token_id)
        return token_ids[:eos_position]
    return token_ids