"""
Instruction Tuning을 위한 QA 쌍 -> (input_ids, labels) 변환 모듈

목적:
    - 질문(question) + 답변(answer)을 하나의 시퀀스로 이어붙이고,
      답변 부분에 대해서만 loss가 계산되도록 labels를 마스킹(masking)한다.
    - 질문 부분은 IGNORE_INDEX(-100)로 채워서,
      CrossEntropyLoss(ignore_index=-100)가 자동으로 무시하게 한다.
"""

IGNORE_INDEX = -100


def build_qa_training_pair(question_ids: list[int], answer_ids: list[int]) -> tuple[list[int], list[int]]:
    """
    질문/답변 토큰 ID 리스트를 받아서, next-token-prediction 학습에 사용할
    (inputs, labels) 쌍을 생성한다. 답변 부분만 loss에 포함되도록 마스킹한다.

    Args:
        question_ids: 질문을 encode()한 토큰 ID 리스트
        answer_ids: 답변을 encode()한 토큰 ID 리스트

    Returns:
        inputs: 전체 시퀀스에서 마지막 토큰을 제외한 입력
        labels: 전체 시퀀스에서 첫 토큰을 제외한 정답.
                질문 영역에 해당하는 위치는 IGNORE_INDEX(-100)로 마스킹됨.

    Raises:
        ValueError: question_ids 또는 answer_ids가 비어 있는 경우.
                    (둘 중 하나라도 비어 있으면 경계 계산이 의미를 가지지 못함)
    """
    if len(question_ids) == 0:
        raise ValueError("question_ids는 비어 있을 수 없습니다.")
    if len(answer_ids) == 0:
        raise ValueError("answer_ids는 비어 있을 수 없습니다.")

    full_ids = question_ids + answer_ids

    # next-token-prediction 기본 shift
    inputs = full_ids[:-1]
    labels = full_ids[1:]

    # 마스킹 경계: len(question_ids) - 1 이전 위치(질문 내부 예측)는 제외
    mask_boundary = len(question_ids) - 1

    masked_labels = [
        label if idx >= mask_boundary else IGNORE_INDEX
        for idx, label in enumerate(labels)
    ]

    return inputs, masked_labels