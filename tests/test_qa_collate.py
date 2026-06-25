"""
build_qa_training_pair() 테스트

검증 항목:
    1. 정상 케이스: 마스킹 경계가 정확한 위치에 적용되는가
    2. labels 길이가 inputs 길이와 같은가 (shift 후 동일한 길이여야 함)
    3. 질문 영역 전부가 IGNORE_INDEX(-100)인가
    4. 답변 영역에는 IGNORE_INDEX가 하나도 없는가
    5. question_ids 또는 answer_ids가 비어 있으면 ValueError가 발생하는가
"""

import pytest
from scripts.data_utils.qa_collate import build_qa_training_pair, IGNORE_INDEX


def test_masking_boundary_position():
    """
    question_ids=[1,2,3] (3개), answer_ids=[4,5,6] (3개)인 경우:
    full_ids = [1,2,3,4,5,6]
    inputs   = [1,2,3,4,5]
    labels   = [2,3,4,5,6]
    mask_boundary = len(question_ids) - 1 = 2
    -> labels 인덱스 0,1은 마스킹, 인덱스 2부터는 원래 값 유지
    """
    question_ids = [1, 2, 3]
    answer_ids = [4, 5, 6]

    inputs, labels = build_qa_training_pair(question_ids, answer_ids)

    assert inputs == [1, 2, 3, 4, 5]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6]


def test_inputs_labels_same_length():
    """inputs와 labels는 항상 같은 길이를 가져야 한다 (CrossEntropyLoss 계산 전제 조건)."""
    question_ids = [10, 11]
    answer_ids = [20, 21, 22]

    inputs, labels = build_qa_training_pair(question_ids, answer_ids)

    assert len(inputs) == len(labels)


def test_question_region_fully_masked():
    """질문 영역에 해당하는 labels 위치는 전부 IGNORE_INDEX여야 한다."""
    question_ids = [1, 2, 3, 4]  # 4개
    answer_ids = [5, 6]

    _, labels = build_qa_training_pair(question_ids, answer_ids)

    mask_boundary = len(question_ids) - 1  # 3
    question_region = labels[:mask_boundary]

    assert all(label == IGNORE_INDEX for label in question_region)


def test_answer_region_not_masked():
    """답변 영역에는 IGNORE_INDEX가 하나도 없어야 한다."""
    question_ids = [1, 2, 3, 4]
    answer_ids = [5, 6, 7]

    _, labels = build_qa_training_pair(question_ids, answer_ids)

    mask_boundary = len(question_ids) - 1
    answer_region = labels[mask_boundary:]

    assert all(label != IGNORE_INDEX for label in answer_region)


def test_single_token_question_and_answer():
    """질문과 답변이 각각 토큰 1개인 최소 경계 케이스."""
    question_ids = [1]
    answer_ids = [2]

    inputs, labels = build_qa_training_pair(question_ids, answer_ids)

    # full_ids = [1, 2] -> inputs = [1], labels = [2]
    # mask_boundary = len([1]) - 1 = 0 -> 인덱스 0부터 포함 (마스킹 없음)
    assert inputs == [1]
    assert labels == [2]


def test_empty_question_raises_value_error():
    with pytest.raises(ValueError):
        build_qa_training_pair([], [1, 2, 3])


def test_empty_answer_raises_value_error():
    with pytest.raises(ValueError):
        build_qa_training_pair([1, 2, 3], [])