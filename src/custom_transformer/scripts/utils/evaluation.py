# src/custom_transformer/scripts/utils/evaluation.py

"""
Held-out 평가 유틸리티: perplexity와 exact match(slot copy 정확도 / label 정확도)를 측정한다.

설계 원칙:
    - compute_perplexity()는 build_qa_training_pair()와 동일한 마스킹 방식(IGNORE_INDEX)을
      재사용해, training loss와 같은 기준(답변 영역만)으로 계산되도록 한다.
      "같은 지표를 다른 데이터셋(training set / held-out set)에 적용해 gap을 비교한다"는
      멘토링 논의의 전제를 그대로 만족시키기 위함이다.
    - exact match는 하나의 점수로 뭉치지 않고, slot copy 정확도(activity가 입력에서
      출력으로 그대로 복사되었는가)와 label 정확도(응/아니 판단이 맞았는가)를 독립적으로
      집계한다. 두 실패 원인을 구분해야 다음에 무엇을 고쳐야 하는지 알 수 있기 때문이다.

Known Limitation:
    - TIME_EXPRESSIONS에 정의되지 않은 새로운 시점 표현이 등장하면
      extract_question_activity()가 activity를 잘못 추출할 수 있다.
    - parse_generated_answer()는 "{label}, {activity} {할|갈} 거야" /
      "{label}, {activity} 대신 {substitute} {할|갈} 거야" 두 형식만 인식한다.
      korean_qa_train.txt / korean_qa_heldout.txt의 현재 문형과 1:1로 대응하며,
      문형이 확장되면(예: 멀티턴, 제안형 표현) 이 함수도 함께 확장해야 한다.
"""

import math

import torch
import torch.nn as nn

from custom_transformer.scripts.utils.qa_collate import build_qa_training_pair, IGNORE_INDEX
from custom_transformer.model.utils.generation_utils import trim_after_eos


TIME_EXPRESSIONS = ["오늘", "내일", "모레", "이번 주말"]


def strip_time_expression(question: str) -> str:
    """질문 문자열 맨 앞의 시점 표현(TIME_EXPRESSIONS)을 제거하고 나머지를 반환한다."""
    stripped_question = question.strip()
    for time_expression in TIME_EXPRESSIONS:
        prefix = time_expression + " "
        if stripped_question.startswith(prefix):
            return stripped_question[len(prefix):]
    return stripped_question


def extract_question_activity(question: str) -> str:
    """
    질문에서 activity(슬롯) 문자열만 추출한다.

    예: "오늘 산책 할 거야?" -> "산책"
        "내일 캠핑 갈 거야?" -> "캠핑"
    """
    without_time_expression = strip_time_expression(question)
    for verb_suffix in ["할 거야?", "갈 거야?"]:
        if without_time_expression.endswith(verb_suffix):
            return without_time_expression[: -len(verb_suffix)].strip()
    return without_time_expression.strip()


def parse_generated_answer(generated_answer: str) -> dict:
    """
    생성된 답변 문자열을 label과 activity(들)로 분리한다.

    Returns:
        {
            "label": "응" 또는 "아니" 또는 None(파싱 실패),
            "primary_activity": str,  # 응답에 등장한 첫 번째 activity
                                       # (아니의 경우 "거절된" activity)
            "substitute_activity": str or None,  # 아니일 때만 존재하는 대안 activity
        }
    """
    text = generated_answer.strip()

    if text.startswith("응"):
        label = "응"
        rest = text[1:].lstrip(", ").strip()
    elif text.startswith("아니"):
        label = "아니"
        rest = text[2:].lstrip(", ").strip()
    else:
        return {"label": None, "primary_activity": "", "substitute_activity": None}

    for verb_suffix in ["할 거야", "갈 거야"]:
        if rest.endswith(verb_suffix):
            rest = rest[: -len(verb_suffix)].strip()
            break

    if label == "아니" and "대신" in rest:
        declined_part, substitute_part = rest.split("대신", 1)
        return {
            "label": label,
            "primary_activity": declined_part.strip(),
            "substitute_activity": substitute_part.strip(),
        }

    return {"label": label, "primary_activity": rest.strip(), "substitute_activity": None}


def compute_perplexity(model, tokenizer, qa_pairs: list[tuple[str, str]]) -> float:
    """
    qa_pairs(질문, 답변) 각각에 대해 답변 영역만의 cross-entropy loss 합을 구하고,
    전체 답변 토큰 수로 나눈 뒤 exp()를 취해 corpus-level perplexity를 반환한다.

    reduction="sum"을 쓰는 이유:
        pair마다 답변 길이가 다르므로, pair 단위 평균(mean)들을 다시 평균 내면
        짧은 답변과 긴 답변이 동일 가중치를 갖게 되어 왜곡된다. 토큰 단위로
        합산한 뒤 마지막에 한 번만 나누어야 corpus 전체에서 토큰 하나당
        평균 loss가 정확히 계산된다.
    """
    model.eval()
    vocab_size = len(tokenizer.token_to_id)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]

    total_loss = 0.0
    total_answer_token_count = 0

    with torch.no_grad():
        for question, answer in qa_pairs:
            question_ids = tokenizer.encode(question)
            answer_ids = tokenizer.encode(answer) + [eos_id]

            if len(question_ids) == 0 or len(answer_ids) == 0:
                continue

            inputs, labels = build_qa_training_pair(question_ids, answer_ids)
            inputs_tensor = torch.tensor([inputs])
            labels_tensor = torch.tensor([labels])

            outputs = model(inputs_tensor)
            loss = criterion(outputs.view(-1, vocab_size), labels_tensor.view(-1))

            answer_token_count = sum(1 for label_id in labels if label_id != IGNORE_INDEX)
            total_loss += loss.item()
            total_answer_token_count += answer_token_count

    if total_answer_token_count == 0:
        raise ValueError(
            "perplexity를 계산할 답변 토큰이 없습니다 (qa_pairs가 비어있거나 전부 스킵됨)."
        )

    average_loss_per_token = total_loss / total_answer_token_count
    return math.exp(average_loss_per_token)


def evaluate_exact_match(
    model,
    tokenizer,
    qa_pairs: list[tuple[str, str]],
    max_new_tokens: int = 15,
) -> dict:
    """
    qa_pairs 각각에 대해 model.generate()로 답변을 생성하고,
    slot copy 정확도와 label 정확도를 독립적으로 집계한다.

    generate() 호출 시 temperature=1e-8, top_k=1로 고정하는 이유:
        평가는 매 실행마다 같은 결과가 재현되어야 하므로(deterministic에 가깝게),
        무작위성을 최대한 줄여 사실상 greedy decoding에 근접한 결과를 유도한다.

    Known Limitation (2026.07.21 기준, 실제 generate() 코드 확인 후 확정):
        transformer_model.py의 generate()는 `logits = logits / temperature`
        연산 뒤 torch.multinomial()로 항상 확률적 샘플링을 하며, greedy decoding
        분기(argmax)가 코드에 없다. temperature=0.0을 그대로 넘기면 0으로 나누기가
        발생해 logits가 inf/nan이 되고 RuntimeError가 발생함을 실제로 확인했다
        (2026.07.21 실행 로그 기준). 이에 따라 0 대신 1e-8을 사용한다. 이 값은
        확률 분포를 1등 토큰에 거의 100% 수렴시키므로 실질적으로는 greedy와
        거의 동일한 결과를 내지만, 여전히 torch.multinomial() 기반 샘플링이라
        완전한 재현성(bit-exact determinism)까지는 보장하지 않는다. 완전한
        재현성이 필요해지면, generate()에 argmax 기반 greedy 분기를 추가하는
        근본 수정(별도 과제)이 필요하다.

    Returns:
        {
            "slot_copy_accuracy": float,
            "label_accuracy": float,
            "details": list[dict],  # 각 샘플의 판정 결과 (디버깅/오류 분석용)
        }
    """
    model.eval()
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]

    slot_copy_correct_count = 0
    label_correct_count = 0
    details = []

    with torch.no_grad():
        for question, gold_answer in qa_pairs:
            question_ids = tokenizer.encode(question)
            input_tensor = torch.tensor([question_ids])

            generated = model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=1e-8,
                top_k=1,
                eos_token_id=eos_id,
            )

            # generate()는 입력(question_ids)까지 포함해 반환하므로(누적 생성 방식),
            # 질문 길이만큼 앞부분을 잘라내고 답변에 해당하는 부분만 취한다.
            generated_ids = generated[0].tolist()[len(question_ids):]
            trimmed_ids = trim_after_eos(generated_ids, eos_token_id=eos_id)
            generated_answer = tokenizer.decode(trimmed_ids)

            gold_parsed = parse_generated_answer(gold_answer)
            predicted_parsed = parse_generated_answer(generated_answer)
            question_activity = extract_question_activity(question)

            is_label_correct = predicted_parsed["label"] == gold_parsed["label"]
            # slot copy 정확도는 "질문의 activity가 답변의 primary_activity로 그대로
            # 복사되었는가"만 본다. substitute_activity는 항등함수 대상이 아니므로
            # (질문에 없던 새 정보를 학습된 패턴에서 골라야 하는 분류에 가까움) 제외한다.
            is_slot_copy_correct = predicted_parsed["primary_activity"] == question_activity

            if is_label_correct:
                label_correct_count += 1
            if is_slot_copy_correct:
                slot_copy_correct_count += 1

            details.append({
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "question_activity": question_activity,
                "is_label_correct": is_label_correct,
                "is_slot_copy_correct": is_slot_copy_correct,
            })

    total_count = len(qa_pairs)
    return {
        "slot_copy_accuracy": slot_copy_correct_count / total_count if total_count else 0.0,
        "label_accuracy": label_correct_count / total_count if total_count else 0.0,
        "details": details,
    }