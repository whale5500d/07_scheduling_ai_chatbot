"""
Level 1: Context Recall (LLM-as-a-Judge 기반)

목표:
- 정답에 필요한 핵심 정보(Ground Truth) 각각이, 검색된 context로부터
  추론 가능한지를 LLM이 의미적으로 판단하여 Context Recall 점수를 계산

Level 0과의 차이:
- Level 0(키워드 기반)은 Ground Truth 문장의 단어가 context에 그대로
  등장하는지만 보았다. Ground Truth가 한국어 문장으로 작성된 경우,
  영어 context와는 단어 자체가 겹치지 않아 항상 매칭에 실패하는 구조적
  한계가 있었다. (DaySync 도메인 전환으로 context 자체도 한국어가 되어,
  이 구조적 한계가 더는 발생하지 않지만, 평가 방식 자체는 언어 표현과
  무관하게 의미를 판단해야 한다는 원칙을 유지한다.)
- Level 1은 LLM에게 "이 Ground Truth 정보가 context로부터 추론 가능한가"를
  언어 표현과 무관하게 의미적으로 판단하게 한다.
"""

from rag_pipeline.generator import TextGenerator


def evaluate_context_recall(
    question: str,
    retrieved_chunks: list[str],
    ground_truth: list[str],
    generator: TextGenerator,
) -> dict:
    """
    Context Recall을 LLM-as-a-Judge 방식으로 평가한다.

    각 Ground Truth 항목에 대해 "이 정보가 검색된 context로부터 추론
    가능한가"를 LLM이 예/아니오로 판단하고, 추론 가능하다고 판단된 항목의
    비율을 점수로 계산한다.

    Args:
        question: 사용자 질문
        retrieved_chunks: 검색된 context 리스트
        ground_truth: 정답에 필요한 핵심 정보 (문장 리스트)
        generator: 평가에 사용할 TextGenerator 인스턴스. 호출하는 쪽에서
                   미리 생성하여 여러 평가 함수가 모델 로딩을 공유할 수 있게 한다.

    Returns:
        {
            "context_recall_score": float,  # 0.0 ~ 1.0
            "judgments": list[dict],        # 각 ground truth 항목에 대한 판단 결과
        }
    """
    if not ground_truth:
        return {
            "context_recall_score": 0.0,
            "judgments": [],
        }

    combined_context = "\n\n".join(retrieved_chunks)

    judgments = []
    matched_count = 0

    for i, gt_sentence in enumerate(ground_truth):
        prompt = f"""당신은 주어진 정보가 context로부터 추론 가능한지 판단하는 평가자입니다.

Context:
{combined_context}

확인할 정보: {gt_sentence}

위 정보가 context로부터 추론 가능합니까? (표현 언어와 무관하게 의미만 판단하세요.)
정확히 한 단어로만 답하세요: 예 또는 아니오.
"""
        response = generator.generate(prompt, max_new_tokens=10)
        is_matched = response.strip().startswith("예")

        if is_matched:
            matched_count += 1

        judgments.append({
            "ground_truth_index": i,
            "ground_truth": gt_sentence,
            "is_matched": is_matched,
            "raw_response": response.strip(),
        })

    context_recall_score = matched_count / len(ground_truth)

    return {
        "context_recall_score": round(context_recall_score, 4),
        "judgments": judgments,
    }


if __name__ == "__main__":
    # Level 1 테스트용 예시 (Level 0과 동일한 데이터로 비교 가능하게 유지)
    question = "DaySync의 기본 API 포트는 무엇인가?"
    retrieved_chunks = [
        "DaySync의 일정 조회 API는 기본적으로 9221번 포트에서 서비스된다.",
        "개발 초기 단계에서는 내부적으로 프로젝트 새벽별(Project Dawnstar)이라는 코드네임으로 불렸다.",
    ]
    ground_truth = [
        "API 포트는 9221이다",
        "DaySync는 팀 일정 관리 시스템이다",
    ]

    generator = TextGenerator()
    result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

    print("=" * 60)
    print(f"[Context Recall Score] {result['context_recall_score']}")
    print("=" * 60)

    for j in result["judgments"]:
        status = "포함됨" if j["is_matched"] else "미포함"
        print(f"[{status}] {j['ground_truth']} (LLM 응답: {j['raw_response']!r})")