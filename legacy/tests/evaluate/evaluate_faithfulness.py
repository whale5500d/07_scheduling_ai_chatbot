"""
Level 0: RAGAS 없이 Faithfulness를 가장 단순한 형태로 평가하는 스크립트

Root from Expansion 원칙에 따라, RAG 파이프라인과의 통합 없이
평가 로직만 독립적으로 구현한다.
"""

from rag_pipeline.generator import TextGenerator
from typing import Optional

def evaluate_faithfulness(
    question: str,
    answer: str,
    retrieved_chunks: list[str],
    generator: Optional[TextGenerator] = None
) -> dict:
    """
    생성된 답변의 Faithfulness(충실도)를 평가한다.
    
    Args:
        question: 사용자 질문
        answer: RAG가 생성한 답변
        retrieved_chunks: 검색된 context 리스트

    Returns:
        {
            "faithfulness_score": float,  # 0.0 ~ 1.0
            "judgments": list[dict]       # 각 주장에 대한 판단 결과
        }
    """
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks가 비어 있습니다.")

    # context를 하나의 문자열로 합침
    context = "\n\n".join(retrieved_chunks)

    # 평가용 프롬프트 (가장 단순한 형태)
    prompt = f"""당신은 주어진 답변이 context에 충실한지 확인하는 평가자입니다.

                    질문: {question}

                    Context:
                    {context}

                    답변: {answer}

                    답변을 개별 주장(claim)으로 나누고, 각 주장을 판단하세요.
                    각 주장에 대해 다음 형식으로 답하세요:
                    주장: <주장 내용>
                    판단: 예 또는 아니오
                    이유: <간단한 이유>

                    위 Context에 있는 정보만 사용하세요. 외부 지식을 사용하지 마세요.
                """

    generator = generator or TextGenerator()
    response = generator.generate(prompt, max_new_tokens=300)

    # 간단한 파싱 (실제로는 더 견고한 파싱이 필요하지만, Level 0에서는 단순화)
    judgments = []
    lines = response.strip().split("\n")
    current_claim = None

    for line in lines:
        line = line.strip()
        if line.startswith("주장:"):
            current_claim = line.replace("주장:", "").strip()
        elif line.startswith("판단:") and current_claim:
            judgment = line.replace("판단:", "").strip()
            is_supported = judgment.startswith("예")
            judgments.append({
                "claim": current_claim,
                "is_supported": is_supported,
                "judgment": judgment
            })
            current_claim = None

    # Faithfulness 점수 계산
    if judgments:
        supported_count = sum(1 for j in judgments if j["is_supported"])
        faithfulness_score = supported_count / len(judgments)
    else:
        faithfulness_score = 0.0

    return {
        "faithfulness_score": round(faithfulness_score, 3),
        "judgments": judgments,
        "raw_response": response
    }


if __name__ == "__main__":
    # Level 0 테스트용 하드코딩 데이터 (debug_retrieval.py에서 검증된 질문)
    question = "DaySync의 내부 코드네임은 무엇인가?"
    answer = "DaySync의 내부 코드네임은 프로젝트 새벽별(Project Dawnstar)이었다."
    retrieved_chunks = [
        "개발 초기 단계에서는 내부적으로 \"프로젝트 새벽별(Project Dawnstar)\"이라는 코드네임으로 불렸다.",
        "DaySync는 팀원들의 일정과 활동 선호도를 관리하기 위해 사내에서 자체 개발한 일정 관리 시스템이다.",
    ]

    result = evaluate_faithfulness(question, answer, retrieved_chunks)

    print("=" * 60)
    print(f"[Faithfulness Score] {result['faithfulness_score']}")
    print("=" * 60)
    for i, j in enumerate(result["judgments"], 1):
        print(f"\n[Claim {i}] {j['claim']}")
        print(f"Judgment: {j['judgment']}")
    print("\n[Raw Response]")
    print(result["raw_response"])