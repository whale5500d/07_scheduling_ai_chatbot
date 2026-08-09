"""
langchain_pipeline 10단계: LangSmith Tracing + Dataset 기반 평가
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 1 #4, 표 5 10단계, §8 / §9 (e))

[Tracing — 코드 변경 없음]
LangSmith Tracing은 LANGSMITH_TRACING=true / LANGSMITH_API_KEY 환경 변수만 설정하면
langchain_core.runnables.Runnable(LCEL) 호출이 자동으로 계측된다 — 공식 문서로 확인함
(https://docs.langchain.com/langsmith/trace-with-langchain, "No extra code is needed
to log a trace to LangSmith. Just run your LangChain code as you normally would.").
8단계 chain.py의 build_rag_chain()/build_answer_only_chain()이 반환하는 체인은 모두
LCEL Runnable이므로, 이 모듈은 Tracing을 위한 별도 코드를 추가하지 않는다. 실행 전
설정해야 하는 환경 변수는 이 파일 맨 아래 __main__ 블록의 안내 메시지를 참고.

[평가 — 기존 4개 평가 함수를 재사용, 새로 만들지 않음]
이 디렉터리(tests/evaluate/)에 이미 구현·테스트되어 있는 4개 함수
(evaluate_faithfulness / evaluate_answer_relevancy / evaluate_context_precision /
evaluate_context_recall, 모두 RAGAS 없이 LLM-as-a-Judge 방식으로 직접 구현한 것)를
그대로 가져와, LangSmith evaluate()가 기대하는 evaluator 시그니처
`(inputs: dict, outputs: dict, reference_outputs: dict) -> dict`로 얇게 래핑한다 —
공식 문서 패턴(https://docs.langchain.com/langsmith/evaluate-llm-application) 그대로다.
4개 중 evaluate_context_recall만 데이터셋의 reference_outputs(ground_truth)를
필요로 하고, 나머지 3개는 question + target()의 출력(answer/retrieved_chunks)만으로
충분하다 — 4개 함수의 소스코드를 직접 읽고 확인한 사실이다(reference-free metric).

[왜 src/langchain_pipeline/이 아니라 tests/evaluate/에 두는가]
이 모듈이 감싸는 4개 평가 함수가 이미 tests/evaluate/에 있고, 평가는 런타임
서비스(main.py)가 의존하지 않는 별도 도구라는 점에서 evaluate_*.py들과 성격이
같다. src/가 tests/를 import하는 역방향 의존을 만들지 않기 위해, 이 모듈도 같은
디렉터리에 둔다(docs/LANGCHAIN_MIGRATION_PLAN.md 표 5 10단계 "대상" 칸 참고).

[평가 대상 — langchain_pipeline만, rag_pipeline은 평가 범위 밖]
표 1 #4 결정에 따라 langchain_pipeline.chain.build_rag_chain()만을 대상으로 한다
(rag_pipeline과의 비교 평가는 범위 밖).

[골든 데이터셋 — 새로 지어내지 않고, 기존 설계 문서를 그대로 옮김]
EVAL_GOLDEN_DATASET은 data/00_Design_Specification.md의 "RAG 검증용 핵심 정보
(Verification Anchors)" 표(6개 항목)를 그대로 옮긴 것이다. daysync_manual.md
한 문서만으로 답할 수 있는 질문만 포함했다 — src/main.py 소스코드로 직접 확인한
대로, langchain_pipeline이 실제로 색인하는 문서는 daysync_manual.md뿐이고
data/daysync_team_records.md는 GraphRAG 전용(이번 마이그레이션 범위 밖, 표 1 #2)이라
langchain_pipeline의 검색 경로로는 도달할 수 없기 때문이다.

[알려진 한계]
evaluate_faithfulness()는 다른 3개 함수와 달리 generator를 주입받지 않고 함수
내부에서 TextGenerator()를 직접 생성한다(기존 함수 시그니처를 그대로 유지했으므로
이 모듈에서 변경하지 않았다). 따라서 evaluate()가 example마다 faithfulness_evaluator를
호출할 때마다 judge 모델이 매번 새로 생성된다 — 다른 3개(judge_generator() 공유)와
다른 비용 구조다. 성능이 문제가 되면 evaluate_faithfulness()에 generator 주입
파라미터를 추가하는 리팩토링을 별도 과제로 분리한다(기존 baseline 함수를 이번
작업 범위에서 임의로 변경하지 않기 위함).
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import Client
from langsmith import evaluate as langsmith_evaluate

from rag_pipeline.generator import TextGenerator

from .evaluate_answer_relevancy import evaluate_answer_relevancy
from .evaluate_context_precision import evaluate_context_precision
from .evaluate_context_recall import evaluate_context_recall
from .evaluate_faithfulness import evaluate_faithfulness


# ----------------------------------------------------------------------
# 골든 데이터셋 (data/00_Design_Specification.md의 Verification Anchors 표 그대로)
# ----------------------------------------------------------------------
EVAL_GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "inputs": {"question": "DaySync의 내부 코드네임은 무엇인가?"},
        "outputs": {
            "ground_truth": [
                "DaySync의 내부 코드네임은 프로젝트 새벽별(Project Dawnstar)이다."
            ]
        },
    },
    {
        "inputs": {"question": "preference_threshold의 기본값은 무엇인가?"},
        "outputs": {"ground_truth": ["preference_threshold의 기본값은 0.65이다."]},
    },
    {
        "inputs": {"question": "추천 주기(recommendation_cycle)의 기본값은 며칠인가?"},
        "outputs": {"ground_truth": ["추천 주기의 기본값은 7일이다."]},
    },
    {
        "inputs": {"question": "DaySync의 기본 API 포트는 무엇인가?"},
        "outputs": {"ground_truth": ["DaySync의 기본 API 포트는 9221이다."]},
    },
    {
        "inputs": {"question": "충돌 코드 SC-114는 무엇을 의미하는가?"},
        "outputs": {
            "ground_truth": [
                "SC-114는 동일 요일에 거절 응답과 수락 응답이 동시에 기록된 경우를 의미한다."
            ]
        },
    },
    {
        "inputs": {"question": "일정 충돌이 발생했을 때 어떤 응답이 최종 값으로 채택되는가?"},
        "outputs": {"ground_truth": ["가장 마지막에 들어온 응답이 최종 값으로 채택된다."]},
    },
]

DATASET_NAME = "daysync-langchain-pipeline-golden-v1"


def ensure_dataset(client: Client, dataset_name: str = DATASET_NAME) -> None:
    """골든 데이터셋이 LangSmith에 이미 있으면 아무것도 하지 않고, 없으면
    생성 후 EVAL_GOLDEN_DATASET을 업로드한다 (재실행 시 중복 업로드 방지).

    공식 문서(evaluate-llm-application)가 보여주는
    "has_dataset() 확인 -> 없으면 create_dataset()+create_examples()" 패턴 그대로다.
    """
    if client.has_dataset(dataset_name=dataset_name):
        return
    client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_name=dataset_name, examples=EVAL_GOLDEN_DATASET)


# ----------------------------------------------------------------------
# target() — langchain_pipeline.chain.build_rag_chain()을 evaluate()가 요구하는
# `(inputs: dict) -> dict` 시그니처로 래핑
# ----------------------------------------------------------------------
def build_target(store: InMemoryVectorStore, llm: Runnable, k: int = 3) -> Callable[[dict], dict]:
    """8단계 build_rag_chain(store, llm, k)을 그대로 호출하는 target 함수를 만든다.

    store/llm을 주입받는 이유는 chain.py의 build_rag_chain()과 동일 — 어느
    백엔드(Gemma/custom_transformer)든, 테스트에서는 어떤 fake든 동일하게 동작해야
    하기 때문이다.
    """
    from langchain_pipeline.chain import build_rag_chain

    chain = build_rag_chain(store, llm, k=k)

    def target(inputs: dict) -> dict:
        return chain.invoke(inputs["question"])

    return target


# ----------------------------------------------------------------------
# evaluator 어댑터 — 기존 4개 평가 함수를 LangSmith evaluator 시그니처로 변환
# ----------------------------------------------------------------------
_judge_generator_singleton: Optional[TextGenerator] = None


def _judge_generator() -> TextGenerator:
    """answer_relevancy/context_precision/context_recall이 공유하는 judge LLM을
    단 한 번만 생성한다 (기존 tests/evaluate/test_evaluate_*.py의
    `@pytest.fixture(scope="module")` generator와 동일한 목적 — 모델 재로딩 비용 회피).
    """
    global _judge_generator_singleton
    if _judge_generator_singleton is None:
        _judge_generator_singleton = TextGenerator()
    return _judge_generator_singleton


def faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """evaluate_faithfulness()를 evaluate()가 요구하는 시그니처로 래핑한다."""
    question = inputs["question"]
    answer = outputs["answer"]
    retrieved_chunks = [chunk["text"] for chunk in outputs["retrieved_chunks"]]
    result = evaluate_faithfulness(question, answer, retrieved_chunks, _judge_generator())
    return {"key": "faithfulness", "score": result["faithfulness_score"]}


def answer_relevancy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """evaluate_answer_relevancy()를 evaluate()가 요구하는 시그니처로 래핑한다."""
    question = inputs["question"]
    answer = outputs["answer"]
    result = evaluate_answer_relevancy(question, answer, _judge_generator())
    return {"key": "answer_relevancy", "score": result["answer_relevancy_score"]}


def context_precision_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """evaluate_context_precision()을 evaluate()가 요구하는 시그니처로 래핑한다."""
    question = inputs["question"]
    retrieved_chunks = [chunk["text"] for chunk in outputs["retrieved_chunks"]]
    result = evaluate_context_precision(question, retrieved_chunks, _judge_generator())
    return {"key": "context_precision", "score": result["context_precision_score"]}


def context_recall_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """evaluate_context_recall()을 evaluate()가 요구하는 시그니처로 래핑한다.

    4개 중 유일하게 reference_outputs(데이터셋의 ground_truth)가 필요하다.
    """
    question = inputs["question"]
    retrieved_chunks = [chunk["text"] for chunk in outputs["retrieved_chunks"]]
    ground_truth = reference_outputs["ground_truth"]
    result = evaluate_context_recall(question, retrieved_chunks, ground_truth, _judge_generator())
    return {"key": "context_recall", "score": result["context_recall_score"]}


EVALUATORS = [
    faithfulness_evaluator,
    answer_relevancy_evaluator,
    context_precision_evaluator,
    context_recall_evaluator,
]


# ----------------------------------------------------------------------
# 오케스트레이션 — 골든셋 업로드 + langchain_pipeline 체인 조립 + evaluate() 실행
# ----------------------------------------------------------------------
def run_evaluation(
    client: Optional[Client] = None,
    dataset_name: str = DATASET_NAME,
    experiment_prefix: str = "langchain_pipeline-rag-eval",
) -> Any:
    """전체 평가 파이프라인을 실행한다.

    client를 주입받는 이유: 테스트에서 fake Client로 대체해 실제 네트워크 호출
    없이 ensure_dataset() 호출 여부/인자만 검증할 수 있게 하기 위함 — 이 모듈의
    다른 함수들(build_target의 store/llm)과 동일한 의존성 주입 패턴이다.
    """
    from paths import DATA_DIR
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.llm import get_gemma_llm
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.vector_store import build_vector_store

    client = client or Client()
    ensure_dataset(client, dataset_name)

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)

    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 다운로드가 필요합니다]")
    llm = get_gemma_llm()

    target = build_target(store, llm)

    return langsmith_evaluate(
        target,
        data=dataset_name,
        evaluators=EVALUATORS,
        client=client,
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )


if __name__ == "__main__":
    print("[안내] 실행 전 다음 환경 변수가 설정되어 있어야 합니다 (LangSmith 공식 문서):")
    print("  LANGSMITH_TRACING=true")
    print("  LANGSMITH_API_KEY=<발급받은 키>")
    print("  LANGSMITH_PROJECT=<선택, 미설정 시 'default' 프로젝트로 기록됨>")
    print("  (참고: 비-US 리전 계정은 LANGSMITH_ENDPOINT도 설정해야 합니다)")
    print()

    from dotenv import load_dotenv
    load_dotenv()

    results = run_evaluation()
    print(results)
