"""
Test for langchain_pipeline 10단계: LangSmith Tracing + Dataset 기반 평가
(tests/evaluate/langsmith_eval.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 10단계: "골든셋 구조·target() 출력
형태·4개 evaluator 어댑터의 입출력 매핑을 fake(가짜) client/judge로 검증"):
1. EVAL_GOLDEN_DATASET: data/00_Design_Specification.md의 Verification Anchors
   6개 항목과 1:1 대응하는 구조(question: str, ground_truth: list[str])를 갖는가
2. ensure_dataset(): 데이터셋이 이미 있으면 업로드를 건너뛰고, 없으면 정확히
   한 번 생성+업로드하는가 (재실행 시 중복 업로드 방지 로직 검증)
3. build_target(): build_rag_chain()을 그대로 호출해 {"question": str} ->
   {"answer", "retrieved_chunks"} 형태로 정확히 변환하는가
4. 4개 evaluator 어댑터: inputs/outputs/reference_outputs에서 올바른 필드를 꺼내
   기존 evaluate_*() 함수에 정확히 전달하고, {"key", "score"} 형태로 반환하는가
   (실제 judge 모델 호출 없이, 기존 함수를 monkeypatch로 대체해서 "연결"만 검증)

[가짜 LLM/임베딩 선택 근거]
tests/langchain_pipeline/test_chain.py와 동일하게, hand-rolled fake 대신
langchain_core의 공식 테스트 더블(DeterministicFakeEmbedding, FakeListLLM)을 쓴다 —
build_target()이 8단계 build_rag_chain()을 그대로 호출하는 얇은 래퍼일 뿐이므로,
같은 방식으로 검증하는 것이 일관적이다.

[실제 LangSmith 연결·실제 모델 로딩은 이 파일의 범위 밖]
run_evaluation()(Client() 실제 생성, get_gemma_llm() 실제 로딩, 실제 evaluate() 호출)은
의도적으로 테스트하지 않는다 — 실제 LangSmith API 키와 실제 모델 다운로드가 필요한
수동 실행 경로이기 때문이다(6단계 test_llm.py가 @pytest.mark.slow로 실제 모델
로딩을 분리한 것과 같은 이유). 대신 ensure_dataset()/build_target()/4개 evaluator
어댑터처럼, fake로 완전히 대체 가능한 "연결 로직"만 단위 테스트로 검증한다.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.vector_store import build_vector_store

from . import langsmith_eval
from .langsmith_eval import (
    DATASET_NAME,
    EVAL_GOLDEN_DATASET,
    answer_relevancy_evaluator,
    build_target,
    context_precision_evaluator,
    context_recall_evaluator,
    ensure_dataset,
    faithfulness_evaluator,
)


# ----------------------------------------------------------------------
# 1. EVAL_GOLDEN_DATASET 구조 검증
# ----------------------------------------------------------------------
class TestEvalGoldenDataset:
    """data/00_Design_Specification.md의 Verification Anchors 6개 항목과
    1:1 대응하는 골든셋 구조 검증."""

    def test_has_exactly_six_examples(self):
        """00_Design_Specification.md의 Verification Anchors 표는 정확히 6개 항목이다."""
        assert len(EVAL_GOLDEN_DATASET) == 6

    def test_each_example_has_inputs_question_and_outputs_ground_truth(self):
        for example in EVAL_GOLDEN_DATASET:
            assert set(example.keys()) == {"inputs", "outputs"}
            assert isinstance(example["inputs"]["question"], str)
            assert example["inputs"]["question"].strip() != ""
            assert isinstance(example["outputs"]["ground_truth"], list)
            assert len(example["outputs"]["ground_truth"]) > 0
            assert all(isinstance(g, str) and g.strip() for g in example["outputs"]["ground_truth"])

    def test_questions_are_unique(self):
        """같은 질문이 중복되면 평가 결과가 왜곡되므로, 6개 질문은 서로 달라야 한다."""
        questions = [example["inputs"]["question"] for example in EVAL_GOLDEN_DATASET]
        assert len(questions) == len(set(questions))


# ----------------------------------------------------------------------
# 2. ensure_dataset() — 재실행 시 중복 업로드 방지 검증 (fake Client)
# ----------------------------------------------------------------------
class _FakeClient:
    """langsmith.Client의 has_dataset/create_dataset/create_examples만 흉내내는
    가짜 client. 실제 네트워크 호출 없이 "호출 여부/인자"만 기록한다."""

    def __init__(self, dataset_already_exists: bool):
        self._dataset_already_exists = dataset_already_exists
        self.create_dataset_calls: list[str] = []
        self.create_examples_calls: list[tuple[str, list]] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self._dataset_already_exists

    def create_dataset(self, dataset_name: str, **kwargs):
        self.create_dataset_calls.append(dataset_name)

    def create_examples(self, *, dataset_name: str, examples: list):
        self.create_examples_calls.append((dataset_name, examples))


class TestEnsureDataset:
    def test_skips_upload_when_dataset_already_exists(self):
        client = _FakeClient(dataset_already_exists=True)

        ensure_dataset(client, dataset_name="any-name")

        assert client.create_dataset_calls == []
        assert client.create_examples_calls == []

    def test_creates_and_uploads_when_dataset_missing(self):
        client = _FakeClient(dataset_already_exists=False)

        ensure_dataset(client, dataset_name="any-name")

        assert client.create_dataset_calls == ["any-name"]
        assert len(client.create_examples_calls) == 1
        uploaded_name, uploaded_examples = client.create_examples_calls[0]
        assert uploaded_name == "any-name"
        assert uploaded_examples == EVAL_GOLDEN_DATASET

    def test_default_dataset_name_is_module_constant(self):
        client = _FakeClient(dataset_already_exists=False)

        ensure_dataset(client)

        assert client.create_dataset_calls == [DATASET_NAME]


# ----------------------------------------------------------------------
# 3. build_target() — build_rag_chain()을 evaluate()의 target 시그니처로 변환
# ----------------------------------------------------------------------
@pytest.fixture
def fake_embedding():
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def store(fake_embedding) -> InMemoryVectorStore:
    documents = [
        Document(page_content="DaySync의 기본 API 포트는 9221이다.", metadata={"idx": 0}),
        Document(page_content="추천 주기의 기본값은 7일이다.", metadata={"idx": 1}),
    ]
    return build_vector_store(documents, fake_embedding)


class TestBuildTarget:
    def test_target_returns_answer_and_retrieved_chunks(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        target = build_target(store, llm, k=2)

        result = target({"question": "기본 API 포트는 무엇인가?"})

        assert set(result.keys()) == {"answer", "retrieved_chunks"}
        assert result["answer"] == "가짜 답변입니다."
        assert len(result["retrieved_chunks"]) == 2

    def test_target_reads_question_from_inputs_dict(self, store):
        """target()은 evaluate()가 넘겨주는 dataset의 inputs(dict)에서 'question' 키를
        꺼내 chain.invoke(question: str)에 전달해야 한다 (evaluate()의 target 시그니처
        계약: target(inputs: dict) -> outputs: dict)."""
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        target = build_target(store, llm, k=1)

        # question 키가 없으면 KeyError가 나야 한다 — "다른 키를 보고 있지 않다"는 증거.
        with pytest.raises(KeyError):
            target({"wrong_key": "기본 API 포트는 무엇인가?"})


# ----------------------------------------------------------------------
# 4. evaluator 어댑터 — inputs/outputs/reference_outputs에서 올바른 필드를 꺼내
#    기존 evaluate_*() 함수에 정확히 전달하는지 검증 (monkeypatch로 실제 함수 대체)
# ----------------------------------------------------------------------
class TestFaithfulnessEvaluator:
    @pytest.mark.slow
    def test_extracts_question_answer_and_chunk_texts(self, monkeypatch):
        captured = {}

        def fake_evaluate_faithfulness(question, answer, retrieved_chunks):
            captured["question"] = question
            captured["answer"] = answer
            captured["retrieved_chunks"] = retrieved_chunks
            return {"faithfulness_score": 0.75, "judgments": [], "raw_response": ""}

        monkeypatch.setattr(langsmith_eval, "evaluate_faithfulness", fake_evaluate_faithfulness)

        result = faithfulness_evaluator(
            inputs={"question": "기본 API 포트는?"},
            outputs={
                "answer": "9221이다.",
                "retrieved_chunks": [{"text": "포트는 9221이다.", "score": 0.1}],
            },
            reference_outputs={"ground_truth": ["API 포트는 9221이다"]},
        )

        assert captured["question"] == "기본 API 포트는?"
        assert captured["answer"] == "9221이다."
        assert captured["retrieved_chunks"] == ["포트는 9221이다."]  # score는 제외, text만
        assert result == {"key": "faithfulness", "score": 0.75}


class TestAnswerRelevancyEvaluator:
    def test_extracts_question_answer_and_shares_judge_generator(self, monkeypatch):
        sentinel_generator = object()
        monkeypatch.setattr(langsmith_eval, "_judge_generator", lambda: sentinel_generator)

        captured = {}

        def fake_evaluate_answer_relevancy(question, answer, generator):
            captured["question"] = question
            captured["answer"] = answer
            captured["generator"] = generator
            return {"answer_relevancy_score": 1.0, "is_relevant": True, "raw_response": "예"}

        monkeypatch.setattr(langsmith_eval, "evaluate_answer_relevancy", fake_evaluate_answer_relevancy)

        result = answer_relevancy_evaluator(
            inputs={"question": "기본 API 포트는?"},
            outputs={"answer": "9221이다.", "retrieved_chunks": []},
            reference_outputs={"ground_truth": []},
        )

        assert captured["question"] == "기본 API 포트는?"
        assert captured["answer"] == "9221이다."
        assert captured["generator"] is sentinel_generator
        assert result == {"key": "answer_relevancy", "score": 1.0}


class TestContextPrecisionEvaluator:
    def test_extracts_question_and_chunk_texts_only(self, monkeypatch):
        monkeypatch.setattr(langsmith_eval, "_judge_generator", lambda: object())

        captured = {}

        def fake_evaluate_context_precision(question, retrieved_chunks, generator):
            captured["question"] = question
            captured["retrieved_chunks"] = retrieved_chunks
            return {"context_precision_score": 0.5, "judgments": []}

        monkeypatch.setattr(langsmith_eval, "evaluate_context_precision", fake_evaluate_context_precision)

        result = context_precision_evaluator(
            inputs={"question": "추천 주기는?"},
            outputs={
                "answer": "7일이다.",
                "retrieved_chunks": [
                    {"text": "추천 주기는 7일이다.", "score": 0.2},
                    {"text": "관계 없는 chunk.", "score": 0.9},
                ],
            },
            reference_outputs={"ground_truth": ["추천 주기는 7일이다"]},
        )

        assert captured["question"] == "추천 주기는?"
        assert captured["retrieved_chunks"] == ["추천 주기는 7일이다.", "관계 없는 chunk."]
        assert result == {"key": "context_precision", "score": 0.5}


class TestContextRecallEvaluator:
    def test_extracts_ground_truth_from_reference_outputs(self, monkeypatch):
        monkeypatch.setattr(langsmith_eval, "_judge_generator", lambda: object())

        captured = {}

        def fake_evaluate_context_recall(question, retrieved_chunks, ground_truth, generator):
            captured["question"] = question
            captured["retrieved_chunks"] = retrieved_chunks
            captured["ground_truth"] = ground_truth
            return {"context_recall_score": 1.0, "judgments": []}

        monkeypatch.setattr(langsmith_eval, "evaluate_context_recall", fake_evaluate_context_recall)

        result = context_recall_evaluator(
            inputs={"question": "추천 주기는?"},
            outputs={
                "answer": "7일이다.",
                "retrieved_chunks": [{"text": "추천 주기는 7일이다.", "score": 0.2}],
            },
            reference_outputs={"ground_truth": ["추천 주기는 7일이다"]},
        )

        assert captured["question"] == "추천 주기는?"
        assert captured["retrieved_chunks"] == ["추천 주기는 7일이다."]
        assert captured["ground_truth"] == ["추천 주기는 7일이다"]
        assert result == {"key": "context_recall", "score": 1.0}

    def test_does_not_touch_ground_truth_when_called_via_other_evaluators(self):
        """다른 3개 evaluator는 reference_outputs를 전혀 사용하지 않아야 한다
        (이미 evaluate_*()의 시그니처 자체로 보장되지만, 인자로 빈 dict를 넘겨도
        에러가 나지 않는다는 점으로 한 번 더 확인)."""
        # context_recall_evaluator만 reference_outputs["ground_truth"]에 접근하므로,
        # 빈 dict를 넘기면 KeyError가 나야 한다 — "필요한 곳에서는 실제로 사용한다"는 증거.
        with pytest.raises(KeyError):
            context_recall_evaluator(
                inputs={"question": "질문"},
                outputs={"answer": "답", "retrieved_chunks": []},
                reference_outputs={},
            )
