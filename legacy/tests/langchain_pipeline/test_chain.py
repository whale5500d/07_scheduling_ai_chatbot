"""
Test for langchain_pipeline 8단계: chain.py (LCEL 체인 조립) (src/langchain_pipeline/chain.py)

검증 항목 (docs/LANGCHAIN_MIGRATION_PLAN.md 표 5, 8단계: "invoke() 결과가 기존
main.py /query 응답과 동일한 형태인지 통합 검증"):
1. build_rag_chain(): invoke() 결과가 {"answer": str, "retrieved_chunks": [{"text", "score"}, ...]}
   형태인가 (기존 QueryResponse와 동일한 형태인가)
2. build_rag_chain(): k가 검색 결과 개수를 실제로 제한하는가
3. build_answer_only_chain(): invoke()가 답변 문자열만 반환하는가 (기존 query_stream()과
   동일하게 retrieved_chunks를 포함하지 않는가)
4. build_answer_only_chain(): stream()이 실제로 여러 조각으로 나뉘어 오는가 (체인의 마지막
   단계(llm)만 진짜로 스트리밍되는, LCEL RunnableSequence.stream()의 표준 동작 확인)
5. 두 체인 모두, 검색된 문서로 조립된 prompt 텍스트가 5단계(prompt.py)의 포맷
   ("[문서 N]", "질문:", "답변:")을 그대로 담고 있는가 (echo LLM으로 prompt 본문을
   직접 검증 — "모양만 맞다"가 아니라 "내용이 실제로 올바르게 연결되었다"를 확인)
6. 저장소가 비어 있을 때, format_docs()의 기존 제약(§5단계 ValueError)이 체인을
   통과해도 그대로 전파되는가 (한계 확인 — 새로운 예외 처리를 추가하지 않았음을 명시)

[가짜 LLM 선택 근거]
hand-rolled fake 대신 langchain_core.language_models.fake의 공식 테스트 더블을 쓴다:
- FakeListLLM(responses=[...])         : invoke() 검증용 (고정된 응답을 그대로 반환)
- FakeStreamingListLLM(responses=[...]) : stream() 검증용 (응답 문자열을 한 글자씩 yield)
이 둘은 LangChain이 자체 테스트에서 쓰는 공식 컴포넌트이므로(공식 문서/소스코드
기반 신뢰), chain.py가 llm.py에 의존하지 않는다는 점(6·7단계의 LLM 구현은 chain.py의
관심사가 아니라 chain.py에 주입되는 "교체 가능한 Runnable"일 뿐이라는 §4.3 설계)을
역으로 증명한다 — torch/transformers 없이도 chain.py 테스트는 완전히 동작한다.

[echo LLM (RunnableLambda(lambda p: p))을 쓰는 테스트의 의도]
prompt 조립이 "실제로 올바른 내용"을 담고 있는지 검증하려면, LLM이 받은 입력을
그대로 들여다봐야 한다. FakeListLLM은 입력을 무시하고 고정 응답만 반환하므로
이 목적에는 맞지 않는다 — 그래서 "받은 그대로 돌려주는" RunnableLambda를 LLM
자리에 대신 넣어, build_rag_chain()/build_answer_only_chain()의 출력에서 곧바로
실제 prompt 텍스트를 확인한다.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.chain import build_answer_only_chain, build_rag_chain  # noqa: E402
from langchain_pipeline.vector_store import build_vector_store  # noqa: E402


@pytest.fixture
def fake_embedding():
    """가벼운 결정론적 가짜 임베딩 (4단계 test_vector_store.py와 동일한 패턴)."""
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The default API port is 8842.", metadata={"idx": 0}),
        Document(page_content="Conflicts are resolved by priority.", metadata={"idx": 1}),
        Document(page_content="NimbusFlow supports webhook retries.", metadata={"idx": 2}),
    ]


@pytest.fixture
def store(sample_documents, fake_embedding):
    return build_vector_store(sample_documents, fake_embedding)


@pytest.fixture
def echo_llm():
    """입력으로 받은 prompt 문자열을 그대로 반환하는 가짜 LLM — prompt 본문 검증용."""
    return RunnableLambda(lambda prompt: prompt)


class TestBuildRagChain:
    """build_rag_chain() — 기존 main.py POST /query 대응 체인."""

    def test_invoke_result_has_answer_and_retrieved_chunks_keys(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=2)

        result = chain.invoke("API 포트가 뭐야?")

        assert set(result.keys()) == {"answer", "retrieved_chunks"}

    def test_answer_is_llm_response(self, store):
        """answer 필드는 llm.invoke(prompt)의 결과여야 한다 (체인 연결 자체의 검증)."""
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=2)

        result = chain.invoke("API 포트가 뭐야?")

        assert result["answer"] == "가짜 답변입니다."

    def test_retrieved_chunks_each_have_text_and_score(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=2)

        result = chain.invoke("API 포트가 뭐야?")

        for chunk in result["retrieved_chunks"]:
            assert set(chunk.keys()) == {"text", "score"}
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["score"], float)

    def test_retrieved_chunks_count_matches_k(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=2)

        result = chain.invoke("API 포트가 뭐야?")

        assert len(result["retrieved_chunks"]) == 2

    def test_retrieved_chunks_text_comes_from_original_documents(self, store, sample_documents):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=3)
        original_contents = {doc.page_content for doc in sample_documents}

        result = chain.invoke("API 포트가 뭐야?")

        assert all(chunk["text"] in original_contents for chunk in result["retrieved_chunks"])

    def test_k_limits_retrieved_chunks_even_when_more_documents_exist(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(store, llm, k=1)

        result = chain.invoke("아무 질문")

        assert len(result["retrieved_chunks"]) == 1

    def test_prompt_passed_to_llm_contains_question_and_context_blocks(self, store, echo_llm):
        """echo_llm을 llm 자리에 넣어, answer 필드 == llm이 실제로 받은 prompt 본문임을 이용해
        prompt 조립(5단계 format_docs + get_prompt_template)이 실제로 올바르게
        연결되었는지 검증한다 (모양이 아니라 "내용")."""
        chain = build_rag_chain(store, echo_llm, k=2)
        question = "NimbusFlow의 webhook 재시도 정책은?"

        result = chain.invoke(question)
        prompt_text = result["answer"]

        assert f"질문: {question}" in prompt_text
        assert "답변:" in prompt_text
        assert "[문서 1]" in prompt_text
        assert "[문서 2]" in prompt_text
        assert "Human:" not in prompt_text  # 5단계에서 확인된 "Human: " 오염이 없어야 한다

    def test_empty_store_propagates_format_docs_value_error(self, fake_embedding):
        """저장소가 비어 있으면 검색 결과가 []이고, format_docs([])가 ValueError를 던진다
        (5단계 prompt.py의 기존 제약). chain.py는 이 예외를 새로 처리하지 않고 그대로
        전파한다 — 이것이 한계이자 의도된 동작이다 (8단계는 새로운 예외 처리 계층을
        추가하는 단계가 아니라, 4~7단계를 LCEL로 "연결"만 하는 단계이기 때문)."""
        empty_store = InMemoryVectorStore(embedding=fake_embedding)
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_rag_chain(empty_store, llm, k=3)

        with pytest.raises(ValueError):
            chain.invoke("아무 질문")


class TestBuildAnswerOnlyChain:
    """build_answer_only_chain() — 기존 main.py POST /query/stream 대응 체인."""

    def test_invoke_returns_plain_string(self, store):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_answer_only_chain(store, llm, k=2)

        result = chain.invoke("API 포트가 뭐야?")

        assert isinstance(result, str)
        assert result == "가짜 답변입니다."

    def test_stream_yields_multiple_chunks_that_join_to_full_answer(self, store):
        """체인의 마지막 단계(llm)만 진짜로 스트리밍되는지 확인한다. FakeStreamingListLLM은
        응답 문자열을 한 글자씩 yield하므로, len(chunks) > 1이면 .stream()이 체인
        전체를 한 번에 invoke()하듯 처리한 게 아니라 실제로 llm 단계까지 흘려보내
        스트리밍했다는 뜻이다."""
        full_answer = "스트리밍 테스트 답변"
        llm = FakeStreamingListLLM(responses=[full_answer])
        chain = build_answer_only_chain(store, llm, k=2)

        chunks = list(chain.stream("API 포트가 뭐야?"))

        assert len(chunks) > 1
        assert "".join(chunks) == full_answer

    def test_prompt_passed_to_llm_matches_rag_chain_prompt(self, store, echo_llm):
        """build_rag_chain()과 build_answer_only_chain()은 동일한 _prompt_text_from_documents()를
        공유하므로, 같은 질문/k에 대해 동일한 prompt 본문을 llm에 전달해야 한다
        (코드 중복 없이 한 곳에서만 prompt 조립 로직을 관리한다는 설계의 검증)."""
        question = "NimbusFlow의 webhook 재시도 정책은?"

        rag_chain = build_rag_chain(store, echo_llm, k=2)
        answer_only_chain = build_answer_only_chain(store, echo_llm, k=2)

        rag_prompt = rag_chain.invoke(question)["answer"]
        answer_only_prompt = answer_only_chain.invoke(question)

        assert rag_prompt == answer_only_prompt

    def test_empty_store_propagates_format_docs_value_error(self, fake_embedding):
        empty_store = InMemoryVectorStore(embedding=fake_embedding)
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        chain = build_answer_only_chain(empty_store, llm, k=3)

        with pytest.raises(ValueError):
            chain.invoke("아무 질문")
