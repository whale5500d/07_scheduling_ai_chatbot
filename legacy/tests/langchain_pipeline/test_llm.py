"""
Test for langchain_pipeline 6,7단계: LLM 계층 (src/langchain_pipeline/llm.py)
"""

import pytest

from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer
from langchain_core.callbacks import BaseCallbackHandler

from langchain_pipeline.llm import (
    CustomTransformerLLM,
    _load_qa_pairs,
    format_gemma_chat_prompt,
    get_gemma_llm,
    select_device,
)


class _FakeProcessor:
    """실제 AutoProcessor 대신, apply_chat_template() 호출 인자만 기록하는 가짜 객체."""

    def __init__(self):
        self.received_messages = None
        self.received_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.received_messages = messages
        self.received_kwargs = kwargs
        return f"<applied>{messages[0]['content']}</applied>"


class TestFormatGemmaChatPrompt:
    """format_gemma_chat_prompt()에 대한 테스트 그룹 - 가짜 processor로 torch 없이 검증 가능."""

    def test_wraps_prompt_as_single_user_message(self):
        """prompt 텍스트가 user 역할의 단일 메시지로 감싸져 전달되어야 한다."""
        fake_processor = _FakeProcessor()

        format_gemma_chat_prompt(fake_processor, "질문 내용입니다")

        assert fake_processor.received_messages == [{"role": "user", "content": "질문 내용입니다"}]

    def test_passes_expected_chat_template_kwargs(self):
        """기존 _build_inputs()/_generate_gemma()와 동일한 kwargs(tokenize=False 등)로 호출되어야 한다."""
        fake_processor = _FakeProcessor()

        format_gemma_chat_prompt(fake_processor, "질문")

        assert fake_processor.received_kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }

    def test_returns_processor_output(self):
        """apply_chat_template()의 반환값을 그대로 돌려줘야 한다."""
        fake_processor = _FakeProcessor()

        result = format_gemma_chat_prompt(fake_processor, "질문")

        assert result == "<applied>질문</applied>"


class TestSelectDevice:
    """select_device()에 대한 테스트 그룹"""

    def test_returns_one_of_known_devices(self):
        """mps/cuda/cpu 중 하나를 반환해야 한다 (실행 환경에 따라 달라짐)."""
        assert select_device() in {"mps", "cuda", "cpu"}


@pytest.mark.slow
class TestGetGemmaLlmIntegration:
    """
    get_gemma_llm()에 대한 통합 테스트 - 실제 Gemma 모델을 로딩한다 (수 GB 다운로드 +
    느린 추론). 기본 테스트 실행에서는 제외하고 싶다면: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    @classmethod
    def gemma_llm(cls):
        return get_gemma_llm()

    def test_invoke_returns_string(self, gemma_llm):
        """invoke() 결과가 문자열이어야 한다."""
        result = gemma_llm.invoke("간단한 질문입니다. 답변:")

        assert isinstance(result, str)

    def test_stream_returns_multiple_chunks(self, gemma_llm):
        """stream() 결과가 단일 청크가 아니라 여러 청크로 도착해야 한다 (진짜 토큰 단위 스트리밍)."""
        chunks = list(gemma_llm.stream("간단한 질문입니다. 답변:"))

        assert len(chunks) > 1


# ----------------------------------------------------------------------
# 7단계 - CustomTransformerLLM
# ----------------------------------------------------------------------
class _FakeRow:
    """torch.Tensor의 한 행(row)을 대신하는 가짜 객체.

    CustomTransformerLLM._call()이 generated[0].tolist()를 호출하므로,
    그 호출만 만족시키면 된다 (실제 텐서일 필요는 없음).
    """

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


class _FakeCustomModel:
    """실제 TransformerLanguageModel 대신 쓰는 가짜 모델.

    생성된 입력 뒤에 미리 정해둔 continuation_ids를 이어붙여 반환하고,
    generate()에 어떤 인자로 호출되었는지를 기록해 둔다 - 테스트에서
    truncate/kwargs 전달 로직을 검증하는 데 쓰인다.
    """

    def __init__(self, continuation_ids):
        self.continuation_ids = list(continuation_ids)
        self.received_kwargs = None

    def generate(self, input_ids, max_new_tokens, temperature, top_k, eos_token_id):
        self.received_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "eos_token_id": eos_token_id,
        }
        full_sequence = list(input_ids[0]) + self.continuation_ids
        return [_FakeRow(full_sequence)]

    def eval(self):
        pass


class _RecordingCallbackHandler(BaseCallbackHandler):
    """run_manager.on_llm_new_token()이 실제로 호출되는지 기록하는 콜백 핸들러."""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)


@pytest.fixture(scope="module")
def trained_tokenizer():
    """작은 고정 코퍼스로 학습된 실제 BPETokenizer (가짜가 아님 - 토치 의존이 없는 순수
    파이썬 구현이라 이 샌드박스에서도 100% 동일하게 동작한다)."""
    tokenizer = BPETokenizer(vocab_size=60)
    tokenizer.train(
        ["this is a fairly long test sentence used only to train a tiny tokenizer for testing purposes"]
    )
    return tokenizer


@pytest.fixture
def eos_token_id(trained_tokenizer):
    return trained_tokenizer.token_to_id[trained_tokenizer.eos_token]


class TestCustomTransformerLLM:
    """CustomTransformerLLM._call()/_stream()에 대한 테스트 그룹.

    실제 모델 대신 _FakeCustomModel을 주입하므로, "모델이 좋은 답을 만드는가"는
    검증 대상이 아니다. 검증 대상은 llm.py 자신이 작성한 로직
    (truncate, eos 이후 제거, kwargs 전달, 단어 단위 스트리밍, run_manager 콜백 호출)이며,
    이는 기존 TextGenerator._generate_custom_transformer()/
    _generate_stream_custom_transformer()와 동일한 동작을 해야 한다.
    """

    def test_llm_type(self, trained_tokenizer, eos_token_id):
        llm = CustomTransformerLLM(model=_FakeCustomModel([]), tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        assert llm._llm_type == "custom_transformer"

    def test_identifying_params_includes_max_input_tokens(self, trained_tokenizer, eos_token_id):
        llm = CustomTransformerLLM(
            model=_FakeCustomModel([]),
            tokenizer=trained_tokenizer,
            eos_token_id=eos_token_id,
            max_input_tokens=123,
        )

        assert llm._identifying_params["max_input_tokens"] == 123

    def test_call_returns_decoded_text_trimmed_at_eos(self, trained_tokenizer, eos_token_id):
        """eos 이후에 남아 있는 토큰("test sentence")은 결과에 포함되면 안 된다."""
        continuation = trained_tokenizer.encode("tiny tokenizer") + [eos_token_id] + trained_tokenizer.encode(
            "test sentence"
        )
        fake_model = _FakeCustomModel(continuation)
        llm = CustomTransformerLLM(model=fake_model, tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        result = llm.invoke("train a tiny tokenizer")

        assert result == "tiny tokenizer"

    def test_call_truncates_input_to_max_input_tokens(self, trained_tokenizer, eos_token_id):
        """max_input_tokens를 넘는 입력은 뒤쪽(질문에 가까운 부분)만 남기고 잘려야 한다."""
        long_prompt = "this is a fairly long test sentence used only to train a tiny tokenizer"
        encoded = trained_tokenizer.encode(long_prompt)
        assert len(encoded) > 5, "이 테스트는 encoded 길이가 5보다 길어야 truncate를 검증할 수 있다"

        fake_model = _FakeCustomModel(trained_tokenizer.encode("tiny tokenizer") + [eos_token_id])
        llm = CustomTransformerLLM(
            model=fake_model,
            tokenizer=trained_tokenizer,
            eos_token_id=eos_token_id,
            max_input_tokens=5,
        )

        llm.invoke(long_prompt)

        received_input_ids = fake_model.received_kwargs["input_ids"][0]
        assert list(received_input_ids) == encoded[-5:]

    def test_call_raises_value_error_on_empty_prompt(self, trained_tokenizer, eos_token_id):
        llm = CustomTransformerLLM(model=_FakeCustomModel([]), tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        with pytest.raises(ValueError):
            llm.invoke("   ")

    def test_call_respects_max_new_tokens_override_kwarg(self, trained_tokenizer, eos_token_id):
        """기존 generate(prompt, max_new_tokens=...)처럼, 호출 시점에 max_new_tokens를
        덮어쓸 수 있어야 한다 (생성자에 박아 넣은 고정값이 아님)."""
        fake_model = _FakeCustomModel(trained_tokenizer.encode("tiny tokenizer") + [eos_token_id])
        llm = CustomTransformerLLM(
            model=fake_model, tokenizer=trained_tokenizer, eos_token_id=eos_token_id, default_max_new_tokens=80
        )

        llm.invoke("train a tiny tokenizer", max_new_tokens=5)

        assert fake_model.received_kwargs["max_new_tokens"] == 5

    def test_stream_yields_chunks_that_join_to_call_result(self, trained_tokenizer, eos_token_id):
        """단어 단위로 잘린 조각들을 이어붙이면 invoke()의 결과와 동일해야 한다
        (기존 _generate_stream_custom_transformer()의 핵심 불변식)."""
        continuation = trained_tokenizer.encode("tiny tokenizer") + [eos_token_id]
        fake_model = _FakeCustomModel(continuation)
        llm = CustomTransformerLLM(model=fake_model, tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        full_result = llm.invoke("train a tiny tokenizer")
        chunks = list(llm.stream("train a tiny tokenizer"))

        assert len(chunks) > 1
        assert "".join(chunks) == full_result

    def test_stream_raises_value_error_on_empty_prompt(self, trained_tokenizer, eos_token_id):
        llm = CustomTransformerLLM(model=_FakeCustomModel([]), tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        with pytest.raises(ValueError):
            list(llm.stream("   "))

    def test_stream_yields_single_empty_chunk_when_generation_is_empty(self, trained_tokenizer, eos_token_id):
        """eos가 곧바로 등장하면(빈 생성), 빈 문자열 청크 1개를 yield해야 한다.

        [실제 실행으로 확인된 사실] BaseLLM.stream()은 _stream()이 단 하나의
        GenerationChunk도 만들지 않으면 ValueError("No generation chunks were
        returned")를 직접 발생시킨다(langchain_core.language_models.llms.BaseLLM.stream()
        소스코드에서 확인함). 처음에는 "조용히 아무 것도 yield하지 않기"로
        구현했다가 이 테스트에서 바로 그 ValueError가 실제로 발생하는 것을
        확인했고, 빈 문자열 청크 1개를 yield하도록 llm.py의 _stream()을 수정해서
        고쳤다 - 기존 _generate_stream_custom_transformer()(조용히 return)와의
        의도된 차이점이다."""
        fake_model = _FakeCustomModel([eos_token_id])
        llm = CustomTransformerLLM(model=fake_model, tokenizer=trained_tokenizer, eos_token_id=eos_token_id)

        chunks = list(llm.stream("train a tiny tokenizer"))

        assert chunks == [""]

    def test_stream_invokes_run_manager_on_llm_new_token(self, trained_tokenizer, eos_token_id):
        """LangChain의 표준 콜백 확장 지점(on_llm_new_token)이 실제로 호출되어야
        스트리밍 진행 상황을 다른 코드(예: WebSocket 전송)에 연결할 수 있다."""
        continuation = trained_tokenizer.encode("tiny tokenizer") + [eos_token_id]
        fake_model = _FakeCustomModel(continuation)
        llm = CustomTransformerLLM(model=fake_model, tokenizer=trained_tokenizer, eos_token_id=eos_token_id)
        handler = _RecordingCallbackHandler()

        chunks = list(llm.stream("train a tiny tokenizer", config={"callbacks": [handler]}))

        assert handler.tokens == chunks


# ----------------------------------------------------------------------
# 7단계 - _load_qa_pairs()
# ----------------------------------------------------------------------
class TestLoadQaPairs:
    """_load_qa_pairs()에 대한 테스트 그룹 - 기존
    TextGenerator._load_qa_pairs()/scripts/train.py의 load_qa_pairs()와 동일한
    파싱 규칙(탭 구분, 빈 줄 무시, 칸이 2개가 아니면 무시)을 검증한다."""

    def test_parses_tab_separated_lines(self, tmp_path):
        qa_path = tmp_path / "qa.txt"
        qa_path.write_text("질문1\t답변1\n질문2\t답변2\n", encoding="utf-8")

        result = _load_qa_pairs(qa_path)

        assert result == [("질문1", "답변1"), ("질문2", "답변2")]

    def test_skips_blank_lines(self, tmp_path):
        qa_path = tmp_path / "qa.txt"
        qa_path.write_text("질문1\t답변1\n\n   \n질문2\t답변2\n", encoding="utf-8")

        result = _load_qa_pairs(qa_path)

        assert result == [("질문1", "답변1"), ("질문2", "답변2")]

    def test_skips_lines_without_exactly_two_parts(self, tmp_path):
        qa_path = tmp_path / "qa.txt"
        qa_path.write_text(
            "질문1\t답변1\n"
            "탭이 없는 줄\n"
            "질문2\t답변2\t여분컬럼\n"
            "질문3\t답변3\n",
            encoding="utf-8",
        )

        result = _load_qa_pairs(qa_path)

        assert result == [("질문1", "답변1"), ("질문3", "답변3")]
