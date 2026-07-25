"""
langchain_pipeline 6·7단계: LLM 계층 (Gemma + custom_transformer 스위처블)

기존 rag_pipeline/generator.py의 LangChain 대응 모듈
(docs/LANGCHAIN_MIGRATION_PLAN.md 표 2 / 표 5, 6~7단계 / §4.3, §4.4).

기존 TextGenerator는 model_name에 따라 if/else로 분기하고, generate()/
generate_stream() 시그니처를 손으로 통일했다. LangChain에서는 두 백엔드를
각각 Runnable 인터페이스(BaseLLM/LLM)를 구현하는 독립된 객체로 만들면,
호출부의 분기 자체가 사라진다 — 어떤 Runnable이 와도 .invoke()/.stream()으로
동일하게 호출된다(§4.3). 이 모듈은 두 백엔드를 각각 만드는 "공장(factory)
함수"만 제공하고, 어느 것을 선택할지는 호출부(8단계 chain.py, 9단계 main.py)의
책임이다.

- Gemma  -> get_gemma_llm()      : HuggingFacePipeline로 wrapping (6단계)
- custom_transformer -> get_custom_transformer_llm() : LLM 서브클래스 (7단계)
"""
import time
from pathlib import Path
from threading import Thread
from typing import Any, Iterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_huggingface import HuggingFacePipeline

from paths import SRC_DIR
from custom_transformer.transformer_model import TransformerLanguageModel
from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer
from custom_transformer.model.utils.generation_utils import trim_after_eos


CUSTOM_TRANSFORMER_QA_DATA_PATH = (
    SRC_DIR / "custom_transformer" / "scripts" / "raw_data" / "korean_qa.txt"
)
CUSTOM_TRANSFORMER_WEIGHTS_PATH = (
    SRC_DIR / "custom_transformer" / "scripts" / "trained_model" / "korean_model.pt"
)


# ----------------------------------------------------------------------
# 공통 — 디바이스 선택 (기존 TextGenerator._init_gemma()의 로직과 동일)
# ----------------------------------------------------------------------
def select_device() -> str:
    """
    Apple Silicon(M3)에서는 MPS, 그 외 NVIDIA GPU 환경에서는 CUDA, 둘 다 없으면 CPU를 사용한다.
    기존 TextGenerator._init_gemma()의 디바이스 선택 로직을 그대로 옮긴 것이다.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------
# 6단계 — Gemma (HuggingFacePipeline)
# ----------------------------------------------------------------------
def format_gemma_chat_prompt(processor: Any, prompt: str) -> str:
    """
    Gemma의 chat template을 적용한다.
    (기존 TextGenerator._build_inputs()/_generate_gemma()의 messages 조립 로직과 동일.
     enable_thinking=False: 학습 목적상 thinking 과정 없이 바로 답변만 받기 위함.)

    [중요] HuggingFacePipeline(및 그 내부의 transformers.pipeline)은 일반 문자열
    prompt를 받으면 chat template을 자동으로 적용하지 않는다 — 공식 소스코드
    (HuggingFacePipeline._generate())를 직접 확인한 결과, self.pipeline(prompts, ...)에
    원본 문자열을 그대로 전달할 뿐이다. 따라서 chat template 적용은 이 함수처럼
    별도 단계로 분리해, get_gemma_llm()이 반환하는 체인 안에서 LLM 호출 "이전"에
    실행되도록 조립해야 한다 — HuggingFacePipeline 자체를 서브클래싱/수정하지 않고도
    기존 chat template 동작을 보존하는, 절충 없는 방법이다.

    Args:
        processor: AutoProcessor.from_pretrained(model_name)의 결과
        prompt: 5단계 prompt.py가 만든, 역할 prefix 없는 순수 지시문 텍스트

    Returns:
        chat template이 적용된, 모델에 바로 입력할 수 있는 텍스트
    """
    messages = [{"role": "user", "content": prompt}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def get_gemma_llm(
    model_name: str = "google/gemma-4-E2B-it",
    max_new_tokens: int = 80,
) -> Runnable:
    """
    Gemma를 HuggingFacePipeline로 wrapping한 Runnable을 만든다.

    모델/프로세서 로딩과 디바이스 배치는 기존 TextGenerator._init_gemma()의 코드를
    그대로 옮긴 것이다(표 5, 6단계: "기존 MPS 디바이스 선택 로직 유지"). transformers의
    pipeline()을 직접 만들어 HuggingFacePipeline(pipeline=...)에 주입하는 방식은
    LangChain 공식 문서가 제시하는 두 가지 생성 방법 중 하나다(from_model_id()는
    device 인자가 CUDA GPU 인덱스(int)만 지원하고 MPS를 지원하지 않으므로
    — 공식 소스코드로 직접 확인됨 — 기존 MPS 로직을 유지하려면 이 방식을 쓸 수 없다).

    Args:
        model_name: HuggingFace 모델 이름 (기본값: Gemma 4 E2B-it)
        max_new_tokens: 생성할 최대 토큰 수

    Returns:
        문자열 prompt를 입력받아 chat template 적용 -> Gemma 생성까지 한 번에
        수행하는 Runnable. .invoke(prompt)는 새로 생성된 텍스트만 반환하고(기존
        _generate_gemma()와 동일하게 prompt 부분은 제외), .stream(prompt)는
        TextIteratorStreamer 기반의 진짜 토큰 단위 스트리밍 청크를 yield한다
        (HuggingFacePipeline._stream()이 이미 구현 — 표 3).
    """
    device = select_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # type: ignore

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=processor.tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    # skip_prompt=True를 호출마다 자동으로 적용하도록 bind한다.
    # - _stream()의 기본값은 이미 skip_prompt=True (공식 소스코드 확인됨) 라서 영향 없음.
    # - _generate()(.invoke()가 사용)의 기본값은 skip_prompt=False라서, bind 없이는
    #   prompt 전체가 포함된 텍스트가 반환된다 — 기존 _generate_gemma()의
    #   "새로 생성된 부분만 반환" 동작과 어긋나므로 명시적으로 True로 고정한다.
    llm = llm.bind(skip_prompt=True)

    chat_template_step = RunnableLambda(lambda prompt: format_gemma_chat_prompt(processor, prompt))
    return chat_template_step | llm



# ----------------------------------------------------------------------
# 7단계 — custom_transformer (LLM 서브클래스)
# ----------------------------------------------------------------------
def _load_qa_pairs(path: Path) -> list[tuple[str, str]]:
    """기존 TextGenerator._load_qa_pairs()/scripts/train.py의 load_qa_pairs()와 동일한 로직."""
    qa_pairs = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            question, answer = parts
            qa_pairs.append((question, answer))
    return qa_pairs


# ----------------------------------------------------------------------
# 8단계 — Qwen2.5-1.5B-Instruct GGUF (LlamaCpp)
# ----------------------------------------------------------------------
_QWEN_GGUF_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_QWEN_GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

def format_qwen_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """
    Qwen의 chat template을 적용한다.
    (기존 format_gemma_chat_prompt()와 동일한 역할.)

    Args:
        tokenizer: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")의 결과
        prompt: 5단계 prompt.py가 만든, 역할 prefix 없는 순수 지시문 텍스트

    Returns:
        chat template이 적용된, 모델에 바로 입력할 수 있는 텍스트
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def get_qwen_gguf_llm(max_new_tokens: int = 80) -> Runnable:
    """
    Qwen2.5-1.5B-Instruct GGUF(Q4_K_M)를 LlamaCpp로 wrapping한 Runnable을 만든다.

    GGUF는 transformers의 AutoModelForCausalLM으로 직접 로딩할 수 없는 llama.cpp
    전용 포맷이라, langchain_community.llms.LlamaCpp를 쓴다. 이미 LangChain의
    LLM(Runnable) 인터페이스를 구현하고 있어 CustomTransformerLLM처럼 별도
    서브클래싱이 필요 없다.

    Args:
        max_new_tokens: 생성할 최대 토큰 수

    Returns:
        문자열 prompt를 입력받아 chat template 적용 -> Qwen GGUF 생성까지 한 번에
        수행하는 Runnable. .invoke(prompt)는 생성된 텍스트를 반환하고,
        .stream(prompt)는 토큰 단위 스트리밍 청크를 yield한다 (get_gemma_llm()과
        동일한 인터페이스).
    """
    from huggingface_hub import hf_hub_download
    from langchain_community.llms import LlamaCpp
    from transformers import AutoTokenizer
    import os as _os

    model_path = hf_hub_download(
        repo_id=_QWEN_GGUF_REPO_ID,
        filename=_QWEN_GGUF_FILENAME,
    )

    # chat template 적용에는 tokenizer만 필요하다(전체 모델 가중치는 불필요) —
    # 실제 생성 실행 자체는 LlamaCpp(GGUF 가중치)가 담당한다.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        n_threads=_os.cpu_count(),
        max_tokens=max_new_tokens,
        temperature=0.0,
        streaming=True,
    )

    chat_template_step = RunnableLambda(lambda prompt: format_qwen_chat_prompt(tokenizer, prompt))
    return chat_template_step | llm

class CustomTransformerLLM(LLM):
    """
    custom_transformer 패키지(직접 구현한 TransformerLanguageModel)를 LangChain의
    LLM 인터페이스로 감싼 서브클래스.

    기존 TextGenerator의 분기 로직(if model_name == "custom_transformer": ...)이
    사라지고(§4.3), 이 클래스 자체가 하나의 독립된 Runnable이 된다 — 호출부는
    Gemma든 이 클래스든 동일하게 .invoke()/.stream()으로 다룬다.

    모델/tokenizer 로딩(비용이 큰 작업, QA 데이터로 BPE tokenizer를 매번 재학습해야 함)은
    pydantic 필드 검증 안에서 수행하기에 부적합하므로, 기존 TextGenerator._init_custom_transformer()의
    로직을 from_pretrained() classmethod로 그대로 옮긴다 — HuggingFacePipeline.from_model_id()와
    동일한 "공식 팩토리 메서드" 패턴이다.

    _call(): 기존 _generate_custom_transformer()의 로직을 그대로 옮김 (인코딩 -> truncate ->
        generate -> eos 이후 제거 -> 디코딩).
    _stream(): 기존 _generate_stream_custom_transformer()의 "단어 단위 가짜 스트리밍" 로직을
        GenerationChunk yield + run_manager.on_llm_new_token() 호출 형태로 옮김 — 이것이
        LangChain이 "토큰을 실시간으로 만들지 못하는 모델을 임의 단위로 잘라 스트리밍처럼
        보이게 하는" 표준 확장 지점 사용법이다(§4.4, 표 3 — HuggingFacePipeline._stream()의
        공식 소스코드도 동일한 GenerationChunk + on_llm_new_token() 패턴을 따른다).

    [중요] LLM(pydantic 모델)의 필드는 클래스 차원에서 타입 선언이 필요하다 — 기존
    TextGenerator.__init__()처럼 self.x = ...로 임의 속성을 추가할 수 없다(extra="forbid"
    제약). model/tokenizer/eos_token_id 등을 모두 필드로 선언한 이유다.
    """

    model: Any = None
    tokenizer: Any = None
    eos_token_id: int = 0
    max_input_tokens: int = 400  # 기존 TextGenerator.CUSTOM_MAX_INPUT_TOKENS과 동일
    default_max_new_tokens: int = 80  # 기존 generate()의 max_new_tokens 기본값과 동일

    @property
    def _llm_type(self) -> str:
        return "custom_transformer"

    @property
    def _identifying_params(self) -> dict:
        return {"model_type": "custom_transformer", "max_input_tokens": self.max_input_tokens}

    @classmethod
    def from_pretrained(
        cls,
        qa_data_path: Path = CUSTOM_TRANSFORMER_QA_DATA_PATH,
        weights_path: Path = CUSTOM_TRANSFORMER_WEIGHTS_PATH,
    ) -> "CustomTransformerLLM":
        """
        기존 TextGenerator._init_custom_transformer()의 로딩 절차를 그대로 옮긴 팩토리 메서드.

        (주의 사항 — 기존 모듈에서 이미 명시된 한계, 그대로 승계함) tokenizer 자체를
        저장/로드하지 않으므로, 매번 같은 QA 데이터로 BPE tokenizer를 재학습해야
        scripts/train.py가 만든 vocab(token_to_id)과 동일한 결과가 나온다.

        Raises:
            FileNotFoundError: 학습된 가중치 파일(weights_path)이 없을 경우
        """
        qa_pairs = _load_qa_pairs(qa_data_path)
        flat_corpus = [q for q, _ in qa_pairs] + [a for _, a in qa_pairs]

        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(flat_corpus)
        eos_token_id = tokenizer.token_to_id[tokenizer.eos_token]

        vocab_size = len(tokenizer.token_to_id)
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            max_len=512,
            dropout=0.1,
        )

        try:
            model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"학습된 모델 파일을 찾을 수 없습니다: {weights_path}. "
                "custom_transformer/scripts/train.py를 먼저 실행해서 모델을 학습/저장해야 합니다."
            ) from exc
        model.eval()

        return cls(model=model, tokenizer=tokenizer, eos_token_id=eos_token_id)

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """기존 TextGenerator._generate_custom_transformer()의 로직과 동일."""
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

        max_new_tokens = kwargs.get("max_new_tokens", self.default_max_new_tokens)

        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) > self.max_input_tokens:
            input_ids = input_ids[-self.max_input_tokens :]

        input_length = len(input_ids)
        input_tensor = torch.tensor([input_ids])

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                eos_token_id=self.eos_token_id,
            )

        generated_ids = generated[0].tolist()
        new_token_ids = generated_ids[input_length:]
        new_token_ids = trim_after_eos(new_token_ids, eos_token_id=self.eos_token_id)

        return self.tokenizer.decode(new_token_ids).strip()

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        기존 TextGenerator._generate_stream_custom_transformer()의 "단어 단위 가짜
        스트리밍"을 GenerationChunk yield 형태로 옮긴 것 (§4.4, 표 3).

        _call()(이미 검증된 비스트리밍 경로)을 그대로 재사용해서 전체 텍스트를 먼저
        만들고, 단어 단위로 잘라 순차적으로 yield한다 — model.generate()가 전체
        시퀀스를 한 번에 만든 뒤 반환하는 구조라, 토큰이 만들어지는 즉시 스트리밍할
        방법이 아직 없기 때문이다 (기존 모듈의 주석과 동일한 한계).
        """
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

        fake_stream_delay = kwargs.get("fake_stream_delay", 0.0)

        full_text = self._call(prompt, stop=stop, run_manager=None, **kwargs)
        if not full_text:
            # [중요 - 실제 실행으로 확인된 LangChain의 공식 제약]
            # BaseLLM.stream()(공개 Runnable 메서드)은 내부적으로 _stream()이
            # 단 하나의 GenerationChunk도 yield하지 않으면
            # ValueError("No generation chunks were returned")를 직접 발생시킨다
            # (langchain_core.language_models.llms.BaseLLM.stream() 소스코드에서
            # 직접 확인함 - "if generation is None: raise ValueError(...)").
            # 기존 _generate_stream_custom_transformer()는 빈 결과에서 그냥
            # 조용히 return(아무 것도 yield하지 않음)했지만, 그 방식은 LangChain의
            # LLM 서브클래스 계약과 맞지 않는다. 따라서 빈 문자열 청크 1개를
            # yield해서 "생성된 내용이 없음"을 표현하되, .stream() 자체는
            # 에러 없이 끝나도록 한다.
            chunk = GenerationChunk(text="")
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            return

        words = full_text.split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            if fake_stream_delay > 0:
                time.sleep(fake_stream_delay)

            chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk



def get_custom_transformer_llm(
    qa_data_path: Path = CUSTOM_TRANSFORMER_QA_DATA_PATH,
    weights_path: Path = CUSTOM_TRANSFORMER_WEIGHTS_PATH,
) -> CustomTransformerLLM:
    """get_gemma_llm()과 대칭을 이루는 얇은 팩토리 함수.

    호출부(8단계 chain.py, 9단계 main.py)가 "CustomTransformerLLM이라는 클래스가
    있고, classmethod로 생성한다"는 세부 사항을 몰라도 되게 한다 - Gemma 쪽
    get_gemma_llm()과 동일하게 "함수 호출 한 번으로 Runnable을 얻는다"는
    인터페이스를 유지한다.
    """
    return CustomTransformerLLM.from_pretrained(qa_data_path, weights_path)



if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.vector_store import build_vector_store, get_retriever
    from langchain_pipeline.prompt import format_docs, get_prompt_template

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)
    retriever = get_retriever(store, k=3)

    question = "DaySync의 내부 코드네임은 무엇인가?"
    retrieved_docs = retriever.invoke(question)
    prompt_value = get_prompt_template().invoke(
        {"context": format_docs(retrieved_docs), "question": question}
    )
    prompt_text = prompt_value.to_messages()[0].content

    print(f"[질문] {question}\n")

    # 두 백엔드 중 하나를 선택 - 호출부는 어느 쪽이든 동일하게 .invoke()/.stream()으로 다룬다.
    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 다운로드가 필요합니다]")
    llm = get_gemma_llm()
    # print("[custom_transformer 로딩 중... korean_qa.txt로 BPE tokenizer를 재학습합니다]")
    # llm = get_custom_transformer_llm()

    print("[비스트리밍 답변] ", end="", flush=True)
    print(llm.invoke(prompt_text))

    print("\n[스트리밍 답변] ", end="", flush=True)
    for chunk in llm.stream(prompt_text):
        print(chunk, end="", flush=True)
    print()
