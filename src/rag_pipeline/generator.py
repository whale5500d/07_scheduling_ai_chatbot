"""
Step B: Generation (+ Step C-3: 스트리밍)

책임(Responsibility): prompt(문자열)를 받아 LLM으로 응답 텍스트를 생성한다.
이 단계 이전의 Retrieval, Prompt Augmentation 결과를 입력으로 받는다.

model_name으로 두 가지 생성 모델을 전환(갈아끼우기)할 수 있다:
    - "google/gemma-4-E2B-it" (기본값) 등 HuggingFace 모델 이름: Gemma 4 E2B-it 사용
    - "custom_transformer": 직접 구현한 TransformerLanguageModel
      (custom_transformer/ 패키지, "일정 묻기" 도메인으로 instruction tuning됨) 사용

설계 원칙:
    - lifespan()은 "언제 생성기를 만들지"만 결정하고, "무엇을 어떻게 로딩할지"는
      이 클래스(TextGenerator)가 전부 캡슐화한다. 호출하는 쪽(main.py)은
      model_name 값만 바꾸면 되고, generate()/generate_stream() 시그니처는
      모델 종류와 무관하게 동일하다.
    - 커스텀 Transformer 쪽 로딩 절차는 기존 generate.py의
      _load_model_and_tokenizer()와 scripts/train.py의 절차를 그대로 따른다
      (동일 데이터로 tokenizer를 재학습해야 train.py가 만든 vocab과 같아짐).

전제 (커스텀 Transformer 사용 시, 중요):
    - 이 모델은 "오늘/내일 [활동] 할 거야?" 형태의 한국어 질문 24쌍으로만
      학습된 극소 규모 모델이다.
    - RAG_Project의 prompt_builder.build_prompt()가 만드는 prompt는
      영어로 된 긴 지시문 + 문서 청크 + 질문 구조이다.
    - 따라서 의미 있는 답변 품질은 기대하지 않는다. 이 클래스의 목적은
      "에러 없이 retrieval -> prompt 조립 -> generation까지 파이프라인이
      끝까지 도는가"라는 기능적 동작 확인이다 (내용 정확도는 추후 개선 예정).

환경 (Gemma 사용 시): 로컬 Mac M3 (Apple Silicon) — MPS(Metal Performance Shaders) 디바이스 사용
"""

from pathlib import Path
from paths import SRC_DIR
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

from custom_transformer.transformer_model import TransformerLanguageModel
from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer
from custom_transformer.model.utils.generation_utils import trim_after_eos


CUSTOM_TRANSFORMER_MODEL_NAME = "custom_transformer"

# custom_transformer 패키지(src/custom_transformer/) 기준 데이터/가중치 경로.
# generator.py가 src/rag_pipeline/에 있으므로, 두 단계 위(src/)에서 custom_transformer로 들어간다.
_CUSTOM_TRANSFORMER_DIR = SRC_DIR / "custom_transformer"
_CUSTOM_QA_DATA_PATH = _CUSTOM_TRANSFORMER_DIR / "scripts" / "raw_data" / "korean_qa.txt"
_CUSTOM_MODEL_WEIGHTS_PATH = _CUSTOM_TRANSFORMER_DIR / "scripts" / "trained_model" / "korean_model.pt"


class TextGenerator:
    """
    하나의 생성 모델(Gemma 4 E2B-it 또는 커스텀 TransformerLanguageModel)을
    한 번만 로딩하고, 여러 번 재사용하여 prompt로부터 텍스트를 생성하는 클래스.

    모델 로딩(비용이 큰 작업)과 생성(반복적으로 호출되는 작업)을 분리하기 위해
    클래스로 구현한다. model_name에 따라 __init__()에서 두 모델 중 하나만 로딩한다
    (둘 다 동시에 로딩하지 않음 — 메모리 낭비 방지).

    generate(): 전체 텍스트를 한 번에 반환한다 (기존 동작, 평가 등에서 재사용).
    generate_stream(): 토큰이 생성되는 즉시 하나씩 반환한다 (Step C-3 스트리밍).
        - 커스텀 Transformer는 아직 진짜 토큰 단위 스트리밍을 지원하지 않는다
          (model.generate()가 전체 시퀀스를 한 번에 만든 뒤 반환하는 구조이기 때문).
          NotImplementedError를 발생시킨다.
    """

    # 커스텀 Transformer의 positional encoding max_len(512)을 넘지 않도록,
    # 입력 prompt를 안전하게 자를 상한선.
    CUSTOM_MAX_INPUT_TOKENS = 400

    def __init__(self, model_name: str = "google/gemma-4-E2B-it"):
        """
        Args:
            model_name: 사용할 모델 이름.
                        "custom_transformer"이면 직접 구현한 TransformerLanguageModel을,
                        그 외 값이면 HuggingFace의 instruction-tuned causal LM(기본값: Gemma 4 E2B-it)을 로딩한다.
        """
        self.model_name = model_name

        if model_name == CUSTOM_TRANSFORMER_MODEL_NAME:
            self._init_custom_transformer()
        else:
            self._init_gemma()

    # ------------------------------------------------------------------
    # 초기화 — Gemma
    # ------------------------------------------------------------------
    def _init_gemma(self) -> None:
        # Apple Silicon(M3)에서는 MPS 디바이스를 사용한다.
        # MPS를 사용할 수 없는 환경(예: NVIDIA GPU 환경)에서는 cuda로,
        # 둘 다 없으면 cpu로 자동 대체된다.
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device) # type:ignore

    # ------------------------------------------------------------------
    # 초기화 — 커스텀 Transformer
    # ------------------------------------------------------------------
    def _init_custom_transformer(self) -> None:
        # 1. QA 데이터 로딩 (train.py와 동일한 방식으로 tokenizer 재구성).
        #    tokenizer 자체를 저장/로드하지 않으므로, 매번 같은 데이터로 재학습해야
        #    train.py가 만든 vocab(token_to_id)과 동일한 결과가 나온다.
        #    (주의 사항) — 이는 현재 구조의 한계이며, 추후 tokenizer를 직렬화해
        #    저장/로드하는 방식으로 개선할 수 있다.
        qa_pairs = self._load_qa_pairs(_CUSTOM_QA_DATA_PATH)
        flat_corpus = [question for question, _ in qa_pairs] + [answer for _, answer in qa_pairs]

        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(flat_corpus)
        self.eos_token_id = self.tokenizer.token_to_id[self.tokenizer.eos_token]

        # 2. 모델 구조 생성 (train.py와 동일한 하이퍼파라미터)
        vocab_size = len(self.tokenizer.token_to_id)
        self.model = TransformerLanguageModel(
            vocab_size=vocab_size,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            max_len=512,
            dropout=0.1,
        )

        # 3. 학습된 가중치 로드
        try:
            self.model.load_state_dict(torch.load(str(_CUSTOM_MODEL_WEIGHTS_PATH), map_location="cpu"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"학습된 모델 파일을 찾을 수 없습니다: {_CUSTOM_MODEL_WEIGHTS_PATH}. "
                "custom_transformer/scripts/train.py를 먼저 실행해서 모델을 학습/저장해야 합니다."
            ) from exc

        self.model.eval()

    @staticmethod
    def _load_qa_pairs(path: Path) -> list[tuple[str, str]]:
        """scripts/train.py의 load_qa_pairs()와 동일한 로직."""
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

    # ------------------------------------------------------------------
    # 생성 — 공개 인터페이스 (모델 종류와 무관하게 동일한 시그니처)
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        주어진 prompt로부터 텍스트를 생성한다.

        Args:
            prompt: build_prompt()로 생성된 완성된 prompt 문자열.
            max_new_tokens: 생성할 최대 토큰 수

        Returns:
            새로 생성된 부분만 반환 (prompt 부분은 제외하고 strip()된 문자열)

        Raises:
            ValueError: prompt가 빈 문자열일 경우
        """
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

        if self.model_name == CUSTOM_TRANSFORMER_MODEL_NAME:
            return self._generate_custom_transformer(prompt, max_new_tokens)
        return self._generate_gemma(prompt, max_new_tokens)

    def _generate_gemma(self, prompt: str, max_new_tokens: int) -> str:
        # build_prompt()의 결과를 user 메시지로 감싸 chat template을 적용한다.
        # enable_thinking=False: 학습 목적상 thinking 과정 없이 바로 답변만 받기 위함.
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # type: ignore
            )

        # 생성된 전체 토큰에서, 입력 prompt에 해당하는 부분을 제외하고 새로 생성된 부분만 추출
        new_tokens = output_ids[0][input_length:]
        generated_text = self.processor.decode(new_tokens, skip_special_tokens=True)

        return generated_text.strip()

    def _generate_custom_transformer(self, prompt: str, max_new_tokens: int) -> str:
        # 1. 인코딩
        input_ids = self.tokenizer.encode(prompt)

        # 2. 길이 초과 방지를 위한 truncate.
        #    (주의 사항) 단순히 뒤쪽(질문에 가까운 부분)을 남기고 앞쪽을 잘라내는 방식이다.
        #    문서 청크가 잘려나갈 수 있으나, 이번 단계의 목표는 "에러 없이 끝까지
        #    도는가"이므로 내용 손실은 허용한다.
        if len(input_ids) > self.CUSTOM_MAX_INPUT_TOKENS:
            input_ids = input_ids[-self.CUSTOM_MAX_INPUT_TOKENS:]

        input_length = len(input_ids)
        input_tensor = torch.tensor([input_ids])

        # 3. 생성 (학습된 eos_token_id로 조기 종료 적용)
        with torch.no_grad():
            generated = self.model.generate( # type: ignore
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                eos_token_id=self.eos_token_id,
            )

        # 4. 입력 길이만큼 슬라이싱하여, 새로 생성된 부분만 추출
        #    (Gemma의 output_ids[0][input_length:]와 동일한 처리)
        generated_ids = generated[0].tolist()
        new_token_ids = generated_ids[input_length:]

        # 5. eos 이후를 한 번 더 안전하게 제거 (조기 종료가 이미 처리했겠지만,
        #    max_new_tokens에 도달해 eos 없이 끝난 경우를 대비)
        new_token_ids = trim_after_eos(new_token_ids, eos_token_id=self.eos_token_id)

        generated_text = self.tokenizer.decode(new_token_ids)
        return generated_text.strip()

    # ------------------------------------------------------------------
    # 스트리밍 — Gemma만 지원, 커스텀 Transformer는 미구현
    # ------------------------------------------------------------------
    def _build_inputs(self, prompt: str):
        """공통 입력 전처리 (Gemma 전용): chat template 적용 후 토큰화한다."""
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.processor(text=text, return_tensors="pt").to(self.device)

    def generate_stream(self, prompt: str, max_new_tokens: int = 80):
        """
        주어진 prompt로부터 텍스트를 생성하며, 토큰이 만들어지는 즉시 하나씩 반환한다.
        (Gemma 전용. 커스텀 Transformer는 아직 미구현 — 아래 설명 참고)

        model.generate()는 끝날 때까지 멈추는(blocking) 함수이므로, 별도 스레드에서
        실행시키고, 이 함수(메인 흐름)는 TextIteratorStreamer를 순회하며 새로 생성된
        텍스트 조각을 즉시 yield한다.

        Args:
            prompt: build_prompt()로 생성된 완성된 prompt 문자열
            max_new_tokens: 생성할 최대 토큰 수

        Yields:
            새로 생성된 텍스트 조각(str). 보통 1개 이상의 토큰 단위.

        Raises:
            ValueError: prompt가 빈 문자열일 경우
            NotImplementedError: 커스텀 Transformer 사용 시
                (이 모델은 model.generate()가 전체 시퀀스를 한 번에 만든 뒤 반환하는
                구조라, Gemma의 TextIteratorStreamer처럼 토큰이 만들어지는 즉시
                스트리밍할 수 있는 구조가 아직 없다. /query(비스트리밍)가 먼저
                검증된 뒤, generate()의 결과를 잘라서 흘려보내는 방식으로 확장할 계획.)
        """
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

        if self.model_name == CUSTOM_TRANSFORMER_MODEL_NAME:
            raise NotImplementedError(
                "커스텀 Transformer의 generate_stream()은 아직 구현되지 않았습니다. "
                "/query(비스트리밍) 경로가 먼저 검증된 뒤 추가될 예정입니다."
            )

        inputs = self._build_inputs(prompt)

        # skip_prompt=True: 입력 prompt 부분은 스트리밍하지 않고, 새로 생성된 부분만 흘려보낸다.
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # model.generate()는 blocking 함수이므로 별도 스레드에서 실행한다.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs) # type: ignore
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_by_section
    from rag_pipeline.embedder import TextEmbedder
    from rag_pipeline.vector_store import InMemoryVectorStore
    from rag_pipeline.retriever import retrieve_top_k
    from rag_pipeline.prompt_builder import build_prompt

    # Step A + Step B 앞부분 재구성
    sample_path = DATA_DIR / "daysync_manual.md"
    document = load_document(str(sample_path))
    chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)

    embedder = TextEmbedder()
    vectors = embedder.encode(chunks)

    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    question = "What is the internal codename of NimbusFlow during development?"
    query_vector = embedder.encode([question])[0]
    retrieved = retrieve_top_k(query_vector, store, k=3)

    prompt = build_prompt(question, retrieved)

    # Generation — model_name을 바꿔서 직접 비교해볼 수 있다.
    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 약 10GB 다운로드가 필요합니다]")
    # generator = TextGenerator(model_name="google/gemma-4-E2B-it")
    generator = TextGenerator(model_name="custom_transformer")

    print(f"[질문] {question}\n")
    print("[비스트리밍 답변] ", end="", flush=True)
    answer = generator.generate(prompt)
    print(answer)