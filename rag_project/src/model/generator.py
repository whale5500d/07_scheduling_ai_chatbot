"""
Step B: Generation (+ Step C-3: 스트리밍)

책임(Responsibility): prompt(문자열)를 받아 LLM(Gemma 4 E2B-it)으로 응답 텍스트를 생성한다.
이 단계 이전의 Retrieval, Prompt Augmentation 결과를 입력으로 받는다.

모델: google/gemma-4-E2B-it
- "-it"는 instruction-tuned(지시문 따르기 학습됨) variant를 의미한다.
- base 모델(-it 없음)은 "다음에 올 텍스트를 통계적으로 이어 쓰는" 방식으로만 학습되어 있어,
  Q&A 형식의 prompt를 주면 답변 후에도 비슷한 패턴(Question/Answer)을 계속 생성하는
  문제가 있었다 (트러블슈팅 #10 참고). instruction-tuned 모델은 "사용자 질문에
  답하고 멈춘다"는 지시를 따르도록 별도로 학습되어 있어 이 문제가 해결된다.
- "근본에서 확장" 원칙에 따라 Gemma 4 중 가장 작은 사이즈(E2B)를 유지한다.
- 답변 품질이 부족하거나 메모리 문제가 발생하면 E4B로 확장할 예정이다 (트러블슈팅 #9 참고).

환경: 로컬 Mac M3 (Apple Silicon) — MPS(Metal Performance Shaders) 디바이스 사용
"""

import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer


class TextGenerator:
    """
    Gemma 4 E2B-it 모델을 한 번만 로딩하고, 여러 번 재사용하여
    prompt로부터 텍스트를 생성하는 클래스.

    TextEmbedder와 동일한 이유로, 모델 로딩(비용이 큰 작업)과
    생성(반복적으로 호출되는 작업)을 분리하기 위해 클래스로 구현한다.

    AutoTokenizer 대신 AutoProcessor를 사용한다 — Gemma 4 공식 예제 기준으로,
    chat template 적용과 멀티모달 확장성을 함께 지원하는 인터페이스이다.
    
    generate(): 전체 텍스트를 한 번에 반환한다 (기존 동작, 평가 등에서 재사용).
    generate_stream(): 토큰이 생성되는 즉시 하나씩 반환한다 (Step C-3 스트리밍).
    """

    def __init__(self, model_name: str = "google/gemma-4-E2B-it"):
        """
        Args:
            model_name: 사용할 Gemma 4 instruction-tuned 모델 이름
        """
        self.model_name = model_name

        # Apple Silicon(M3)에서는 MPS 디바이스를 사용한다.
        # MPS를 사용할 수 없는 환경(예: NVIDIA GPU 환경)에서는 cuda로,
        # 둘 다 없으면 cpu로 자동 대체된다.
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        주어진 prompt로부터 텍스트를 생성한다.

        Args:
            prompt: build_prompt()로 생성된 완성된 prompt 문자열.
                    chat template의 user 메시지 content로 그대로 전달된다.
            max_new_tokens: 생성할 최대 토큰 수 (짧은 사실 기반 답변에 충분한 기본값)

        Returns:
            LLM이 생성한 응답 텍스트 (prompt 부분은 제외하고 새로 생성된 부분만)

        Raises:
            ValueError: prompt가 빈 문자열일 경우
        """
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

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
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 학습/검증 목적상 결정론적(deterministic) 출력을 위해 greedy decoding 사용
            )

        # 생성된 전체 토큰에서, 입력 prompt에 해당하는 부분을 제외하고 새로 생성된 부분만 추출
        new_tokens = output_ids[0][input_length:]
        generated_text = self.processor.decode(new_tokens, skip_special_tokens=True)

        return generated_text.strip()

    def _build_inputs(self, prompt: str):
        """공통 입력 전처리: chat template 적용 후 토큰화한다."""
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

        model.generate()는 끝날 때까지 멈추는(blocking) 함수이므로, 별도 스레드에서
        실행시키고, 이 함수(메인 흐름)는 TextIteratorStreamer를 순회하며 새로 생성된
        텍스트 조각을 즉시 yield한다 — 생산자(백그라운드 스레드)와 소비자(이 generator)
        가 분리된 구조이다.

        Args:
            prompt: build_prompt()로 생성된 완성된 prompt 문자열
            max_new_tokens: 생성할 최대 토큰 수

        Yields:
            새로 생성된 텍스트 조각(str). 보통 1개 이상의 토큰 단위.

        Raises:
            ValueError: prompt가 빈 문자열일 경우
        """
        if not prompt.strip():
            raise ValueError("prompt가 비어 있습니다. 생성할 내용이 없습니다.")

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
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag_pipeline.document_loader import load_document
    from rag_pipeline.chunker import chunk_by_section
    from rag_pipeline.embedder import TextEmbedder
    from rag_pipeline.vector_store import InMemoryVectorStore
    from rag_pipeline.retriever import retrieve_top_k
    from model.prompt_builder import build_prompt

    # Step A + Step B 앞부분 재구성
    sample_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_manual.md"
    )
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

    # Generation
    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 약 10GB 다운로드가 필요합니다]")
    generator = TextGenerator()

    print(f"[질문] {question}\n")
    # 방법 1 - 비스트리밍, generate()
    print("[비스트리밍 답변] ", end="", flush=True)
    answer = generator.generate(prompt)
    print()

    # 방법 2 - 스트리밍, generate_stream()
    # print("[스트리밍 답변] ", end="", flush=True)
    # for token in generator.generate_stream(prompt):
    #     print(token, end="", flush=True)
    #     time.sleep(0.3)  # 토큰 사이에 0.3초 지연을 줘서 점진적으로 출력되는지 눈으로 확인
    # print()