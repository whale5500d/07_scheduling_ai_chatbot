"""
tests/ 전체 공용 conftest.py — torch/transformers/sentence_transformers가 설치되지
않은 환경(이 작업을 진행한 sandbox)에서도 src/langchain_pipeline/llm.py,
src/rag_pipeline/embedder.py, src/rag_pipeline/generator.py, src/main.py를
import할 수 있도록 하기 위한 "import 통과용" 가짜(stub) 모듈 주입.

[이 stub이 하는 일 / 하지 않는 일 — 반드시 구분할 것]
- 하는 일: `import torch`, `import torch.nn as nn`, `from transformers import
  AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer, pipeline`,
  `from sentence_transformers import SentenceTransformer` 등 "이름이 존재하는지"만
  만족시킨다. 모든 클래스는 인자를 무시하고 아무 동작도 하지 않는 깡통(placeholder)이다.
- 하지 않는 일: 실제 수치 연산, 실제 모델 로딩/추론을 절대 흉내내지 않는다. 따라서
  이 stub은 "실제 가중치를 가진 모델을 생성해서 의미 있는 텍스트/임베딩을 만들어내는"
  경로를 테스트하는 데는 쓸 수 없다 — 그런 테스트는 여전히 @pytest.mark.slow로
  표시되어 기본 실행에서 제외되거나(6단계 test_llm.py), 처음부터 disclosed-untestable로
  남는다(test_embedding.py의 4개 에러 — sentence_transformers 자체가 필요한 경로).
  대신, "model/embedder/generator를 생성자에 직접 주입하거나 duck-typed fake로
  완전히 대체해서, 그 주변 로직(분기, 조립, 직렬화)만 검증하는" 테스트를 가능하게 한다.

[실제 torch/transformers/sentence_transformers가 설치된 환경(예: 사용자 로컬 PC)에서는?]
각 _ensure_*_stub() 함수가 먼저 진짜 라이브러리를 import해 보고, 성공하면 아무 것도
하지 않고 즉시 반환한다. 따라서 이 stub은 sandbox에서만 활성화되고, 실제 환경에서는
한 줄도 영향을 주지 않는다.
"""

import sys
import types
from contextlib import AbstractContextManager
from typing import Any


class _Dummy:
    """존재만 하면 되는, 미사용 placeholder (nn.Linear, nn.LayerNorm, nn.Module,
    SentenceTransformer, AutoModelForCausalLM 등).

    base class로도 쓰일 수 있어야 하므로(class Foo(nn.Module): ...) 일반 클래스로
    정의한다. __init__/__call__은 어떤 인자가 와도 받아주고 아무 일도 하지 않는다.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _AutoNamespace(types.ModuleType):
    """torch.nn / torch.nn.functional처럼, 어떤 속성에 접근해도 _Dummy를 반환하는 가짜 모듈.

    실제 torch.nn에는 Linear/Embedding/LayerNorm/Dropout/Sequential/ModuleList/ReLU 등
    수십 개의 이름이 있는데, 이를 일일이 나열하는 대신 "어떤 이름이든 _Dummy로 응답"하는
    __getattr__ fallback을 쓴다 — custom_transformer/model/*.py가 정확히 어떤 이름을
    쓰는지와 무관하게 항상 import에 성공한다(올바른 동작을 흉내내는 게 아니라, import
    시점에만 존재하면 되는 이름이기 때문에 안전하다).
    """

    def __getattr__(self, item: str) -> Any:
        return _Dummy


class _NoGradContext(AbstractContextManager):
    """torch.no_grad()의 대체.

    실제 torch.no_grad는 with 문(context manager)으로도, @torch.no_grad()처럼
    데코레이터로도 쓰인다 — custom_transformer/transformer_model.py의
    generate() 메서드가 @torch.no_grad() 데코레이터를 쓰기 때문에(클래스 본문이
    실행되는 import 시점에 데코레이터가 즉시 호출됨), __call__도 함께 구현해야
    import가 성공한다.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def __call__(self, func: Any) -> Any:
        return func  # 데코레이터로 쓰일 때: 함수를 그대로 통과시킨다 (실제 no_grad 의미는 없음).


def _install_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _ensure_torch_stub() -> None:
    try:
        import torch  # noqa: F401

        return  # 실제 torch가 설치되어 있으면 stub을 쓰지 않는다.
    except ImportError:
        pass

    torch_mod = types.ModuleType("torch")
    nn_mod = _AutoNamespace("torch.nn")
    functional_mod = _AutoNamespace("torch.nn.functional")
    backends_mod = types.ModuleType("torch.backends")
    mps_mod = types.ModuleType("torch.backends.mps")
    cuda_mod = types.ModuleType("torch.cuda")

    mps_mod.is_available = lambda: False
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    backends_mod.mps = mps_mod

    torch_mod.nn = nn_mod
    torch_mod.nn.functional = functional_mod
    torch_mod.backends = backends_mod
    torch_mod.cuda = cuda_mod
    torch_mod.Tensor = _Dummy
    torch_mod.no_grad = lambda: _NoGradContext()
    # torch.tensor(data)는 실제로는 텐서를 만들지만, 여기서는 입력을 그대로
    # 반환하는 identity로 대체한다 — CustomTransformerLLM._call()에서
    # torch.tensor([input_ids])로 감싼 뒤 가짜 model.generate()에 그대로
    # 넘기기만 하므로, 텐서로서의 실제 동작은 필요 없다.
    torch_mod.tensor = lambda data, *args, **kwargs: data
    torch_mod.load = lambda *args, **kwargs: {}

    _install_module("torch", torch_mod)
    _install_module("torch.nn", nn_mod)
    _install_module("torch.nn.functional", functional_mod)
    _install_module("torch.backends", backends_mod)
    _install_module("torch.backends.mps", mps_mod)
    _install_module("torch.cuda", cuda_mod)


def _ensure_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401

        return  # 실제 transformers가 설치되어 있으면 stub을 쓰지 않는다.
    except ImportError:
        pass

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = _Dummy
    transformers_mod.AutoProcessor = _Dummy
    transformers_mod.TextIteratorStreamer = _Dummy
    transformers_mod.pipeline = lambda *args, **kwargs: None

    _install_module("transformers", transformers_mod)


def _ensure_sentence_transformers_stub() -> None:
    try:
        import sentence_transformers  # noqa: F401

        return  # 실제 sentence_transformers가 설치되어 있으면 stub을 쓰지 않는다.
    except ImportError:
        pass

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _Dummy

    _install_module("sentence_transformers", st_mod)


_ensure_torch_stub()
_ensure_transformers_stub()
_ensure_sentence_transformers_stub()
