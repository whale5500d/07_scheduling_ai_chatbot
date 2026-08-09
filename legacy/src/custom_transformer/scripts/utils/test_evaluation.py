import pytest
import torch
import torch.nn as nn

from .evaluation import (
    compute_perplexity,
    evaluate_exact_match,
    extract_question_activity,
    parse_generated_answer,
)


# ----------------------------------------------------------------------
# 순수 함수 테스트 (모델/토크나이저 불필요)
# ----------------------------------------------------------------------
class TestExtractQuestionActivity:
    def test_extracts_activity_after_removing_time_and_verb(self):
        assert extract_question_activity("오늘 산책 할 거야?") == "산책"
        assert extract_question_activity("내일 캠핑 갈 거야?") == "캠핑"

    def test_handles_new_time_expressions(self):
        assert extract_question_activity("모레 베이킹 할 거야?") == "베이킹"
        assert extract_question_activity("이번 주말 청소 할 거야?") == "청소"


class TestParseGeneratedAnswer:
    def test_parses_accept_answer(self):
        result = parse_generated_answer("응, 산책 할 거야")
        assert result["label"] == "응"
        assert result["primary_activity"] == "산책"
        assert result["substitute_activity"] is None

    def test_parses_reject_answer_with_substitute(self):
        result = parse_generated_answer("아니, 조깅 대신 요가 할 거야")
        assert result["label"] == "아니"
        assert result["primary_activity"] == "조깅"
        assert result["substitute_activity"] == "요가"

    def test_malformed_answer_returns_none_label(self):
        assert parse_generated_answer("완전 이상한 문장")["label"] is None


# ----------------------------------------------------------------------
# compute_perplexity 테스트 (uniform logits -> perplexity == vocab_size,
# 수식으로 정확히 검증 가능한 결정론적 케이스)
# ----------------------------------------------------------------------
class UniformLogitsModel(nn.Module):
    """모든 위치에서 균등분포(logits 전부 0)를 출력하는 가짜 모델.
    이 경우 cross-entropy = log(vocab_size)이므로 perplexity == vocab_size가
    수식으로 정확히 보장된다 (perplexity 자체를 신뢰할 수 있는 정답으로 검증)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs_tensor):
        batch_size, seq_len = inputs_tensor.shape
        return torch.zeros(batch_size, seq_len, self.vocab_size)


class CharTokenizer:
    """테스트 전용 char-level 토크나이저. 실제 BPETokenizer의 인터페이스
    (token_to_id, eos_token, encode(), decode())만 동일하게 재현한다."""

    def __init__(self, corpus_texts):
        chars = sorted(set("".join(corpus_texts)))
        self.token_to_id = {ch: idx for idx, ch in enumerate(chars)}
        self.eos_token = "<eos>"
        self.token_to_id[self.eos_token] = len(self.token_to_id)
        self._id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        return [self.token_to_id[ch] for ch in text]

    def decode(self, ids):
        eos_id = self.token_to_id[self.eos_token]
        return "".join(self._id_to_token[i] for i in ids if i != eos_id)


class TestComputePerplexity:
    def test_uniform_model_perplexity_equals_vocab_size(self):
        qa_pairs = [("ab", "ba"), ("aa", "bb")]
        tokenizer = CharTokenizer([q for q, _ in qa_pairs] + [a for _, a in qa_pairs])
        vocab_size = len(tokenizer.token_to_id)
        model = UniformLogitsModel(vocab_size)

        perplexity = compute_perplexity(model, tokenizer, qa_pairs)

        assert perplexity == pytest.approx(vocab_size, rel=1e-4)

    def test_raises_value_error_when_all_pairs_are_empty(self):
        tokenizer = CharTokenizer(["a"])
        model = UniformLogitsModel(len(tokenizer.token_to_id))

        with pytest.raises(ValueError):
            compute_perplexity(model, tokenizer, [("", "")])


# ----------------------------------------------------------------------
# evaluate_exact_match 테스트 (generate()가 미리 정해진 답을 반환하도록 고정해
# slot_copy_accuracy / label_accuracy 집계 로직 자체만 검증)
# ----------------------------------------------------------------------
class FixedOutputModel:
    """model.generate()가 question -> 미리 정해진 answer로 매핑된 값을
    반환하도록 고정한 가짜 모델. 파싱·집계 로직의 배선(wiring)만 검증하는 것이
    목적이므로, 실제 생성 품질과는 무관하다."""

    def __init__(self, tokenizer, question_to_answer_map, eos_id):
        self.tokenizer = tokenizer
        self.question_to_answer_map = question_to_answer_map
        self.eos_id = eos_id

    def eval(self):
        return self

    def generate(self, input_ids, max_new_tokens, temperature, top_k, eos_token_id):
        question_ids = input_ids[0].tolist()
        question_text = self.tokenizer.decode(question_ids)
        answer_text = self.question_to_answer_map[question_text]
        answer_ids = self.tokenizer.encode(answer_text) + [self.eos_id]
        full_ids = question_ids + answer_ids
        return torch.tensor([full_ids])


class TestEvaluateExactMatch:
    def test_accuracy_reflects_correct_and_incorrect_predictions(self):
        qa_pairs = [
            ("오늘 산책 할 거야?", "응, 산책 할 거야"),
            ("내일 캠핑 갈 거야?", "응, 캠핑 갈 거야"),
        ]
        # 모델이 실제로 생성하는 답 (1번은 정답과 동일, 2번은 label/slot 모두 다름)
        question_to_answer_map = {
            "오늘 산책 할 거야?": "응, 산책 할 거야",
            "내일 캠핑 갈 거야?": "아니, 등산 대신 헬스장 갈 거야",
        }
        tokenizer = CharTokenizer(
            [q for q, _ in qa_pairs]
            + [a for _, a in qa_pairs]
            + list(question_to_answer_map.values())
        )
        eos_id = tokenizer.token_to_id[tokenizer.eos_token]
        model = FixedOutputModel(tokenizer, question_to_answer_map, eos_id)

        result = evaluate_exact_match(model, tokenizer, qa_pairs)

        assert result["slot_copy_accuracy"] == pytest.approx(0.5)
        assert result["label_accuracy"] == pytest.approx(0.5)
        assert len(result["details"]) == 2
        assert result["details"][0]["is_slot_copy_correct"] is True
        assert result["details"][1]["is_slot_copy_correct"] is False