import torch
from custom_transformer.transformer_model import TransformerLanguageModel


class TestTransformerLanguageModelForwardPass:
    def test_forward_pass_returns_correct_logits_shape(self):
        model = TransformerLanguageModel(
            vocab_size=1000, d_model=128, num_heads=8, num_layers=4, d_ff=512, max_len=256, dropout=0.1
        )
        input_ids = torch.randint(0, 1000, (2, 12))
        logits = model(input_ids)
        assert logits.shape == (2, 12, 1000)

    def test_forward_pass_with_causal_mask_preserves_shape(self):
        model = TransformerLanguageModel(
            vocab_size=1000, d_model=128, num_heads=8, num_layers=4, d_ff=512, max_len=256, dropout=0.1
        )
        input_ids = torch.randint(0, 1000, (2, 12))
        causal_mask = torch.tril(torch.ones(12, 12)).unsqueeze(0).unsqueeze(0).expand(2, 1, -1, -1)
        logits = model(input_ids, mask=causal_mask)
        assert logits.shape == (2, 12, 1000)


class TestTransformerLanguageModelGenerate:
    def test_generate_without_eos_reaches_max_new_tokens(self):
        model = TransformerLanguageModel(
            vocab_size=1000, d_model=128, num_heads=8, num_layers=4, d_ff=512, max_len=256, dropout=0.1
        )
        generated = model.generate(torch.tensor([[1]]), max_new_tokens=15, temperature=0.8, top_k=30)
        assert generated.shape[1] == 1 + 15

    def test_generate_stops_immediately_when_eos_is_sampled(self, monkeypatch):
        """어떤 토큰이 나오는지는 monkeypatch로 통제하고, 조기 종료 메커니즘 자체만 검증한다.
        torch 버전/플랫폼에 따라 manual_seed 재현 결과가 달라지는 문제를 피하기 위해,
        시드에 의존하지 않고 torch.multinomial의 반환값을 직접 고정한다."""
        model = TransformerLanguageModel(
            vocab_size=10, d_model=16, num_heads=2, num_layers=1, d_ff=32, max_len=64
        )
        eos_token_id = 3

        def fake_multinomial(probs, num_samples):
            return torch.tensor([[eos_token_id]])

        monkeypatch.setattr(torch, "multinomial", fake_multinomial)

        generated = model.generate(
            torch.tensor([[1]]), max_new_tokens=15, temperature=0.8, top_k=5, eos_token_id=eos_token_id
        )
        generated_ids = generated.tolist()[0]

        assert generated_ids[-1] == eos_token_id, "eos가 나왔는데 마지막 토큰이 아닙니다 (조기 종료 실패)."
        assert len(generated_ids) == 2, "eos가 첫 스텝에 나왔는데도 조기 종료되지 않았습니다."

    def test_generate_continues_when_eos_never_sampled(self, monkeypatch):
        """eos가 아닌 토큰만 계속 나오는 경우, max_new_tokens까지 끝까지 생성되는지 확인한다."""
        model = TransformerLanguageModel(
            vocab_size=10, d_model=16, num_heads=2, num_layers=1, d_ff=32, max_len=64
        )
        eos_token_id = 3
        non_eos_token_id = 5

        def fake_multinomial(probs, num_samples):
            return torch.tensor([[non_eos_token_id]])

        monkeypatch.setattr(torch, "multinomial", fake_multinomial)

        generated = model.generate(
            torch.tensor([[1]]), max_new_tokens=15, temperature=0.8, top_k=5, eos_token_id=eos_token_id
        )
        generated_ids = generated.tolist()[0]

        assert eos_token_id not in generated_ids
        assert len(generated_ids) == 1 + 15, "eos가 한 번도 안 나왔는데 조기 종료된 것처럼 보입니다."