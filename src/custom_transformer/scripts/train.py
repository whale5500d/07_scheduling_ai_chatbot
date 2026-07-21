# scripts/train.py
from custom_transformer.scripts.utils.evaluation import compute_perplexity, evaluate_exact_match

"""
Instruction Tuning 훈련 루프

변경 사항 (기존 사전학습 train.py 대비):
    1. 데이터: korean_corpus.txt(독립 문장) -> korean_qa.txt(질문\t답변 쌍)
    2. Loss 계산: 전체 시퀀스 -> 답변(answer) 부분만 (질문 부분은 마스킹)
    3. build_qa_training_pair()를 통해 (inputs, labels) 생성

전제: 기존 korean_model.pt는 보존하지 않고, vocab/모델 모두 처음부터 새로 학습.
      (vocab을 corpus 단어로 제한해야 하는 제약이 없어짐)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from custom_transformer.transformer_model import TransformerLanguageModel
from custom_transformer.model.utils.generation_utils import trim_after_eos
from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer
from custom_transformer.scripts.utils.qa_collate import build_qa_training_pair, IGNORE_INDEX


def load_qa_pairs(path: str) -> list[tuple[str, str]]:
    """
    '질문\t답변' 형식의 파일을 읽어 (질문, 답변) 튜플 리스트로 반환.

    Known Limitation:
        - 탭이 정확히 1개 있다고 가정. 형식이 깨진 줄은 건너뛴다.
    """
    qa_pairs = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue  # 형식이 맞지 않는 줄은 제외
            question, answer = parts
            qa_pairs.append((question, answer))
    return qa_pairs


def main():
    print("=== Instruction Tuning 훈련 루프 시작 ===\n")

    # 1. QA 쌍 데이터 로딩
    qa_pairs = load_qa_pairs("src/custom_transformer/scripts/raw_data/korean_qa_train.txt")
    heldout_qa_pairs = load_qa_pairs("src/custom_transformer/scripts/raw_data/korean_qa_heldout.txt")
    print(f"학습 데이터 수: {len(qa_pairs)} 쌍")
    print(f"Held-out 데이터 수: {len(heldout_qa_pairs)} 쌍")

    # 2. BPE Tokenizer 초기화 및 훈련
    # tokenizer 학습은 여전히 문장 단위 리스트를 입력으로 받으므로,
    # 질문과 답변을 모두 풀어서(flatten) corpus로 사용
    flat_corpus = (
        [question for question, _ in qa_pairs] + [answer for _, answer in qa_pairs]
        + [question for question, _ in heldout_qa_pairs] + [answer for _, answer in heldout_qa_pairs]
    )

    VOCAB_SIZE = 300
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(flat_corpus)
    print(f"Tokenizer Vocabulary size: {len(tokenizer.token_to_id)}")

    # 3. TransformerLanguageModel 초기화
    vocab_size = len(tokenizer.token_to_id)
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_len=512,
        dropout=0.1
    )
    model.train()
    print(f"Model initialized with vocab_size={vocab_size}")

    # 4. Loss Function, Optimizer 정의
    # ignore_index=-100: build_qa_training_pair()가 마스킹한 질문 영역은 loss에서 제외됨
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # 5. 훈련 루프 (Instruction Tuning: 답변 부분만 loss 계산)
    num_epochs = 50
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]

    for epoch in range(num_epochs):
        total_loss = 0
        num_skipped = 0

        for question, answer in qa_pairs:
            question_ids = tokenizer.encode(question)
            raw_answer_ids = tokenizer.encode(answer)

            # 질문 또는 답변이 토큰화 후 비어 있으면 해당 쌍은 건너뜀
            # (raw_answer_ids 기준으로 체크해야 함 — eos 추가 후에는 항상 비어있지 않게 되므로)
            if len(question_ids) == 0 or len(raw_answer_ids) == 0:
                num_skipped += 1
                continue

            # 답변 끝에 <eos>를 추가하여, 모델이 "답변이 끝나면 <eos>를 예측해야 한다"는
            # 패턴을 학습하도록 한다. (이게 빠지면 generate()의 조기 종료가 동작할 수 없음)
            answer_ids = raw_answer_ids + [eos_id]

            inputs, labels = build_qa_training_pair(question_ids, answer_ids)

            inputs_tensor = torch.tensor([inputs])          # (1, seq_len)
            labels_tensor = torch.tensor([labels])          # (1, seq_len)

            # Forward
            outputs = model(inputs_tensor)  # (1, seq_len, vocab_size)

            # Loss 계산 (ignore_index에 해당하는 위치는 자동으로 제외됨)
            loss = criterion(
                outputs.view(-1, vocab_size),
                labels_tensor.view(-1)
            )

            # Backward + Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        denom = max(1, len(qa_pairs) - num_skipped)
        avg_loss = total_loss / denom
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.4f} | Skipped: {num_skipped}")

    # 6. 훈련 후 생성 결과 확인
    print("\n=== 훈련 후 생성 결과 확인 ===\n")

    model.eval()

    test_prompts = [
        "오늘 산책 할 거야?",
        "내일 운동 할 거야?",
        "오늘 회의 할 거야?",
    ]

    with torch.no_grad():
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids])

            generated = model.generate(
                input_ids=input_tensor,
                max_new_tokens=15,
                temperature=0.8,
                top_k=50,
                eos_token_id=eos_id
            )

            generated_ids = generated[0].tolist()
            # <eos> 이후는 잘라내서 디코딩 (생성 시점에는 이미 eos에서 멈췄지만,
            # max_new_tokens에 도달해 eos 없이 끝난 경우를 대비해 한 번 더 안전하게 처리)
            trimmed_ids = trim_after_eos(generated_ids, eos_token_id=eos_id)

            generated_text = tokenizer.decode(trimmed_ids)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}\n")
    
    # 7. held-out 평가
    print("\n=== Held-out 평가 (perplexity / exact match) ===\n")

    train_perplexity = compute_perplexity(model, tokenizer, qa_pairs)
    heldout_perplexity = compute_perplexity(model, tokenizer, heldout_qa_pairs)
    print(f"Training Perplexity : {train_perplexity:.4f}")
    print(f"Held-out Perplexity  : {heldout_perplexity:.4f}")
    print(f"Perplexity Gap       : {heldout_perplexity - train_perplexity:.4f}")

    exact_match_result = evaluate_exact_match(model, tokenizer, heldout_qa_pairs)
    print(f"\nHeld-out Slot Copy Accuracy : {exact_match_result['slot_copy_accuracy']:.4f}")
    print(f"Held-out Label Accuracy     : {exact_match_result['label_accuracy']:.4f}")

    # 진단용 임시 출력 — 원인 확인 후 제거
    print("\n=== Held-out 상세 결과 (진단용) ===")
    for detail in exact_match_result["details"]:
        print(f"Q: {detail['question']}")
        print(f"  정답: {detail['gold_answer']}")
        print(f"  생성: {detail['generated_answer']!r}")  # !r로 공백/특수문자까지 노출
        print(f"  질문 activity: {detail['question_activity']}")
        print(f"  label 정답 여부: {detail['is_label_correct']}, slot 정답 여부: {detail['is_slot_copy_correct']}")
        print()

    # 8. 모델 저장
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/korean_model.pt")
    print("모델 저장 완료: checkpoints/korean_model.pt")

    print("=== Instruction Tuning 훈련 루프 종료 ===")


if __name__ == "__main__":
    main()