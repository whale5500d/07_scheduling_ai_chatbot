# scripts/train.py

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

from model.transformer_model import TransformerLanguageModel
from tokenizer.bpe_tokenizer import BPETokenizer
from data_utils.qa_collate import build_qa_training_pair, IGNORE_INDEX


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
    qa_pairs = load_qa_pairs("scripts/data/korean_qa.txt")
    print(f"학습 데이터 수: {len(qa_pairs)} 쌍")

    # 2. BPE Tokenizer 초기화 및 훈련
    # tokenizer 학습은 여전히 문장 단위 리스트를 입력으로 받으므로,
    # 질문과 답변을 모두 풀어서(flatten) corpus로 사용
    flat_corpus = [question for question, _ in qa_pairs] + [answer for _, answer in qa_pairs]

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

    for epoch in range(num_epochs):
        total_loss = 0
        num_skipped = 0

        for question, answer in qa_pairs:
            question_ids = tokenizer.encode(question)
            answer_ids = tokenizer.encode(answer)

            # 질문 또는 답변이 토큰화 후 비어 있으면 해당 쌍은 건너뜀
            if len(question_ids) == 0 or len(answer_ids) == 0:
                num_skipped += 1
                continue

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
                top_k=50
            )

            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}\n")

    # 7. 모델 저장
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/korean_model.pt")
    print("모델 저장 완료: model/korean_model.pt")

    print("=== Instruction Tuning 훈련 루프 종료 ===")


if __name__ == "__main__":
    main()