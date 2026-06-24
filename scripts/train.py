# scripts/train.py

"""
최소 학습 루프 (2단계)
- random weights 상태의 모델을 간단히 학습시켜보는 것이 목적
- 작은 데이터로 시작해서 점차 확장하는 방식으로 진행
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer_model import TransformerLanguageModel
from tokenizer.bpe_tokenizer import BPETokenizer


def main():
    print("=== 최소 학습 루프 시작 ===\n")

    # 학습 데이터 준비 (작은 규모로 시작)
    train_corpus = [
        "hello how are you",
        "i am fine thank you",
        "what is your name",
        "my name is gpt",
        "today the weather is good",
        "i like to play soccer",
        "she is reading a book",
        "he went to school yesterday",
        "we are learning python",
        "this is a simple example",
        "the cat is on the mat",
        "i want to eat pizza",
        "she likes to dance",
        "he plays the guitar well",
        "they are watching a movie",
    ]

    # BPE Tokenizer 초기화 및 학습
    VOCAB_SIZE = 200 # 초기값으로 150~300 정도, 추후 학습하면서 조정하는 방식으로 진행
    # 50~100으로 너무 작으면, <unk>가 많이 발생
    # 500~1000으로 너무 크면, 작은 데이터에서는 의미없는 토큰이 많아질 수 있음
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(train_corpus)
    print(f"Tokenizer Vocabulary size: {len(tokenizer.token_to_id)}")

    # TransformerLanguageModel 초기화
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

    # Loss Function, Optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 루프 구현 (Next Token Prediction)
    """
        현재 학습 루프의 특징
        - 데이터가 매우 작기 때문에 문장 단위로 학습하도록 구현
        - 배치 처리는 아직 하지 않고, 한 문장씩 학습 (최소 구현)
        - num_epochs = 50으로 설정 (나중에 조정 가능)
    """
    num_epochs = 50

    for epoch in range(num_epochs):
        total_loss = 0

        for sentence in train_corpus:
            # 1. 문장을 토큰화
            input_ids = tokenizer.encode(sentence)

            # 너무 짧은 문장은 제외 (최소 2토큰 이상)
            if len(input_ids) < 2:
                continue

            input_ids = torch.tensor([input_ids])  # (1, seq_len)

            # 2. Next Token Prediction을 위한 shift
            # input: [t0, t1, t2, ..., t_{n-1}]
            # label: [t1, t2, ..., t_n]
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            # 3. Forward
            outputs = model(inputs)  # (1, seq_len-1, vocab_size)

            # 4. Loss 계산
            loss = criterion(
                outputs.view(-1, vocab_size),   # (seq_len-1, vocab_size)
                labels.view(-1)                 # (seq_len-1)
            )

            # 5. Backward + Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_corpus)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.4f}")

    # TODO 6: 학습 후 generate 결과 확인

    print("\n=== 학습 루프 종료 ===")


if __name__ == "__main__":
    main()