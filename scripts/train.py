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

    # TODO 3: TransformerLanguageModel 초기화
    # TODO 4: Loss Function, Optimizer 정의
    # TODO 5: 학습 루프 구현 (Next Token Prediction)
    # TODO 6: 학습 후 generate 결과 확인

    print("\n=== 학습 루프 종료 ===")


if __name__ == "__main__":
    main()