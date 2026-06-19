from tokenizer.bpe_tokenizer import BPETokenizer

def test_bpe_tokenizer():
    print("=== BPE Tokenizer 테스트 시작 ===\n")

    # ======= 테스트 1 =======
    print("\n========== 테스트 1: _get_stats() ==========")
    vocab = {
        'l o w</w>': 5,
        'l o w e r</w>': 2,
    }
    tokenizer = BPETokenizer(vocab_size=100)
    stats = tokenizer._get_stats(vocab)
    print("✅ 테스트 1 통과: _get_stats 동작\n", stats)

    # ======= 테스트 2 =======
    print("\n========== 테스트 2: _merge_vocab() ==========")

    vocab = {
        'l o w</w>': 5,
        'l o w e r</w>': 2,
    }
    tokenizer = BPETokenizer(vocab_size=100)
    new_vocab = tokenizer._merge_vocab(('l', 'o'), vocab)     # 'l o' pair를 병합
    print("✅ 테스트 2 통과: _merge_vocab 동작\n", new_vocab)

    # ======= 테스트 3: train() =======
    print("\n========== 테스트 3: train() ==========")
    corpus = [
        "low lower lowest",
        "new newer newest",
        "low low low",
        "new new new",
        "hello world hello world"
    ]

    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(corpus)

    print(f"최종 vocab 크기     : {len(tokenizer.vocab)}")
    print(f"merge_rules 개수    : {len(tokenizer.merge_rules)}")
    print(f"token_to_id 개수    : {len(tokenizer.token_to_id)}")
    print(f"id_to_token 개수    : {len(tokenizer.id_to_token)}")

    print("\n[merge_rules 상위 8개]")
    print(tokenizer.merge_rules[:8])

    print("\n[token_to_id 상위 10개]")
    for i, (token, idx) in enumerate(tokenizer.token_to_id.items()):
        if i >= 10:
            break
        print(f"  {token}: {idx}")

    print("\n✅ 테스트 3 통과: train() 동작\n")

    # ======= 테스트 4: encode() =======
    print("\n========== 테스트 4: encode() ==========")
    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(["low lower lowest", "new newer newest"])

    print(tokenizer.encode("low"))
    print(tokenizer.encode("lower"))
    print(tokenizer.encode("hello"))

    print("\n✅ 테스트 4 통과: encode() 동작\n")

    # ======= 테스트 5: encode + decode =======
    print("\n========== 테스트 5: encode + decode ==========")

    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(["low lower lowest", "new newer newest", "hello world"])

    test_texts = ["low", "lower", "hello", "world"]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"원본: {text:10} → 인코딩: {encoded} → 디코딩: {decoded}")

    print("\n✅ 테스트 5 통과: encode + decode 동작\n")
    
    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_bpe_tokenizer()