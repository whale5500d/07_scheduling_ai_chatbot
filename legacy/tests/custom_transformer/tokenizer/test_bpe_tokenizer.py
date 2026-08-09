from custom_transformer.tokenizer.bpe_tokenizer import BPETokenizer

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

    # ======= 테스트 4: encode() - 학습 데이터 처리 =======
    print("\n========== 테스트 4: encode() - 학습 데이터 처리 ==========")
    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(["low lower lowest", "new newer newest"])

    print(tokenizer.encode("low"))
    print(tokenizer.encode("lower"))
    print(tokenizer.encode("hello"))

    print("\n✅ 테스트 4 통과: encode() - 학습 데이터 처리 동작\n")

    # ======= 테스트 4: encode() - OOV 처리 테스트 =======
    print("\n========== 테스트 4: encode() - OOV 처리 테스트 ==========")
    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(["hello world", "low lower"])

    # 학습되지 않은 단어
    result = tokenizer.encode("hello apple banana")

    # <unk> (ID 0)이 포함되어 있는지 확인
    assert 0 in result, "OOV가 <unk>로 치환되지 않았습니다."

    print("\n✅ 테스트 4 통과: encode() - OOV 처리 테스트 통과\n")

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

    # ======= 테스트 6: <eos> 토큰 예약 =======
    print("\n========== 테스트 6: <eos> 토큰 예약 ==========")

    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(["low lower lowest", "new newer newest"])

    print(f"unk_token: {tokenizer.unk_token} -> id {tokenizer.token_to_id[tokenizer.unk_token]}")
    print(f"eos_token: {tokenizer.eos_token} -> id {tokenizer.token_to_id[tokenizer.eos_token]}")

    # <unk>=0, <eos>=1로 고정 예약되는지 확인
    assert tokenizer.token_to_id[tokenizer.unk_token] == 0, "<unk>이 ID 0으로 고정되지 않았습니다."
    assert tokenizer.token_to_id[tokenizer.eos_token] == 1, "<eos>가 ID 1로 고정되지 않았습니다."

    # id_to_token 역매핑도 정상인지 확인
    assert tokenizer.id_to_token[0] == tokenizer.unk_token, "id_to_token[0]이 <unk>으로 역매핑되지 않았습니다."
    assert tokenizer.id_to_token[1] == tokenizer.eos_token, "id_to_token[1]이 <eos>로 역매핑되지 않았습니다."

    # 일반 서브워드 토큰은 ID 2부터 시작해야 함
    non_special_ids = [
        idx for token, idx in tokenizer.token_to_id.items()
        if token not in (tokenizer.unk_token, tokenizer.eos_token)
    ]
    assert all(idx >= 2 for idx in non_special_ids), "서브워드 토큰이 ID 2 이전 영역을 침범했습니다."

    print("\n✅ 테스트 6 통과: <eos> 토큰 예약 동작\n")
    
    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_bpe_tokenizer()