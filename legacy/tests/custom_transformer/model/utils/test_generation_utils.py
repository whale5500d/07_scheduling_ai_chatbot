from custom_transformer.model.utils.generation_utils import trim_after_eos

def test_trim_after_eos():
    print("=== trim_after_eos 테스트 시작 ===\n")

    # === 테스트 1: eos가 중간에 있는 경우 ===
    print("[테스트 1] eos가 중간에 있는 경우")
    token_ids = [5, 6, 7, 1, 8, 9]  # 1 = eos
    result = trim_after_eos(token_ids, eos_token_id=1)
    print(f"입력: {token_ids}")
    print(f"결과: {result}")

    assert result == [5, 6, 7], "eos 이전까지만 남아야 합니다."

    print("\n✅ 테스트 1 통과: eos 중간 위치 처리 정상 동작\n")

    # === 테스트 2: eos가 없는 경우 ===
    print("[테스트 2] eos가 없는 경우")
    token_ids = [5, 6, 7, 8]
    result = trim_after_eos(token_ids, eos_token_id=1)
    print(f"입력: {token_ids}")
    print(f"결과: {result}")

    assert result == [5, 6, 7, 8], "eos가 없으면 원본 그대로 반환해야 합니다."

    print("\n✅ 테스트 2 통과: eos 미등장 처리 정상 동작\n")

    # === 테스트 3: eos가 맨 앞에 있는 경우 ===
    print("[테스트 3] eos가 맨 앞에 있는 경우")
    token_ids = [1, 5, 6, 7]
    result = trim_after_eos(token_ids, eos_token_id=1)
    print(f"입력: {token_ids}")
    print(f"결과: {result}")

    assert result == [], "eos가 맨 앞에 있으면 빈 리스트를 반환해야 합니다."

    print("\n✅ 테스트 3 통과: eos 맨 앞 위치 처리 정상 동작\n")

    # === 테스트 4: eos가 여러 번 등장하는 경우 ===
    print("[테스트 4] eos가 여러 번 등장하는 경우")
    token_ids = [5, 1, 6, 1, 7]
    result = trim_after_eos(token_ids, eos_token_id=1)
    print(f"입력: {token_ids}")
    print(f"결과: {result}")

    assert result == [5], "첫 번째 eos 위치를 기준으로 잘라야 합니다."

    print("\n✅ 테스트 4 통과: eos 다중 등장 처리 정상 동작\n")

    # === 테스트 5: 빈 입력 ===
    print("[테스트 5] 빈 입력")
    result = trim_after_eos([], eos_token_id=1)
    print(f"입력: []")
    print(f"결과: {result}")

    assert result == [], "빈 입력은 빈 리스트를 반환해야 합니다."

    print("\n✅ 테스트 5 통과: 빈 입력 처리 정상 동작\n")

    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_trim_after_eos()