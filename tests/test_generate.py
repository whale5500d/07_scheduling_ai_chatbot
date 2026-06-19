from generate import generate

print("=== Test 1 ===")
print(generate("today the weather is", max_new_tokens=25))

print("\n=== Test 2 ===")
print(generate("hello how are you", max_new_tokens=20))

print("\n=== Test 3 (stop 조건) ===")
print(generate("today the weather is", max_new_tokens=30, stop_sequences=["good"]))