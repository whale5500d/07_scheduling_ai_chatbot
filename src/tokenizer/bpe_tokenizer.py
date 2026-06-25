from collections import defaultdict, Counter
import re

class BPETokenizer:
    """
    BPE Tokenizer 구현체
    """
    def __init__(self, vocab_size: int, unk_token: str = "<unk>"):
        """
        vocab_size: 최종적으로 만들 어휘 사전의 크기
        unk_token: Out-of-Vacabulary 토큰 (기본값: <unk>)
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {} # 최종 vocabulary
        self.merge_rules = [] # 병합 규칙 (학습된 순서대로 저장)
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, corpus: list[str]):
        """
        BPE 학습 메인 함수
        - corpus: 학습에 사용할 텍스트 리스트
        """
        
        # 1. 초기 word_freq 구성 (단어 -> 문자 단위 + </w>)
        word_freq = {}
        for sentence in corpus:             
            for word in sentence.split(): # 입력된 corpus를 단어 단위로 구분
                # 각 단어를 문자 단위로 쪼갠 후 </w>를 붙여 초기 word_freq 만들기
                word_tokens = ' '.join(list(word)) + ' </w>'
                word_freq[word_tokens] = word_freq.get(word_tokens, 0) + 1

        # 2. 병합 횟수 설정
        # 이전에는 vocab_size - 초기 토큰 수 방식으로 계산했으나,
        # 안정적인 학습을 위해 고정 offset + 최소값을 사용하는 방식으로 변경
        merges_needed = self.vocab_size - 100
        merges_needed = max(50, merges_needed)

        # 3. 병합 반복 수행
        # 루프를 돌면서
        for _ in range(merges_needed):
            # pair 빈도 계산(_get_stats())
            pairs = self._get_stats(word_freq)
            if not pairs: # 목표: 더 이상 병합할 pair가 없을 때까지 반복
                break

            # 가장 빈번한 pair 선택
            best_pair = max(pairs.items(), key=lambda item: item[1])[0]

            # _merge_vocab()로 병합
            word_freq = self._merge_vocab(best_pair, word_freq)

            # merge_rules에 기록
            self.merge_rules.append(best_pair)

        # 3. 최종 vocabulary 및 ID 매핑 생성
        self.vocab = word_freq

        # 4. token_to_id, id_to_token 생성
        # <unk>는 항상 ID 0으로 고정하여 미리 설정
        self.token_to_id = {self.unk_token: 0}
        self.id_to_token = {0: self.unk_token}

        # 5. 개선된 vocabulary 구축
        # TODO: 나중에 더 나은 vocabulary selection 전략을 적용해야 함.
        #       (예: 모든 병합 단계별로 등장한 토큰 기록, frequency 기반 필터링 등)
        all_tokens = set()

        # 6. 현재 word_freq에 있는 모든 서브워드 수집
        for word in word_freq:
            all_tokens.update(word.split())

        # 7. merge_rules에 등장한 모든 토큰도 수집 (더 풍부한 vocabulary를 위해)
        for pair in self.merge_rules:
            all_tokens.add(''.join(pair)) # 병합된 형태 추가

        # token_to_id, id_to_token 생성(토큰을 정렬하여 일관된 순서로 ID 부여)
        sorted_tokens = sorted(all_tokens) # 오름차순

        # 기존 subword들을 1부터 시작해서 추가
        for idx, token in enumerate(sorted_tokens, start=1): # <unk>이 0으로 고정이므로 1부터 시작
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _get_stats(self, vocab: dict) -> dict:
        """
        현재 vocab에서 인접한 pair 빈도 계산
        - return: {(token1, token2): 빈도수, ...}
        """
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += freq
        
        return pairs

    def _merge_vocab(self, pair: tuple, vocab: dict) -> dict:
        """
        가장 빈번한 pair를 병합
        
        args:
            pair: 병합할 두 토큰 (예시: ('l', 'o'))
            vocab: 현재 단어-빈도 딕셔너리

        returns:
            병합이 적용된 새로운 vocab 딕셔너리
        """
        new_vocab = {}

        # pair를 문자열로 변환
        # 예시: ('l', 'o') -> 'l o'(검색용), 'lo' (대체용)
        bigram = ' '.join(pair) # 검색용
        replacement = ''.join(pair) # 대체용

        for word, freq in vocab.items():
            # 단어 내에서 bigram을 replacement로 치환
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = new_vocab.get(new_word, 0) + freq

        return new_vocab

    def encode(self, text: str) -> list[int]:
        """
        텍스트를 BPE 방식으로 토큰화하여 token ID 리스트로 반환

        Known Limitation:
            - <bos>, <eos> 등 특수 토큰에 대한 명시적 처리가 아직 없음.
            - vocabulary 품질이 낮을 경우, 긴 시퀀스가 짧게 압축되는 현상이 발생할 수 있음.
        """
        
        # 1. Pre-tokenization (단어 단위로 나누고 문제 + </w>로 분리)
        words = text.split()
        tokens = []
        for word in words:
            # 예: "low" → ['l', 'o', 'w', '</w>']
            word_tokens = list(word) + ['</w>']
            tokens.extend(word_tokens) # 요소를 평평하게 넣기 위해 append 대신 extend 사용

        # print(f"[encode] 초기 tokens: {tokens}") # 디버그: 초기 상태

        # 2. 학습된 merge_rules 순서대로 병합 적용
        for pair in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 현재 위치에서 pair와 일치하는지 확인
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    # 병합
                    merged = ''.join(pair)
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 3. token_to_id를 사용해 ID로 변환
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id[self.unk_token]) # <unk> ID로 치환

        # print(f"[encode] 최종 token_ids: {token_ids}")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        token ID 리스트를 받아서 원래 텍스트로 복원
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                # 등록되지 않은 ID는 <unk>로 처리
                tokens.append(self.unk_token)

        # 토큰들을 이어붙임
        text = ''.join(tokens)

        # </w>를 공백으로 치환하여 문장 형태로 복원
        text = text.replace("</w>", " ")

        # 앞뒤 공백 제거
        text = text.strip()

        return text