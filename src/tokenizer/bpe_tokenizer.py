from collections import defaultdict, Counter
import re

class BPETokenizer:
    """
    BPE Tokenizer 구현체

    Known Limitation (2026.06.19 기준):
        - train() 메서드의 vocabulary 품질이 아직 충분하지 않음.
          전체 단어가 통째로 토큰으로 남거나, 일관성 없는 서브워드 분해가 발생할 수 있음.
        - 이로 인해 encode() 결과가 기대보다 짧거나, 의미 있는 subword 단위로 분해되지 않을 수 있음.
        - 현재는 decode()는 정상 동작하지만, 생성 품질 저하의 주요 원인이 됨.
        - 개선 방향: merges_needed 계산 방식 재검토 + vocabulary 구축 로직 정교화
          (RESTROSPECTIVE.md 트러블슈팅 9, 10 참조)
    """
    def __init__(self, vocab_size: int):
        """
        vocab_size: 최종적으로 만들 어휘 사전의 크기
        """
        self.vocab_size = vocab_size
        self.vocab = {} # 최종 vocabulary
        self.merge_rules = [] # 병합 규칙 (학습된 순서대로 저장)
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, corpus: list[str]):
        """
        BPE 학습 메인 함수
        - corpus: 학습에 사용할 텍스트 리스트
        """
        
        # TODO: 현재 train()은 BPE의 기본적인 동작은 수행하지만,
        #       최종 vocabulary의 품질이 아직 만족스럽지 않음.
        #       특히 token_to_id에 전체 단어(low</w>, hello</w> 등)가 많이 남아 있고,
        #       일관된 서브워드 단위로 vocabulary가 구성되지 않는 문제가 있음.
        #       나중에 vocabulary 구축 로직을 더 체계적으로 개선할 필요가 있음.
        #       (예: 모든 병합 단계에서 등장한 토큰을 기록하거나,
        #        더 정교한 vocabulary selection 전략 적용)
        # 1. 초기 word_freq 구성
        word_freq = {}
        for sentence in corpus:             
            for word in sentence.split(): # 입력된 corpus를 단어 단위로 구분
                # 각 단어를 문자 단위로 쪼갠 후 </w>를 붙여 초기 word_freq 만들기
                word_tokens = ' '.join(list(word)) + ' </w>'
                word_freq[word_tokens] = word_freq.get(word_tokens, 0) + 1
                
        # 2. 목표 병합 횟수 계산 (대략적인 추정)
        # word_freq_size에 도달하기 위해 대략 몇 번 병합해야 하는지 계산
        # TODO: 현재 merges_needed 계산 방식(len(word_freq) 기반)은
        #       BPE 학습의 본질과 잘 맞지 않을 수 있음.
        #       더 나은 방식(예: 고정된 병합 횟수 또는 vocabulary 성장률 기반)으로
        #       개선하는 것을 고려해야 함.
        initial_tokens = set()
        for word in word_freq:
            initial_tokens.update(word.split())

        # TODO: BPE 학습 로직 개선 필요 (2026.06.19)
        # 현재 merges_needed를 len(initial_tokens) 기준으로 계산하고 있으나,
        # 이는 BPE의 본질(정보 이론 기반 압축, MDL 최소화)과 잘 맞지 않을 수 있음.
        # 또한 최종 vocabulary 구축 방식이 ad-hoc하여 품질이 낮음.
        # → 향후 고정된 병합 횟수 또는 vocabulary 성장률 기반 제어로 변경 검토 필요.
        merges_needed = max(0, self.vocab_size - len(initial_tokens))

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

        # 4. 개선된 vocabulary 구축
        # TODO: 현재 vocabulary 구축 방식은 ad-hoc(임시방편)임.
        #       word_freq의 현재 상태 + merge_rules에서 토큰을 수집하고 있는데,
        #       이는 BPE 학습 과정에서 만들어진 모든 유의미한 서브워드를
        #       체계적으로 수집하는 방법이 아님.
        #       나중에 더 나은 vocabulary selection 전략을 적용해야 함.
        #       (예: 모든 병합 단계별로 등장한 토큰 기록, frequency 기반 필터링 등)
        all_tokens = set()

        # 5. 현재 word_freq에 있는 모든 서브워드 수집
        for word in word_freq:
            all_tokens.update(word.split())

        # 5. merge_rules에 등장한 모든 토큰도 수집 (더 풍부한 vocabulary를 위해)
        for pair in self.merge_rules:
            all_tokens.add(''.join(pair)) # 병합된 형태 추가

        # token_to_id, id_to_token 생성(토큰을 정렬하여 일관된 순서로 ID 부여)
        sorted_tokens = sorted(all_tokens) # 오름차순

        self.token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}        
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

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
            - OOV 토큰을 -1로 처리하고 있음. 이는 모델 입력 시 문제가 될 수 있음.
            - <unk>, <bos>, <eos> 등 특수 토큰에 대한 명시적 처리가 아직 없음.
            - vocabulary 품질이 낮을 경우, 긴 시퀀스가 짧게 압축되는 현상이 발생할 수 있음.
        """
        
        # 1. Pre-tokenization (단어 단위로 나누고 문제 + </w>로 분리)
        words = text.split()
        tokens = []
        for word in words:
            # 예: "low" → ['l', 'o', 'w', '</w>']
            word_tokens = list(word) + ['</w>']
            tokens.extend(word_tokens)

        print(f"[encode] 초기 tokens: {tokens}") # 디버그: 초기 상태

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
                token_ids.append(-1)   # OOV 표시 (나중에 <unk> 처리 가능)

        print(f"[encode] 최종 token_ids: {token_ids}")
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
                # 등록되지 않은 ID는 <unk>로 처리 (필요시 다른 방식으로 변경 가능)
                tokens.append("<unk>")

        # 토큰들을 이어붙임
        text = ''.join(tokens)

        # </w>를 공백으로 치환하여 문장 형태로 복원
        text = text.replace("</w>", " ")

        # 앞뒤 공백 제거
        text = text.strip()

        return text