"""
디버깅 스크립트: "9221"을 포함한 chunk가 실제로 몇 번째 인덱스이고,
검색 질문에 대해 몇 위로 검색되는지 확인한다.

트러블슈팅 #8 검증 (DaySync 도메인으로 재현): Fixed-size chunking과
Section-based chunking을 동일한 절차로 각각 실행하여, 두 전략의
"9221" chunk 순위를 비교한다.
"""

from paths import DATA_DIR
from rag_pipeline.document_loader import load_document
from rag_pipeline.chunker import chunk_by_section, chunk_fixed_size
from rag_pipeline.embedder import TextEmbedder
from rag_pipeline.vector_store import InMemoryVectorStore
from rag_pipeline.retriever import cosine_similarity

DATA_PATH = DATA_DIR / "daysync_manual.md"
document = load_document(str(DATA_PATH))

# (질문, 정답을 찾기 위한 키워드) 쌍으로 직접 지정 — 자동 추출 대신 명확성을 우선한다
questions = [
    ("DaySync의 기본 API 포트는 무엇인가?", "9221"),
    ("DaySync의 내부 코드네임은 무엇인가?", "Dawnstar"),
    ("preference_threshold의 기본값은 무엇인가?", "preference_threshold"),
    ("SC-114는 무엇을 의미하는가?", "SC-114"),
]
embedder = TextEmbedder()

for question, target_keyword in questions:
    print(f"\n\n{'=' * 70}")
    print(f"질문: {question}")
    print(f"{'=' * 70}")

    print("\n########## [전략 1] Fixed-size chunking ##########\n")

    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    # 1. 정답 키워드를 포함한 chunk(들)의 인덱스를 모두 찾는다
    target_indices = [i for i, c in enumerate(chunks) if target_keyword.lower() in c.lower()]
    print(f"[점검 1] '{target_keyword}' 키워드를 포함한 chunk 인덱스: {target_indices}")

    print()

    # 2. 전체 임베딩 및 저장
    vectors = embedder.encode(chunks)
    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    # 3. 질문에 대해 모든 chunk의 유사도 점수와 순위를 계산
    query_vector = embedder.encode([question])[0]

    all_scores = []
    for i in range(len(store)):
        assert store.vectors is not None
        score = cosine_similarity(query_vector, store.vectors[i])
        all_scores.append((i, score))

    all_scores.sort(key=lambda pair: pair[1], reverse=True)

    print(f"[점검 2] 전체 chunk 순위 (인덱스, 점수) - 상위 5개만 표시:")
    for rank, (index, score) in enumerate(all_scores[:5], start=1):
        marker = " <== TARGET" if index in target_indices else ""
        print(f"  Rank {rank}: chunk[{index}] score={score:.4f}{marker}")

    print("\n########## [전략 2] Section-based chunking ##########\n")

    chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)

    # 1. 정답 관련 키워드를 포함한 chunk(들)의 인덱스를 모두 찾는다
    target_indices = [i for i, c in enumerate(chunks) if target_keyword.lower() in c.lower()]
    print(f"[점검 1] '{target_keyword}' 키워드를 포함한 chunk 인덱스: {target_indices}")

    print()

    # 2. 전체 임베딩 및 저장
    vectors = embedder.encode(chunks)
    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    # 3. 질문에 대해 모든 chunk의 유사도 점수와 순위를 계산
    query_vector = embedder.encode([question])[0]

    all_scores = []
    for i in range(len(store)):
        assert store.vectors is not None
        score = cosine_similarity(query_vector, store.vectors[i])
        all_scores.append((i, score))

    all_scores.sort(key=lambda pair: pair[1], reverse=True)

    print(f"[점검 2] 전체 chunk 순위 (인덱스, 점수) - 상위 5개만 표시:")
    for rank, (index, score) in enumerate(all_scores[:5], start=1):
        marker = " <== TARGET" if index in target_indices else ""
        print(f"  Rank {rank}: chunk[{index}] score={score:.4f}{marker}")