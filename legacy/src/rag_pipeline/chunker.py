"""
Step A-2: Chunking (Fixed-size 전략 + Section-based 전략)

책임(Responsibility): raw text(string)를 받아 chunk 리스트(list[str])를 반환한다.
이 단계에서는 벡터화(embedding)를 하지 않는다.

전략 1: Fixed-size chunking (chunk_fixed_size)
- 텍스트의 구조(헤더, 문단 등)를 고려하지 않고, 정해진 문자 수(chunk_size) 단위로 자른다.
- chunk_overlap을 두어, chunk 경계에서 문장이 잘려 의미가 손실되는 것을 완화한다.

전략 2: Section-based chunking (chunk_by_section)
- 마크다운 `##` 헤더를 기준으로 섹션 단위로 자른다 — 서로 무관한 주제가 한 chunk에
  섞이는 것을 방지한다 (트러블슈팅 #8에서 발견된 fixed-size chunking의 한계).
- 섹션이 chunk_size보다 길면, 그 섹션 내부에서만 chunk_fixed_size()를 재사용해
  추가로 분할한다 (하이브리드 방식) — 섹션 경계는 절대 넘지 않는다.
"""

import re


def chunk_fixed_size(
    raw_text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    raw text를 고정 길이(chunk_size) 단위로 분할한다. 인접 chunk는 chunk_overlap만큼 겹친다.

    Args:
        raw_text: Document Loading 단계에서 반환된 원본 텍스트
        chunk_size: 한 chunk에 포함될 최대 문자 수
        chunk_overlap: 인접한 두 chunk가 공유하는 문자 수

    Returns:
        고정 길이로 분할된 chunk 리스트

    Raises:
        ValueError: chunk_overlap이 chunk_size보다 크거나 같을 경우
                    (이 경우 step이 0 이하가 되어 무한 루프 발생)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 합니다."
        )

    step = chunk_size - chunk_overlap  # 다음 chunk 시작 위치까지의 이동 거리

    chunks = []
    start = 0
    text_length = len(raw_text)

    while start < text_length:
        end = start + chunk_size
        chunk = raw_text[start:end].strip()

        if chunk:  # 공백만 남은 chunk는 제외
            chunks.append(chunk)

        start += step

    return chunks


def chunk_by_section(
    raw_text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    section_pattern: str = r"(?=^## )",
) -> list[str]:
    """
    raw text를 마크다운 `##` 헤더 기준으로 섹션 단위로 분할한다.
    섹션이 chunk_size보다 길면, 그 섹션 내부에서만 chunk_fixed_size()로 추가 분할한다.

    Args:
        raw_text: Document Loading 단계에서 반환된 원본 텍스트
        chunk_size: 섹션이 이 길이를 초과하면 내부적으로 추가 분할한다
        chunk_overlap: 내부 추가 분할 시 사용할 chunk_overlap (chunk_fixed_size에 전달)
        section_pattern: 섹션 경계를 찾기 위한 정규식. 기본값은 `## `로 시작하는 줄
                          (lookahead를 사용해 구분자 자체는 다음 섹션에 포함시킨다)

    Returns:
        섹션 경계를 보존하며 분할된 chunk 리스트. 섹션 경계는 절대 넘지 않는다.

    Raises:
        ValueError: raw_text가 비어 있을 경우
    """
    if not raw_text.strip():
        raise ValueError("raw_text가 비어 있습니다. 분할할 텍스트가 필요합니다.")

    # re.MULTILINE: '^'가 각 줄의 시작을 의미하도록 설정 (문서 전체의 시작이 아니라)
    sections = re.split(section_pattern, raw_text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            # 섹션이 충분히 짧으면 그대로 하나의 chunk로 사용
            chunks.append(section)
        else:
            # 섹션이 chunk_size를 초과하면, 그 섹션 내부에서만 fixed-size로 재분할한다.
            # 다른 섹션과는 절대 섞이지 않는다 (섹션 경계 보존).
            sub_chunks = chunk_fixed_size(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks.extend(sub_chunks)

    return chunks


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document

    sample_path = DATA_DIR / "daysync_manual.md"
    document = load_document(str(sample_path))

    print("=" * 60)
    print("[전략 1] Fixed-size chunking")
    print("=" * 60)
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)
    print(f"총 chunk 개수: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        preview = chunk[:60].replace("\n", " ")
        print(f"--- Chunk {i} (길이: {len(chunk)}자) ---\n{preview}...\n")

    print("=" * 60)
    print("[전략 2] Section-based chunking")
    print("=" * 60)
    section_chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)
    print(f"총 chunk 개수: {len(section_chunks)}\n")
    for i, chunk in enumerate(section_chunks):
        preview = chunk[:60].replace("\n", " ")
        print(f"--- Chunk {i} (길이: {len(chunk)}자) ---\n{preview}...\n")