from pathlib import Path


def load_document(file_path: str) -> str:
    """
    단일 텍스트 파일(.txt, .md)을 읽어 raw text로 반환한다.

    Args:
        file_path: 읽을 파일의 경로

    Returns:
        파일 전체 내용을 담은 문자열(raw text)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 파일 내용이 비어 있을 경우
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    raw_text = path.read_text(encoding="utf-8")

    if not raw_text.strip():
        raise ValueError(f"파일 내용이 비어 있습니다: {file_path}")

    return raw_text


if __name__ == "__main__":
    # 동작 확인용 테스트
    sample_path = "data/nimbusflow_manual.md"
    document = load_document(sample_path)

    print(f"[로딩 성공] 총 문자 수: {len(document)}")
    print(f"[미리보기]\n{document[:300]}")