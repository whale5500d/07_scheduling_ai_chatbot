"""
Step C-4: Graph RAG — Entity/Relation Extraction

책임(Responsibility): 텍스트(문서)를 받아, LLM에게 "RELATION: A | relation | B"
형식의 고정 패턴으로 관계를 말하게 한 뒤, 정규표현식으로 파싱하여
그래프 자료구조({"nodes": [...], "edges": [...]})로 변환한다.

설계 결정 (트러블슈팅 #8/#12에서 배운 원리 적용):
- 본문 전체를 분석하게 하지 않고, 이미 구조화된 "요약 테이블" 섹션을
  추출 대상으로 우선한다 — 정보가 정리되어 있을수록 추출 품질이 좋다.
- LLM에게 자유 형식 자연어 대신, 고정된 패턴("RELATION: ...")으로
  답하도록 요청하여 정규표현식 파싱의 안정성을 높인다.

그래프 표현 방식 ("근본에서 확장" 원칙 — 전용 GraphDB 없이 가장 단순한 형태로 시작):
- 노드: {"id": str, "type": str}
- 엣지: {"source": str, "relation": str, "target": str}
"""

import re

from rag_pipeline.generator import TextGenerator

RELATION_PATTERN = re.compile(r"RELATION:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", re.MULTILINE)


def extract_relations(summary_table_text: str, generator: TextGenerator) -> list[tuple[str, str, str]]:
    """
    구조화된 요약 텍스트(표 등)를 받아, LLM에게 관계를 고정 패턴으로
    말하게 하고, 그 결과를 정규표현식으로 파싱하여 (주어, 관계, 목적어)
    튜플의 리스트로 반환한다.

    Args:
        summary_table_text: 관계 정보가 담긴 구조화된 텍스트 (예: 마크다운 표)
        generator: 추출에 사용할 TextGenerator 인스턴스

    Returns:
        (source, relation, target) 튜플의 리스트

    Raises:
        ValueError: summary_table_text가 빈 문자열일 경우
    """
    if not summary_table_text.strip():
        raise ValueError("summary_table_text가 비어 있습니다. 추출할 내용이 없습니다.")

    prompt = f"""당신은 관계 추출(relation extraction) 어시스턴트입니다. 아래 표를 읽고,
표에서 찾은 모든 관계를 정확히 아래 형식으로, 한 줄에 하나씩, 다른 텍스트 없이 출력하세요:

RELATION: <엔티티 A> | <관계 유형> | <엔티티 B>

관계 유형은 다음 값만 사용하세요: prefers_activity, managed_by, changed_config, experienced_conflict

표:
{summary_table_text}

RELATION 줄만 출력하세요. 설명을 추가하지 마세요.
"""
    response = generator.generate(prompt, max_new_tokens=300)

    matches = RELATION_PATTERN.findall(response)
    relations = [(source.strip(), relation.strip(), target.strip()) for source, relation, target in matches]

    return relations


def build_graph(relations: list[tuple[str, str, str]]) -> dict:
    """
    (source, relation, target) 튜플 리스트를 받아, 노드/엣지로 구성된
    그래프 자료구조로 변환한다.

    Args:
        relations: extract_relations()의 반환값

    Returns:
        {"nodes": list[dict], "edges": list[dict]}
    """
    node_ids: set[str] = set()
    edges: list[dict] = []

    for source, relation, target in relations:
        node_ids.add(source)
        node_ids.add(target)
        edges.append({"source": source, "relation": relation, "target": target})

    nodes = [{"id": node_id} for node_id in sorted(node_ids)]

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    from paths import DATA_DIR
    from rag_pipeline.document_loader import load_document

    sample_path = DATA_DIR / "daysync_team_records.md"
    document = load_document(str(sample_path))

    # 요약 테이블(섹션 4)만 추출 대상으로 사용한다 (가장 구조화된 부분)
    summary_section = document.split("## 4. Summary Table")[1]

    print("[Gemma 4 E2B-it 로딩 중...]")
    generator = TextGenerator()

    relations = extract_relations(summary_section, generator)

    print(f"\n[추출된 관계 {len(relations)}개]")
    for source, relation, target in relations:
        print(f"  {source} --{relation}--> {target}")

    graph = build_graph(relations)
    print(f"\n[그래프] 노드 {len(graph['nodes'])}개, 엣지 {len(graph['edges'])}개")