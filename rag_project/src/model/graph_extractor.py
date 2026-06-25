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

from model.generator import TextGenerator

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

    prompt = f"""You are a relation extraction assistant. Read the table below and output
every relationship you find using EXACTLY this format, one per line, with no extra text:

RELATION: <entity A> | <relation_type> | <entity B>

Use these relation_type values only: uses_engine_mode, managed_by, changed_config, experienced_error

Table:
{summary_table_text}

Output only RELATION lines. Do not add explanations.
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
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag_pipeline.document_loader import load_document

    sample_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_team_incidents.md"
    )
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