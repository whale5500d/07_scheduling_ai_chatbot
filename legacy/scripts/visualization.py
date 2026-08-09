from langgraph_pipeline_2.graph import graph

_NODE_LABELS = {
    "judge_schedule": "일정 질문 판단 (judge_schedule)",
    "judge_response": "응답 긍정/부정 판단 (judge_response)",
    "judge_date": "날짜 정규화 (judge_date)",
    "save_rdb": "RDB 저장 (save_rdb)",
    "call_model": "LLM 판정 (call_model)",
    "tools": "사례 검색 (tools)",
    "confirm_save": "저장 확인, human-in-the-loop (confirm_save)",
}

def build_labeled_mermaid() -> str:
    mermaid_text = graph.get_graph().draw_mermaid()

    for node_id, label in _NODE_LABELS.items():
        mermaid_text = mermaid_text.replace(
            f"{node_id}({node_id})",
            f'{node_id}("{label}")'
        )

    return mermaid_text

if __name__ == "__main__":
    print(build_labeled_mermaid())
