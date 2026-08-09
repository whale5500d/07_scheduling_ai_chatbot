"""
langgraph_pipeline.utils.nodes: Agent 그래프 노드 함수 정의
"""
from __future__ import annotations


def make_call_model_node(llm):
    """call_model 노드 함수를 반환합니다.

    state["messages"] 전체(대화 히스토리)를 LLM에 전달하고,
    반환된 AIMessage를 {"messages": [response]}로 감싸 반환한다.
    add_messages reducer가 이 새 메시지를 히스토리에 append한다.
    """
    def call_model(state) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    return call_model