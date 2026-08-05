# 4. run graph
from typing import cast
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph_pipeline_2.graph import graph
from langgraph_pipeline_2.state import AgentState

# print
def print_result(label: str, result: AgentState):
    print(f"\n[{label}]")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content}")
    print(f"  pending_question: {result.get('pending_question')}")
    print("===================================================")

config: RunnableConfig = {"configurable": {"thread_id": "test-thread-1"}}

turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content="응 좋아")],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("2턴", turn2)

turn3 = graph.invoke({
        "messages": [HumanMessage(content="모레 영화 볼래?")],
    }, config=config)
turn3 = cast(AgentState, turn3)
print_result("3턴", turn3)