# 4. run graph
from typing import cast
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph_pipeline_2.graph import graph
from langgraph_pipeline_2.state import AgentState

RESPONSE_TEST = {
    "content": "응 좋아",
    "query_request": True
}

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
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("2턴", turn2)

turn3 = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
turn3 = cast(AgentState, turn3)
print_result("3턴", turn3)

# def print_stream_chunk(chunk: dict):
#     for node_name, updates in chunk.items():
#         print(f"[노드: {node_name}]")

#         if node_name == "__interrupt__":
#             for interrupt in updates:
#                 print(f"  value: {interrupt.value}")
#             print()
#             continue

#         for key, value in updates.items():
#             if key == "messages":
#                 print("  messages:")
#                 for message in value:
#                     print(f"    - {type(message).__name__}: {message.content}")
#             else:
#                 print(f"  {key}: {value}")
#         print()

# def print_state_snapshot(snapshot):
#     node_label = snapshot.next[0] if snapshot.next else "완료"
#     print(f"[체크포인트: {node_label}]")
#     print(f"  다음 노드: {snapshot.next}")
#     messages = snapshot.values.get("messages", [])
#     if messages:
#         print("  messages:")
#         for message in messages:
#             print(f"    - {type(message).__name__}: {message.content}")
#     else:
#         print("  messages: (없음)")
#     print()


# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="내일 산책 할래?")]},
#     config=config,
#     stream_mode="updates"
# ):
#     print_stream_chunk(chunk)

# print("\n[get_state_history 조회]")
# for snapshot in graph.get_state_history(config):
#     print_state_snapshot(snapshot)