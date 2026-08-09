# 4. run graph
from typing import cast
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph_pipeline_2.graph import graph
from langgraph_pipeline_2.state import AgentState

# print
def print_result(label: str, result: AgentState):
    print(f"\n[{label}]")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content}")
    print(f"  pending_question: {result.get('pending_question')}")
    print("===================================================")

# 케이스 1 - 일정 질문 X - 즉시 END
config: RunnableConfig = {"configurable": {"thread_id": "case-1"}}
case1_result = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할거야")],
        "pending_question": None
    }, config=config)
case1_result = cast(AgentState, case1_result)
print_result("case1", case1_result)

# 케이스 2 - 일정 질문 O, 긍정 응답, 날짜 O, 저장 승인 - judge_date, save_rdb 진행
config: RunnableConfig = {"configurable": {"thread_id": "case-2"}}
RESPONSE_TEST = {
    "content": "응 좋아",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case2 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case2 - 2턴", turn2)

case2_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case2_result = cast(AgentState, case2_result)
print_result("case2 - 3턴", case2_result)

# 케이스 3 - 일정 질문 O, 긍정 응답, 날짜 O, 저장 거부 - END
config: RunnableConfig = {"configurable": {"thread_id": "case-3"}}
RESPONSE_TEST = {
    "content": "응 좋아",
    "query_request": False
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case3 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case3 - 2턴", turn2)

case3_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case3_result = cast(AgentState, case3_result)
print_result("case3 - 3턴", case3_result)

# 케이스 4 - 일정 질문 O, 긍정 응답, 날짜 X, 저장 승인 - judge_date, save_rdb 진행
config: RunnableConfig = {"configurable": {"thread_id": "case-4"}}
RESPONSE_TEST = {
    "content": "응 좋아",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case4 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case4 - 2턴", turn2)

case4_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case4_result = cast(AgentState, case4_result)
print_result("case4 - 3턴", case4_result)

# 케이스 5 - 일정 질문 O, 긍정 응답, 날짜 X, 저장 거부 - END
config: RunnableConfig = {"configurable": {"thread_id": "case-5"}}
RESPONSE_TEST = {
    "content": "응 좋아",
    "query_request": False
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case5 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case5 - 2턴", turn2)

case5_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case5_result = cast(AgentState, case5_result)
print_result("case5 - 3턴", case5_result)

# 케이스 6 - 일정 질문 O, 애매한 응답(긍정으로 추론 가능), 날짜 O, 저장 승인 - saved_rdb 진행
config: RunnableConfig = {"configurable": {"thread_id": "case-6"}}
RESPONSE_TEST = {
    "content": "비 안 오면 가자",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case6 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case6 - 2턴", turn2)

case6_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case6_result = cast(AgentState, case6_result)
print_result("case6 - 3턴", case6_result)

# 케이스 7 - 일정 질문 O, 애매한 응답(긍정으로 추론 가능), 날짜 O, 저장 거부 - END
config: RunnableConfig = {"configurable": {"thread_id": "case-7"}}
RESPONSE_TEST = {
    "content": "비 안 오면 가자",
    "query_request": False
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case7 - 1턴", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
turn2 = cast(AgentState, turn2)
print_result("case7 - 2턴", turn2)

case7_result = graph.invoke(Command(resume=RESPONSE_TEST['query_request']), config=config)
case7_result = cast(AgentState, case7_result)
print_result("case7 - 3턴", case7_result)

# 케이스 8 - 일정 질문 O, 부정 응답 - END
config: RunnableConfig = {"configurable": {"thread_id": "case-8"}}
RESPONSE_TEST = {
    "content": "아니",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case8 - 1턴", turn1)

case8_result = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
case8_result = cast(AgentState, case8_result)
print_result("case8 - 2턴", case8_result)

# 케이스 9 - 일정 질문 O, 애매한 응답(부정으로 추론 가능) - END
config: RunnableConfig = {"configurable": {"thread_id": "case-9"}}
RESPONSE_TEST = {
    "content": "꼭 가야 돼?",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case9 - 1턴", turn1)

case9_result = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
case9_result = cast(AgentState, case9_result)
print_result("case9 - 2턴", case9_result)

# 케이스 10 - 일정 질문 O, 애매한 응답(추론 불가능) - END
config: RunnableConfig = {"configurable": {"thread_id": "case-10"}}
RESPONSE_TEST = {
    "content": "배고파",
    "query_request": True
}
turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
turn1 = cast(AgentState, turn1)
print_result("case10 - 1턴", turn1)

case10_result = graph.invoke({
        "messages": [HumanMessage(content=RESPONSE_TEST["content"])],
    }, config=config)
case10_result = cast(AgentState, case10_result)
print_result("case10 - 2턴", case10_result)


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