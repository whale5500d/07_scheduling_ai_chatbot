from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

def judge_schedule(state: AgentState):
    """
    가장 최근 사용자 메시지에 물음표가 있으면 일정 질문으로 간주하는
    최소 규칙 버전. 사례집 기반 RAG는 이후 단계에서 추가 예정.
    """

    last_message = state['messages'][-1]
    is_question = "?" in last_message.content

    if is_question:
        return {
            "messages": [AIMessage(content="일정 질문 O")],
            "pending_question": last_message.content
        }

    return {
        "messages": [AIMessage(content="일정 질문 X")],
        "pending_question": None
    }

# 1. declarate state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_question: str | None

graph = StateGraph(AgentState)

# 2. design workflow
graph.add_node(judge_schedule)
graph.add_edge(START, "judge_schedule")
graph.add_edge("judge_schedule", END)

# 3. create graph
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)

# 4. run graph
config: RunnableConfig = {"configurable": {"thread_id": "test-thread-1"}}

turn1 = graph.invoke({
        "messages": [HumanMessage(content="내일 산책 할래?")],
        "pending_question": None
    }, config=config)
print("1턴 결과:", turn1)

turn2 = graph.invoke({
        "messages": [HumanMessage(content="응 좋아")],
        "pending_question": None
    }, config=config)
print("2턴 결과:", turn2)

# 1턴 결과: {'messages': [
#       HumanMessage(
#           content='내일 산책 할래?', 
#           additional_kwargs={},
#           response_metadata={},
#           id='735552f0-2b33-48c3-8c65-d8ff72973fa1'
#       ),
#       AIMessage(
#           content='일정 질문 O',
#           additional_kwargs={},
#           response_metadata={},
#           id='8de75594-4fae-4311-8311-ddec8d7e7c2d',
#           tool_calls=[],
#           invalid_tool_calls=[]
#       )
#   ],
#   'pending_question': '내일 산책 할래?'
# }
# 2턴 결과: {'messages': [
#     HumanMessage(
#         content='내일 산책 할래?',
#         additional_kwargs={},
#         response_metadata={},
#         id='735552f0-2b33-48c3-8c65-d8ff72973fa1'
#     ),
#     AIMessage(
#         content='일정 질문 O',
#         additional_kwargs={},
#         response_metadata={},
#         id='8de75594-4fae-4311-8311-ddec8d7e7c2d',
#         tool_calls=[],
#         invalid_tool_calls=[]
#     ),
#     HumanMessage(
#         content='응 좋아',
#         additional_kwargs={},
#         response_metadata={},
#         id='7fee0f9c-4f21-4860-b364-304b4db311a0'
#     ),
#     AIMessage(
#         content='일정 질문 X',
#         additional_kwargs={},
#         response_metadata={},
#         id='73200465-8d35-4b51-8863-d39600c34816',
#         tool_calls=[],
#         invalid_tool_calls=[]
#     )
#   ],
#   'pending_question': None
# }