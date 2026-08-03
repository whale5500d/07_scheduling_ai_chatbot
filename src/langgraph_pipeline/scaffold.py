from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def judge_schedule(state: AgentState):
    """
    가장 최근 사용자 메시지에 물음표가 있으면 일정 질문으로 간주하는
    최소 규칙 버전. 사례집 기반 RAG는 이후 단계에서 추가 예정.
    """

    last_message = state['messages'][-1]
    is_question = "?" in last_message.content

    answer = "일정 질문으로 판단됩니다." if is_question else "일정 질문이 아닙니다."
    return {"messages": [AIMessage(content=answer)]}

graph = StateGraph(AgentState)
graph.add_node(judge_schedule)
graph.add_edge(START, "judge_schedule")
graph.add_edge("judge_schedule", END)
graph = graph.compile()

result = graph.invoke({"messages": [HumanMessage(content="내일 산책 할래?")]})
print(result)

# {'messages': [
#         HumanMessage(
#                 content='hi!',
#                 additional_kwargs={},
#                 response_metadata={},
#                 id='58178f5d-bdf8-4b2f-96dd-c45c1bd55ada'
#             ),
#         AIMessage(
#                 content='hello world',
#                 additional_kwargs={},
#                 response_metadata={},
#                 id='547e6850-8f6c-42aa-a5bc-c98f70057919',
#                 tool_calls=[],
#                 invalid_tool_calls=[]
#             )
# ]}