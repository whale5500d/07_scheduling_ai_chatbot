from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def mock_llm(state: AgentState):
    return {"messages": [AIMessage(content="hello world")]}

graph = StateGraph(AgentState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

result = graph.invoke({"messages": [HumanMessage(content="hi!")]})
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