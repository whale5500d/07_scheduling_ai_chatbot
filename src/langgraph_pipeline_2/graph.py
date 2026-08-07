from typing import cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_pipeline_2.state import AgentState, ResponseVerdict
from langgraph_pipeline_2.utils.nodes import judge_schedule, judge_response, judge_date, save_rdb, call_model, confirm_save
from langgraph_pipeline_2.utils.tools import get_tools

tools = get_tools()

def route_from_start(state: AgentState):
    return "judge_response" if state.get("pending_question") else "judge_schedule"

def route_after_schedule(state: AgentState):
    return "judge_response" if state.get("pending_question") else END

def route_after_response(state: AgentState):
    if state.get("response_verdict") == ResponseVerdict.POSITIVE.value:
        return "judge_date"
    elif state.get("response_verdict") == ResponseVerdict.UNCLEAR.value:
        return "call_model"
    else:
        return END

def route_after_call_model(state: AgentState):
    casted_state = cast(dict, state)
    if tools_condition(casted_state) == "tools":
        return "tools"
    elif state.get("response_verdict") == ResponseVerdict.POSITIVE.value:
        return "judge_date"
    else:
        return END

def route_after_confirm(state: AgentState):
    if state.get("is_confirmed"):
        return "save_rdb"
    else:
        return END

# 1. create state graph object
graph = StateGraph(AgentState)

# 2. design workflow
graph.add_node(judge_schedule)
graph.add_node(judge_response)
graph.add_node(judge_date)
graph.add_node(save_rdb)
graph.add_conditional_edges(START, route_from_start, {"judge_schedule": "judge_schedule", "judge_response": "judge_response"},)
graph.add_conditional_edges("judge_schedule", route_after_schedule, {"judge_response": "judge_response", END: END})
graph.add_conditional_edges("judge_response", route_after_response, {"judge_date": "judge_date", "call_model": "call_model", END: END})
graph.add_node("call_model", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("call_model", route_after_call_model, {"tools": "tools", "judge_date": "judge_date", END: END})
graph.add_edge("tools", "call_model")
graph.add_node("confirm_save", confirm_save)
graph.add_edge("judge_date", "confirm_save")
graph.add_conditional_edges("confirm_save", route_after_confirm, {"save_rdb": "save_rdb", END: END})
graph.add_edge("save_rdb", END)

# 3. create graph
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)