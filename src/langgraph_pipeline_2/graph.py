from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph_pipeline_2.state import AgentState, ResponseVerdict
from langgraph_pipeline_2.utils.nodes import judge_schedule, judge_response, judge_date, save_rdb

def route_from_start(state: AgentState):
    return "judge_response" if state.get("pending_question") else "judge_schedule"

def route_after_schedule(state: AgentState):
    return "judge_response" if state.get("pending_question") else END

def route_after_response(state: AgentState):
    return "judge_date" if state.get("response_verdict") == ResponseVerdict.POSITIVE.value else END

# 1. create state graph object
graph = StateGraph(AgentState)

# 2. design workflow
graph.add_node(judge_schedule)
graph.add_node(judge_response)
graph.add_node(judge_date)
graph.add_node(save_rdb)
graph.add_conditional_edges(START, route_from_start, {"judge_schedule": "judge_schedule", "judge_response": "judge_response"},)
graph.add_conditional_edges("judge_schedule", route_after_schedule, {"judge_response": "judge_response", END: END})
graph.add_conditional_edges("judge_response", route_after_response, {"judge_date": "judge_date", END: END})
graph.add_edge("judge_date", "save_rdb")
graph.add_edge("save_rdb", END)

# 3. create graph
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)