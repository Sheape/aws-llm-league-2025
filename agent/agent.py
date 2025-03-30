from typing import Literal
from langgraph.graph import END, START, StateGraph

from agent.nodes import OverallState, check_relevance_accuracy, choose_best_response, generate_response1, generate_response2

def dummy_node(state: OverallState):
    return state

def regenerate_response(state: OverallState) -> Literal[END, "dummy_node"]:
    if state["is_relevant_accurate"]:
        return END
    else:
        return "dummy_node"


builder = StateGraph(OverallState)
builder.add_node("generate_response1", generate_response1)
builder.add_node("generate_response2", generate_response2)
builder.add_node("choose_best_response", choose_best_response)
builder.add_node("check_relevance_accuracy", check_relevance_accuracy)
builder.add_node("dummy_node", dummy_node)

builder.add_edge(START, "dummy_node")
builder.add_edge("dummy_node", "generate_response1")
builder.add_edge("dummy_node", "generate_response2")
builder.add_edge("generate_response1", "choose_best_response")
builder.add_edge("generate_response2", "choose_best_response")
builder.add_edge("choose_best_response", "check_relevance_accuracy")
builder.add_conditional_edges("check_relevance_accuracy", regenerate_response)

graph = builder.compile()
