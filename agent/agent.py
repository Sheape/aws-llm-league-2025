from typing import Literal
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.nodes import MainInputState, MainOverallState, OverallState, check_relevance_accuracy, choose_best_response, continue_subtopic_gen, generate_response1, generate_response2, generate_subtopics, initialize_db, retrieve_base_dataset, route_input_mode, save_as_jsonl, save_response_to_db, save_subtopic_to_db, score_subtopics

def dummy_node(state: OverallState):
    return state

def regenerate_response(state: OverallState) -> Literal[END, "dummy_node"]: # type: ignore
    if state["is_relevant_accurate"]:
        return END
    else:
        return "dummy_node"


gen_response_builder = StateGraph(OverallState)
gen_response_builder.add_node("generate_response1", generate_response1)
gen_response_builder.add_node("generate_response2", generate_response2)
gen_response_builder.add_node("choose_best_response", choose_best_response)
gen_response_builder.add_node("check_relevance_accuracy", check_relevance_accuracy)
gen_response_builder.add_node("dummy_node", dummy_node)

gen_response_builder.add_edge(START, "dummy_node")
gen_response_builder.add_edge("dummy_node", "generate_response1")
gen_response_builder.add_edge("dummy_node", "generate_response2")
gen_response_builder.add_edge("generate_response1", "choose_best_response")
gen_response_builder.add_edge("generate_response2", "choose_best_response")
gen_response_builder.add_edge("choose_best_response", "check_relevance_accuracy")
gen_response_builder.add_conditional_edges("check_relevance_accuracy", regenerate_response)

gen_response = gen_response_builder.compile()

async def call_gen_response_subgraph(state: OverallState):
    response = await gen_response.ainvoke({
        "topic": state["topic"],
        "subtopic": state["subtopic"],
        "question": state["question"],
    })
    return {
        "best_responses": [{
            "topic": state["topic"],
            "subtopic": state["subtopic"],
            "question": state["question"],
            "response": response["best_response"],
            "subtopic_id": state["subtopic_id"],
        }]
    }

def continue_gen_response(state: MainOverallState):
    return [Send("call_gen_response_subgraph", {
        "question": x["question"],
        "topic": x["topic"],
        "subtopic": x["subtopic"],
        "subtopic_id": x["subtopic_id"]
    }) for x in state["dataset"]]

main_builder = StateGraph(MainOverallState, input=MainInputState, output=MainOverallState)
main_builder.add_node("retrieve_base_dataset", retrieve_base_dataset)
main_builder.add_node("call_gen_response_subgraph", call_gen_response_subgraph)
main_builder.add_node("initialize_db", initialize_db)
main_builder.add_node("save_subtopic_to_db", save_subtopic_to_db)
main_builder.add_node("save_response_to_db", save_response_to_db)
main_builder.add_node("save_as_jsonl", save_as_jsonl)
main_builder.add_node("generate_subtopics", generate_subtopics)
main_builder.add_node("rank_subtopics", score_subtopics)

main_builder.add_edge(START, "initialize_db")
main_builder.add_conditional_edges("initialize_db", route_input_mode)
main_builder.add_conditional_edges("retrieve_base_dataset", continue_gen_response, ["call_gen_response_subgraph"]) # type: ignore
main_builder.add_edge("call_gen_response_subgraph", "save_response_to_db")
main_builder.add_edge("save_response_to_db", "save_as_jsonl")
main_builder.add_edge("save_as_jsonl", END)
main_builder.add_edge("generate_subtopics", "rank_subtopics")
main_builder.add_conditional_edges("rank_subtopics", continue_subtopic_gen)
main_builder.add_edge("save_subtopic_to_db", END)

graph = main_builder.compile()
