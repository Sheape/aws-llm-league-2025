from typing import Literal
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.nodes import MainInputState, MainOverallState, OverallState, QuestionOverallState, check_relevance_accuracy, check_relevance_questions, choose_best_questions, choose_best_response, continue_subtopic_gen, generate_questions_1, generate_questions_2, generate_response1, generate_response2, generate_subtopics, initialize_db, output_to_csv, output_to_jsonl, retrieve_base_dataset, retrieve_dataset, retrieve_next_subtopic, retrieve_subtopics, route_gen_answer, route_input_mode, save_answers_to_db, save_as_jsonl, save_questions_to_db, save_response_to_db, save_subtopic_to_db, score_subtopics

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

def dummy_questions_node(state: QuestionOverallState):
    return state

def regenerate_questions(state: QuestionOverallState) -> Literal[END, "dummy_questions_node"]: # type: ignore
    if state["is_relevant"]:
        return END
    else:
        return "dummy_questions_node"

gen_questions_builder = StateGraph(QuestionOverallState)
gen_questions_builder.add_node("generate_questions1", generate_questions_1)
gen_questions_builder.add_node("generate_questions2", generate_questions_2)
gen_questions_builder.add_node("choose_best_questions", choose_best_questions)
gen_questions_builder.add_node("check_relevance_questions", check_relevance_questions)
gen_questions_builder.add_node("dummy_questions_node", dummy_questions_node)

gen_questions_builder.add_edge(START, "dummy_questions_node")
gen_questions_builder.add_edge("dummy_questions_node", "generate_questions1")
gen_questions_builder.add_edge("dummy_questions_node", "generate_questions2")
gen_questions_builder.add_edge("generate_questions1", "choose_best_questions")
gen_questions_builder.add_edge("generate_questions2", "choose_best_questions")
gen_questions_builder.add_edge("choose_best_questions", "check_relevance_questions")
gen_questions_builder.add_conditional_edges("check_relevance_questions", regenerate_questions)

gen_questions = gen_questions_builder.compile()

async def call_gen_response_subgraph(state: OverallState):
    response = await gen_response.ainvoke({
        "topic": state["topic"],
        "subtopic": state["subtopic"],
        "question": state["question"],
    })
    return {
        "best_responses": [{
            "qa_id": state["qa_id"],
            "question": state["question"],
            "response": response["best_response"],
            "subtopic_id": state["subtopic_id"],
        }]
    }

def continue_gen_response(state: MainOverallState):
    return [Send("call_gen_response_subgraph", {
        "qa_id": x["qa_id"],
        "question": x["question"],
        "topic": x["topic"],
        "subtopic": x["subtopic"],
        "subtopic_id": x["subtopic_id"]
    }) for x in state["dataset"]]

async def call_gen_answers_subgraph(state: OverallState):
    response = await gen_response.ainvoke({
        "topic": state["topic"],
        "subtopic": state["subtopic"],
        "question": state["question"],
    })
    return {
        "best_responses": [{
            "qa_id": state["qa_id"],
            "question": state["question"],
            "response": response["best_response"],
            "subtopic_id": state["subtopic_id"],
        }]
    }

def continue_gen_answers(state: MainOverallState):
    return [Send("call_gen_answers_subgraph", {
        "qa_id": x["qa_id"],
        "question": x["question"],
        "topic": x["topic"],
        "subtopic": x["subtopic"],
        "subtopic_id": x["subtopic_id"]
    }) for x in state["dataset"]]

async def call_gen_questions_subgraph(state: MainOverallState):
    current_index = state.get("current_subtopic_index", 0)
    response = await gen_questions.ainvoke({
        "topic": state["topic"],
        "subtopic": state["subtopics"][current_index]["subtopic"],
        "subtopic_id": state["subtopics"][current_index]["id"]
    })
    return {
        "best_question_set": [(state["subtopics"][current_index]["id"], x) for x in response["best_set"]],
        "current_subtopic_index": current_index + 1,
    }

def continue_gen_questions(state: MainOverallState) -> Literal["call_gen_questions_subgraph", END]: # type: ignore
    if state.get("current_subtopic_index", 0) < len(state["subtopics"]):
        return "call_gen_questions_subgraph"
    else:
        return END

main_builder = StateGraph(MainOverallState, input=MainInputState, output=MainOverallState)
main_builder.add_node("retrieve_base_dataset", retrieve_base_dataset)
main_builder.add_node("call_gen_response_subgraph", call_gen_response_subgraph)
main_builder.add_node("call_gen_answers_subgraph", call_gen_answers_subgraph)
main_builder.add_node("call_gen_questions_subgraph", call_gen_questions_subgraph)
main_builder.add_node("initialize_db", initialize_db)
main_builder.add_node("save_subtopic_to_db", save_subtopic_to_db)
main_builder.add_node("save_questions_to_db", save_questions_to_db)
main_builder.add_node("save_response_to_db", save_response_to_db)
main_builder.add_node("save_as_jsonl", save_as_jsonl)
main_builder.add_node("output_to_jsonl", output_to_jsonl)
main_builder.add_node("output_to_csv", output_to_csv)
main_builder.add_node("generate_subtopics", generate_subtopics)
main_builder.add_node("rank_subtopics", score_subtopics)
main_builder.add_node("retrieve_subtopics", retrieve_subtopics)
main_builder.add_node("retrieve_next_subtopic", retrieve_next_subtopic)
main_builder.add_node("retrieve_dataset", retrieve_dataset)
main_builder.add_node("save_answers_to_db", save_answers_to_db)

main_builder.add_edge(START, "initialize_db")
main_builder.add_conditional_edges("initialize_db", route_input_mode)
main_builder.add_conditional_edges("retrieve_base_dataset", continue_gen_response, ["call_gen_response_subgraph"]) # type: ignore
main_builder.add_edge("call_gen_response_subgraph", "save_response_to_db")
main_builder.add_edge("save_response_to_db", "save_as_jsonl")
main_builder.add_edge("save_as_jsonl", END)
main_builder.add_edge("generate_subtopics", "rank_subtopics")
main_builder.add_conditional_edges("rank_subtopics", continue_subtopic_gen)
main_builder.add_edge("save_subtopic_to_db", END)
main_builder.add_edge("retrieve_subtopics", "call_gen_questions_subgraph")
main_builder.add_edge("call_gen_questions_subgraph", "save_questions_to_db")
main_builder.add_conditional_edges("save_questions_to_db", continue_gen_questions)
main_builder.add_edge("retrieve_next_subtopic", "retrieve_dataset")
main_builder.add_conditional_edges("retrieve_dataset", continue_gen_answers, ["call_gen_answers_subgraph"]) # type: ignore
main_builder.add_edge("call_gen_answers_subgraph", "save_answers_to_db")
main_builder.add_conditional_edges("save_answers_to_db", route_gen_answer)
main_builder.add_edge("output_to_jsonl", END)
main_builder.add_edge("output_to_csv", END)

graph = main_builder.compile()
