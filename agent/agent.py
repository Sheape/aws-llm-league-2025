from langgraph.graph import END, START, StateGraph

from agent.nodes import OverallState, continue_to_generate, extract_subtopics, generate_question_answer, save_to_db


builder = StateGraph(OverallState)
builder.add_node("extract_subtopics", extract_subtopics)
builder.add_node("generate_question_answer", generate_question_answer)
builder.add_node("save_to_db", save_to_db)

builder.add_edge(START, "extract_subtopics")
builder.add_conditional_edges("extract_subtopics", continue_to_generate,
                              ["generate_question_answer"])
builder.add_edge("generate_question_answer", "save_to_db")
builder.add_edge("save_to_db", END)

graph = builder.compile()
