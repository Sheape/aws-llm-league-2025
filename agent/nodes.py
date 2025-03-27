import operator
from typing import Annotated, Any, Tuple, TypedDict
from langchain_core.utils import convert_to_secret_str
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState
from langgraph.types import Send
from langgraph.prebuilt import create_react_agent
from datetime import datetime

import sqlite3
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from .tools import TOOLS
from .prompts import GENERATE_ANSWER_PROMPT, GENERATE_QUESTION_PROMPT

load_dotenv()
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    seed=69
)

_reasoning_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_REASONING"),
    api_key=convert_to_secret_str(os.environ["AZURE_OPENAI_API_KEY_REASONING"]),
    azure_deployment="o3-mini",
    api_version="2024-12-01-preview",
    max_tokens=None,
    timeout=None,
    seed=69
)


class OverallState(MessagesState):
    topics_subtopics: list[Tuple[int,str,str]]
    questions_answers: Annotated[list, operator.add]
    records: list[Tuple[str, str, str, int]]

class SubtopicState(TypedDict):
    subtopic_id: int
    topic: str
    subtopic: str

class Question(BaseModel):
    """Questions that are relevant to the subtopic."""
    questions: list[str]

class Answer(BaseModel):
    """Answers to the questions."""
    answers: list[str]

reasoning_llm = create_react_agent(model=_reasoning_llm, tools=TOOLS, response_format=Answer)

def extract_subtopics(state: OverallState):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM subtopics WHERE done = 0;")
    subtopics = cursor.fetchall()
    pairs = [(subtopic[0], subtopic[1], subtopic[2]) for subtopic in subtopics]

    cursor.close()
    conn.commit()
    conn.close()

    return {"topics_subtopics": pairs}

async def generate_question_answer(state: SubtopicState):
    prompt = GENERATE_QUESTION_PROMPT.format(topic=state["topic"], subtopic=state["subtopic"])
    question_response: Question | Any = await llm.with_structured_output(Question).ainvoke(prompt)
    answer_prompt = GENERATE_ANSWER_PROMPT.format(topic=state["topic"], subtopic=state["subtopic"])
    print(question_response.questions)
    answer_response = await reasoning_llm.ainvoke({
        "messages": [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": "\n".join(question_response.questions)}
        ]
    })
    print("Answers: ")
    print(answer_response)
    return {"questions_answers": [{
        "subtopic_id": state["subtopic_id"],
        "questions": question_response.questions,
        "answers": answer_response["structured_response"].answers,
    }]}

async def continue_to_generate(state: OverallState):
    return [Send("generate_question_answer", {"subtopic_id": subtopic_id, "topic": topic, "subtopic": subtopic}) for subtopic_id, topic, subtopic in state["topics_subtopics"]]

def save_to_db(state: OverallState):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    today = datetime.now().strftime("%m-%d-%Y")
    pairs = [(question, answer, today, qa["subtopic_id"]) for qa in
             state["questions_answers"] for question,answer in
             zip(qa["questions"], qa["answers"])]
    ids = [(qa["subtopic_id"],) for qa in state["questions_answers"]]

    print("Saving to database...")
    cursor.executemany("""INSERT INTO question_answers (questions, answers, created_at, subtopic_id)
    VALUES (?,?,?,?);""", pairs)

    cursor.executemany("""UPDATE subtopics SET done = 1
    WHERE id = ?;""", ids)

    cursor.close()
    conn.commit()
    conn.close()
    return {"records": pairs}
