from enum import Enum
from typing import Annotated, Any, Literal, Tuple, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import AzureChatOpenAI
from datetime import datetime
import os
import operator
import json

from dotenv import load_dotenv
from langgraph.graph import END
from pydantic import BaseModel
import sqlite3

from agent.prompts import CHECK_RESPONSE_RELEVANCE_PROMPT, CHOOSE_BEST_RESPONSE_PROMPT, GENERATE_ANSWER_PROMPT, GENERATE_SUBTOPIC_PROMPT, RANK_SUBTOPICS_PROMPT


load_dotenv()
fast_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2025-03-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    seed=69
)

creative_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2025-03-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT4O"],
    api_key=convert_to_secret_str(os.environ["AZURE_OPENAI_GPT4O_API_KEY"]),
    temperature=0.6,
    max_tokens=None,
    timeout=None,
)


### States ###
###### Main Graph ######
class Mode(Enum):
    PROMPT_TESTING_SOME = "prompt_testing_some"
    PROMPT_TESTING_ALL = "prompt_testing_all"
    SUBTOPIC_GENERATION = "subtopic_generation"

class Topic(Enum):
    PROMPT_ENGINEERING = "Prompt Engineering"
    FOUNDATIONAL_MODELS = "Foundational Models"
    AGENTIC_AI = "Agentic AI"
    RESPONSIBLE_AI = "Responsible AI"

class MainInputState(TypedDict):
    mode: Mode
    topic: Topic

class MainOverallState(TypedDict):
    mode: Mode
    topic: Topic
    subtopics: list[dict]
    subtopics_with_ranking: list[Tuple[str, int]]
    subtopic_generation: int
    filename_db: str
    dataset: list[dict]
    best_responses: Annotated[list[dict], operator.add]
    status: str

###### GenResponse Graph ######
class OverallState(TypedDict):
    question: str
    topic: str
    subtopic: str
    subtopic_id: int
    response1: str
    response2: str
    best_response: str
    is_relevant_accurate: bool

### Models ###
###### GenResponse Graph ######
class GenResponse(BaseModel):
    """Response/Answer based from the given question"""
    answer: str

class BestResponse(BaseModel):
    """Best Response/Answer based from the given question"""
    best_response: str

class IsRelevantAccurate(BaseModel):
    """Is the response relevant and accurate from the question?"""
    is_relevant_accurate: bool

###### GenSubtopics Graph ######
class Subtopics(BaseModel):
    """Subtopics based from the given topic. The list should be maximum of 50
    elements. """
    subtopics: list[str]

class SubtopicsRankingSingle(BaseModel):
    """Subtopic and its score"""
    subtopic: str
    score: int

class SubtopicsRanking(BaseModel):
    """Subtopics ordered by rank that is based on their score.
    Returns a list of tuples where the first element is the subtopic and the
    second element is the total score.
    The list should be maximum of 50 elements.
    """
    subtopics: list[SubtopicsRankingSingle]

### Nodes ###
###### GenResponse Graph ######
async def generate_response1(state: OverallState):
    sys_msg = SystemMessage(
        content=GENERATE_ANSWER_PROMPT.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
        )
    )
    human_msg = HumanMessage(content=state["question"])
    query = [sys_msg] + [human_msg]
    response: GenResponse | Any = await creative_llm.with_structured_output(GenResponse).ainvoke(query)
    return {
        "response1": response.answer
    }

async def generate_response2(state: OverallState):
    sys_msg = SystemMessage(
        content=GENERATE_ANSWER_PROMPT.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
        )
    )
    human_msg = HumanMessage(content=state["question"])
    query = [sys_msg] + [human_msg]
    response: GenResponse | Any = await creative_llm.with_structured_output(GenResponse).ainvoke(query)
    return {
        "response2": response.answer
    }

async def choose_best_response(state: OverallState):
    sys_msg = SystemMessage(
        content=CHOOSE_BEST_RESPONSE_PROMPT.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
            question=state["question"]
        )
    )
    human_msg = HumanMessage(content=f"1. f{state['response1']}\n\n2. f{state['response2']}")
    query = [sys_msg] + [human_msg]
    response: BestResponse | Any = await fast_llm.with_structured_output(BestResponse).ainvoke(query)
    return {
        "best_response": response.best_response
    }

async def check_relevance_accuracy(state: OverallState):
    sys_msg = SystemMessage(
        content=CHECK_RESPONSE_RELEVANCE_PROMPT.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
            question=state["question"]
        )
    )
    human_msg = HumanMessage(content=f"Response: f{state['best_response']}")
    query = [sys_msg] + [human_msg]
    response: IsRelevantAccurate | Any = await fast_llm.with_structured_output(IsRelevantAccurate).ainvoke(query)
    return {
        "is_relevant_accurate": response.is_relevant_accurate
    }

###### GenSubtopics Graph ######
async def generate_subtopics(state: MainInputState):
    sys_msg = SystemMessage(
        content=GENERATE_SUBTOPIC_PROMPT.format(
            num_subtopics=50
        )
    )
    human_msg = HumanMessage(content=f"{state['topic']}")
    query = [sys_msg] + [human_msg]
    response: Subtopics | Any = await creative_llm.with_structured_output(Subtopics).ainvoke(query)
    return {
        "topic": state["topic"],
        "subtopics": response.subtopics
    }

async def score_subtopics(state: MainOverallState):
    sys_msg = SystemMessage(
        content=RANK_SUBTOPICS_PROMPT.format(
            topic=state["topic"]
        )
    )
    human_msg = HumanMessage(content="\n".join([f"{i+1}. {subtopic}" for i, subtopic in enumerate(state["subtopics"])]))
    query = [sys_msg] + [human_msg]
    response: SubtopicsRanking | Any = await fast_llm.with_structured_output(SubtopicsRanking).ainvoke(query)
    subtopics_with_ranking = [(subtopic_ranking.subtopic, subtopic_ranking.score) for subtopic_ranking in response.subtopics]
    subtopics_filtered = subtopics_with_ranking[:25]
    if state.get("subtopic_generation", 0) > 0:
        if is_new_subtopic_list_better(state["subtopics_with_ranking"], subtopics_filtered):
            final_subtopics = subtopics_filtered
        else:
            final_subtopics = state["subtopics_with_ranking"]
    else:
        final_subtopics = subtopics_filtered
    return {
        "subtopics_with_ranking": final_subtopics,
        "subtopic_generation": state.get("subtopic_generation", 0) + 1
    }

def is_new_subtopic_list_better(
    prev_subtopics: list[Tuple[str, int]],
    new_subtopics: list[Tuple[str, int]]
) -> bool:
    prev_subtopics_score = 0
    new_subtopics_score = 0
    for _, score in prev_subtopics:
        prev_subtopics_score += score

    for _, score in new_subtopics:
        new_subtopics_score += score

    return new_subtopics_score > prev_subtopics_score

def continue_subtopic_gen(state: MainOverallState) -> Literal["generate_subtopics", "save_subtopic_to_db"]: # type: ignore
    if state["subtopic_generation"] < 5:
        return "generate_subtopics"
    else:
        return "save_subtopic_to_db"

def initialize_db(state: MainOverallState):
    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    if state["mode"] == Mode.PROMPT_TESTING_SOME or state["mode"] == Mode.PROMPT_TESTING_ALL:
        filename = f"{current_date_dash}-dataset-test.db"
    else:
        filename = f"{current_date_dash}-dataset.db"
    conn = sqlite3.connect(f"./db/{filename}")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS "subtopics" (
        "id"	INTEGER NOT NULL UNIQUE,
        "topic"	TEXT NOT NULL,
        "subtopic"	TEXT NOT NULL,
        "questions_generated" INTEGER NOT NULL DEFAULT 0,
        "answers_generated" INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY("id" AUTOINCREMENT)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS "questions_answers" (
        "id"	INTEGER NOT NULL UNIQUE,
        "created_at" TEXT NOT NULL,
        "question"	TEXT NOT NULL,
        "answer"	TEXT,
        "subtopic_id"	INTEGER,
        PRIMARY KEY("id" AUTOINCREMENT),
        FOREIGN KEY (subtopic_id) REFERENCES "subtopics" (id)
    );
    """)

    if state["mode"] == Mode.PROMPT_TESTING_SOME or state["mode"] == Mode.PROMPT_TESTING_ALL:
        conn_base = sqlite3.connect("./db/base_dataset.db")
        cursor_base = conn_base.cursor()
        cursor.execute("SELECT COUNT(*) FROM subtopics;")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor_base.execute("SELECT topic, subtopic FROM subtopics;")
            subtopic_rows = cursor_base.fetchall()
            subtopics = [(row[0], row[1]) for row in subtopic_rows]
            cursor.executemany("""INSERT INTO subtopics (topic, subtopic)
            VALUES (?, ?);""", subtopics)
            conn.commit()
        cursor_base.close()
        conn_base.close()

    cursor.close()
    conn.commit()
    conn.close()
    return { "filename_db": filename }

def save_subtopic_to_db(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    topic_subtopic_pairs = [(state['topic'], subtopic[0]) for subtopic in state['subtopics_with_ranking']]
    cursor.executemany("INSERT INTO subtopics (topic, subtopic) VALUES (?, ?);", topic_subtopic_pairs)
    cursor.close()
    conn.commit()
    conn.close()

###### Main Graph ######
def route_input_mode(state: MainInputState) -> Literal["retrieve_base_dataset", "generate_subtopics", END]: # type: ignore
    if state["mode"] == Mode.PROMPT_TESTING_SOME.value or state["mode"] == Mode.PROMPT_TESTING_ALL.value:
        return "retrieve_base_dataset"
    if state["mode"] == Mode.SUBTOPIC_GENERATION.value:
        return "generate_subtopics"
    else:
        return END

def retrieve_base_dataset(state: MainInputState):
    conn = sqlite3.connect("./db/base_dataset.db")
    cursor = conn.cursor()
    cursor.execute("""SELECT qa.question, qa.answer, s.topic, s.subtopic, s.id FROM questions_answers qa
    JOIN subtopics s ON qa.subtopic_id = s.id;""")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    dataset = [{"question": row[0], "answer": row[1], "topic": row[2], "subtopic": row[3], "subtopic_id": row[4]} for row in rows]
    if state["mode"] == Mode.PROMPT_TESTING_SOME.value:
        return {
            "mode": state["mode"],
            "dataset": dataset[:1]
        }
    elif state["mode"] == Mode.PROMPT_TESTING_ALL.value:
        return {
            "mode": state["mode"],
            "dataset": dataset
        }

def save_response_to_db(state: MainOverallState):
    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    current_time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()

    qa_responses = [(f"{current_date_dash}_{current_time}", qa["question"], qa["response"], qa["subtopic_id"]) for qa in state["best_responses"]]
    cursor.executemany("""INSERT INTO questions_answers (created_at, question, answer, subtopic_id)
    VALUES (?, ?, ?, ?);""", qa_responses)

    cursor.close()
    conn.commit()
    conn.close()

    return { "status": "save_response_to_db success!" }

def save_as_jsonl(state: MainOverallState):
    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    current_time = now.strftime("%H_%M_%S")

    os.makedirs(os.path.dirname(f"./output/{current_date_dash}/"), exist_ok=True)
    with open(f"./output/{current_date_dash}/{current_date_dash}-{current_time}-train.jsonl", "w") as f:
        for qa in state["best_responses"]:
            json_line = json.dumps({
                "instruction": qa["question"],
                "context": "",
                "response": qa["response"],
            })
            f.write(json_line + '\n')
    return { "status": "save_as_jsonl success!" }
