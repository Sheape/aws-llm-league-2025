from enum import Enum
from typing import Annotated, Any, Literal, Tuple, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import AzureChatOpenAI
from datetime import datetime
import os
import operator
import json
import csv

from dotenv import load_dotenv
from langgraph.graph import END
from pydantic import BaseModel
import sqlite3

from agent.prompts import CHECK_QUESTIONS_RELEVANCE_PROMPT, CHECK_RESPONSE_RELEVANCE_PROMPT, CHOOSE_BEST_QUESTION, CHOOSE_BEST_RESPONSE_PROMPT, GENERATE_ANSWER_PROMPT, GENERATE_QUESTION_PROMPT, GENERATE_SUBTOPIC_NEW_PROMPT, GENERATE_SUBTOPIC_PROMPT, RANK_SUBTOPICS_PROMPT


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
    SUBTOPIC_NEW_GENERATION = "subtopic_new_generation"
    QUESTION_GENERATION = "question_generation"
    RESPONSE_GENERATION = "response_generation"
    RESPONSE_GENERATION_SOME = "response_generation_some"
    OUTPUT_JSONL = "output_jsonl"
    OUTPUT_CSV = "output_csv"

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
    current_subtopic_id: int
    current_subtopic_index: int
    subtopics_with_ranking: list[Tuple[str, int]]
    subtopic_generation: int
    filename_db: str
    dataset: list[dict]
    best_responses: Annotated[list[dict], operator.add]
    best_question_set: list[Tuple[int, str]]
    status: str

###### GenResponse Graph ######
class OverallState(TypedDict):
    qa_id: int
    question: str
    topic: str
    subtopic: str
    subtopic_id: int
    response1: str
    response2: str
    best_response: str
    is_relevant_accurate: bool

###### GenQuestion Graph ######
class QuestionOverallState(TypedDict):
    questions1: list[str]
    questions2: list[str]
    subtopic_id: int
    best_set: list[str]
    is_relevant: bool
    subtopic: str
    topic: str

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

###### GenQuestion Graph ######
class QuestionsGenerated(BaseModel):
    """Questions generated based from the given topic and subtopic
    The list should be maximum of 50 elements."""
    questions: list[str]

class BestQuestionSet(BaseModel):
    """Determines the best set of question.
    Can either be 1 or 2 which represents which # of set they are."""
    best_set: int

class QuestionSetRelevance(BaseModel):
    """Determines if the set of questions is relevant to the topic and subtopic."""
    is_relevant: bool

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
    human_msg = HumanMessage(content=f"1. {state['response1']}\n\n2. {state['response2']}")
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
    human_msg = HumanMessage(content=f"Response: {state['best_response']}")
    query = [sys_msg] + [human_msg]
    response: IsRelevantAccurate | Any = await fast_llm.with_structured_output(IsRelevantAccurate).ainvoke(query)
    return {
        "is_relevant_accurate": response.is_relevant_accurate
    }

###### GenSubtopics Graph ######
async def generate_subtopics(state: MainOverallState):
    if state["mode"] == Mode.SUBTOPIC_GENERATION.value:
        sys_msg = SystemMessage(
            content=GENERATE_SUBTOPIC_PROMPT.format(
                num_subtopics=50
            )
        )
    else:
        conn = sqlite3.connect(f"./db/{state['filename_db']}")
        cursor = conn.cursor()
        cursor.execute("SELECT subtopic FROM subtopics WHERE topic = ?;", (state["topic"],))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        subtopics = [row[0] for row in rows]
        subtopics_str = "\n".join([f"- {subtopic}" for subtopic in subtopics])
        sys_msg = SystemMessage(
            content=GENERATE_SUBTOPIC_NEW_PROMPT.format(
                num_subtopics=50,
                subtopics=subtopics_str
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
    if state["mode"] == Mode.PROMPT_TESTING_SOME.value or state["mode"] == Mode.PROMPT_TESTING_ALL.value:
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

    if state["mode"] == Mode.PROMPT_TESTING_SOME.value or state["mode"] == Mode.PROMPT_TESTING_ALL.value:
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

###### GenQuestions Graph ######
def retrieve_subtopics(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    cursor.execute(f"""SELECT id, subtopic FROM subtopics
    WHERE topic = '{state['topic']}' AND questions_generated = 0;""")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    subtopics = [{"id": row[0], "subtopic": row[1]} for row in rows]
    return {
        "subtopics": subtopics,
        "current_subtopic_index": 0
    }

async def generate_questions_1(state: QuestionOverallState):
    sys_msg = SystemMessage(
        content=GENERATE_QUESTION_PROMPT.format(
            num_questions=50
        )
    )
    human_msg = HumanMessage(content=f"Topic: {state['topic']}, Subtopic: {state['subtopic']}")
    query = [sys_msg] + [human_msg]
    response: QuestionsGenerated | Any = await creative_llm.with_structured_output(QuestionsGenerated).ainvoke(query)
    return {
        "questions1": response.questions
    }

async def generate_questions_2(state: QuestionOverallState):
    sys_msg = SystemMessage(
        content=GENERATE_QUESTION_PROMPT.format(
            num_questions=50
        )
    )
    human_msg = HumanMessage(content=f"Topic: {state['topic']}, Subtopic: {state['subtopic']}")
    query = [sys_msg] + [human_msg]
    response: QuestionsGenerated | Any = await creative_llm.with_structured_output(QuestionsGenerated).ainvoke(query)
    return {
        "questions2": response.questions
    }

def convert_list_to_str_formatted(questions: list[str]) -> str:
    return "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])

async def choose_best_questions(state: QuestionOverallState):
    sys_msg = SystemMessage(
        content=CHOOSE_BEST_QUESTION.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
        )
    )
    human_msg = HumanMessage(content=f"Set 1:\n{convert_list_to_str_formatted(state['questions1'])}\n\nSet 2: {convert_list_to_str_formatted(state['questions2'])}")
    query = [sys_msg] + [human_msg]
    response: BestQuestionSet | Any = await fast_llm.with_structured_output(BestQuestionSet).ainvoke(query)
    best_set = state['questions1'] if response.best_set == 1 else state['questions2']
    return {
        "best_set": best_set
    }

async def check_relevance_questions(state: QuestionOverallState):
    sys_msg = SystemMessage(
        content=CHECK_QUESTIONS_RELEVANCE_PROMPT.format(
            topic=state["topic"],
            subtopic=state["subtopic"],
        )
    )
    human_msg = HumanMessage(content=f"{convert_list_to_str_formatted(state['best_set'])}")
    query = [sys_msg] + [human_msg]
    response: QuestionSetRelevance | Any = await fast_llm.with_structured_output(QuestionSetRelevance).ainvoke(query)
    return {
        "is_relevant": response.is_relevant
    }

def save_questions_to_db(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    current_time = now.strftime("%H:%M:%S")

    data = [(f"{current_date_dash}_{current_time}", subtopic_id, question) for subtopic_id, question in state["best_question_set"]]
    subtopic_id = data[0][1]
    cursor.executemany("""INSERT INTO questions_answers (created_at, subtopic_id,
                       question) VALUES (?, ?, ?);""", data)
    cursor.execute(f"UPDATE subtopics SET questions_generated = 1 WHERE id = {subtopic_id};")
    cursor.close()
    conn.commit()
    conn.close()

###### Main Graph ######
def route_input_mode(
    state: MainInputState
) -> Literal["retrieve_base_dataset", "generate_subtopics",
             "retrieve_subtopics", "retrieve_next_subtopic", "output_to_jsonl",
             "output_to_csv", END]: # type: ignore
    if state["mode"] == Mode.PROMPT_TESTING_SOME.value or state["mode"] == Mode.PROMPT_TESTING_ALL.value:
        return "retrieve_base_dataset"
    if state["mode"] == Mode.SUBTOPIC_GENERATION.value:
        return "generate_subtopics"
    if state["mode"] == Mode.SUBTOPIC_NEW_GENERATION.value:
        return "generate_subtopics"
    if state["mode"] == Mode.QUESTION_GENERATION.value:
        return "retrieve_subtopics"
    if state["mode"] == Mode.RESPONSE_GENERATION.value:
        return "retrieve_next_subtopic"
    if state["mode"] == Mode.RESPONSE_GENERATION_SOME.value:
        return "retrieve_next_subtopic"
    if state["mode"] == Mode.OUTPUT_JSONL.value:
        return "output_to_jsonl"
    if state["mode"] == Mode.OUTPUT_CSV.value:
        return "output_to_csv"
    else:
        return END

def route_gen_answer(state: MainOverallState) -> Literal[END, "retrieve_next_subtopic"]: # type: ignore
    if state["current_subtopic_id"] == -1:
        return END
    else:
        return "retrieve_next_subtopic"

def output_to_jsonl(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM questions_answers WHERE answer IS NOT NULL")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    current_time = now.strftime("%H_%M_%S")

    os.makedirs(os.path.dirname(f"./output/{current_date_dash}/"), exist_ok=True)
    with open(f"./output/{current_date_dash}/{current_date_dash}-{current_time}-train.jsonl", "w") as f:
        for question, answer in rows:
            json_line = json.dumps({
                "instruction": question,
                "context": "",
                "response": answer,
            })
            f.write(json_line + '\n')

    return { "status": "output_to_jsonl success!" }

def output_to_csv(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM questions_answers WHERE answer IS NOT NULL")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    header = ["Question", "Answer"]
    qa_pairs = [[row[0], row[1]] for row in rows]
    qa_pairs.insert(0, header)

    now = datetime.now()
    current_date_dash = now.strftime("%m-%d-%Y")
    current_time = now.strftime("%H_%M_%S")

    filename = f"./output/{current_date_dash}/{current_date_dash}-{current_time}-train.csv"
    os.makedirs(os.path.dirname(f"./output/{current_date_dash}/"), exist_ok=True)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(qa_pairs)

    return { "status": "output_to_csv success!" }

def retrieve_next_subtopic(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM subtopics WHERE topic = '{state['topic']}' AND answers_generated = 0;")
    row = cursor.fetchone()
    if row:
        subtopic_id = row[0]
    else:
        subtopic_id = -1
    return { "current_subtopic_id": subtopic_id }

def retrieve_dataset(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()
    if state["mode"] == Mode.RESPONSE_GENERATION_SOME.value:
        cursor.execute("""SELECT qa.id, qa.question, s.subtopic, s.id
            FROM questions_answers qa
            JOIN subtopics s ON qa.subtopic_id = s.id
            WHERE s.id = ? AND qa.answer IS NULL
            LIMIT 5;
            """, (state["current_subtopic_id"],))
    else:
        cursor.execute("""SELECT qa.id, qa.question, s.subtopic, s.id
            FROM questions_answers qa
            JOIN subtopics s ON qa.subtopic_id = s.id
            WHERE s.id = ? AND qa.answer IS NULL;
            """, (state["current_subtopic_id"],))
    rows = cursor.fetchall()
    dataset = [{"qa_id": row[0], "question": row[1], "topic": state["topic"],
                "subtopic": row[2], "subtopic_id": row[3]} for row in rows]
    return {
        "dataset": dataset
    }

def retrieve_base_dataset(state: MainOverallState):
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

def save_answers_to_db(state: MainOverallState):
    conn = sqlite3.connect(f"./db/{state['filename_db']}")
    cursor = conn.cursor()

    qa_responses = [(qa["response"], qa["qa_id"]) for qa in state["best_responses"]]
    cursor.executemany("""UPDATE questions_answers SET answer = ?
    WHERE id = ?;""", qa_responses)
    cursor.execute(f"UPDATE subtopics SET answers_generated = 1 WHERE id = {state['current_subtopic_id']}")

    cursor.close()
    conn.commit()
    conn.close()

    return {
        "best_responses": [],
        "status": "save_answers_to_db success!"
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
