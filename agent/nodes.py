from typing import Tuple, TypedDict
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState

import sqlite3
from dotenv import load_dotenv

load_dotenv()
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None
)

class OverallState(MessagesState):
    topics_subtopics: list[Tuple[int,str,str]]
    pass

class SubtopicState(TypedDict):
    subtopic_id: int
    topic: str
    subtopic: str

def extract_subtopics():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM subtopics WHERE done = 0")
    subtopics = cursor.fetchall()
    pairs = [(subtopic[0], subtopic[1], subtopic[2]) for subtopic in subtopics]

    cursor.close()
    conn.commit()
    conn.close()

    return {"topics_subtopics": pairs}
