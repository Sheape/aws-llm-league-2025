from typing import Any, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import AzureChatOpenAI
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.prompts import CHECK_RESPONSE_RELEVANCE_PROMPT, CHOOSE_BEST_RESPONSE_PROMPT, GENERATE_ANSWER_PROMPT


load_dotenv()
fast_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    seed=69
)

creative_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT4O"],
    api_key=convert_to_secret_str(os.environ["AZURE_OPENAI_GPT4O_API_KEY"]),
    temperature=0.6,
    max_tokens=None,
    timeout=None,
)

# _llama_llm = AzureAIChatCompletionsModel(
#     model_name="Llama-3.3-70B-Instruct",
#     temperature=0.2,
#     max_tokens=None,
# )

### States ###
class OverallState(TypedDict):
    question: str
    topic: str
    subtopic: str
    response1: str
    response2: str
    best_response: str
    is_relevant_accurate: bool

### Models ###
class GenResponse(BaseModel):
    """Response/Answer based from the given question"""
    answer: str

class BestResponse(BaseModel):
    """Best Response/Answer based from the given question"""
    best_response: str

class IsRelevantAccurate(BaseModel):
    """Is the response relevant and accurate from the question?"""
    is_relevant_accurate: bool

### Nodes ###
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
