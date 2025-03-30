GENERATE_QUESTION_PROMPT="""You are an expert at generating data for fine-tuning instruct models such as LLama 3.2 3B Instruct.
Given the following topic {topic} and subtopic {subtopic},
generate atleast 5 questions from this subtopic.
Here are some examples to guide you through:
    - What is the role of temperature setting in prompt engineering?
    - Explain the importance of prompt testing and iteration.
    - How do Foundational Models address the problem of data efficiency?
    - What are the key architectural components of a typical foundational Model?
"""

GENERATE_SUBTOPIC_PROMPT="""You are an expert at generating subtopics of a given
topic. These subtopics should be as simple as one to 5 words and they should all
be related to AI. Examples of subtopics include "topic: prompt engineering,
subtopic: zero shot prompting", "topic: prompt engineering, subtopic:
chain-of-thought prompting".

Generate {num_subtopics} subtopics for the following topic: {topic}.
These subtopics must be relevant to the main topic and should also be useful for
fine-tuning LLama 3.2 3B Instruct model.
"""

CHECK_SUBTOPIC_RELEVANCE_PROMPT="""You are an expert at checking the relevance of subtopics to the main topic.
Given the following subtopic **{subtopic}** and topic **{topic}**,
check the relevance of the subtopic to the main topic. Here are the following factors to consider:
    - Is the subtopic helpful in understanding the main topic?
    - Is the subtopic related to the main topic?
    - Is the subtopic helpful in providing a good quality dataset for fine-tuning instruct models (LLama 3.2 3B Instruct)?

All factors must be true before you mark it as relevant. Otherwise, mark it as irrelevant.
"""

CHECK_RESPONSE_RELEVANCE_PROMPT="""You are an expert at checking the relevance
and accuracy of an answer from the question and its topic.
Given the following topic **{topic}** and subtopic **{subtopic}**,
check the relevance and accuracy of the answer from the question: {question}.
Here are some factors to consider:
    - Is the answer helpful in answering the question?
    - Is the answer related to the main topic and its subtopic?
    - Is the answer helpful in providing a good quality dataset for fine-tuning instruct models (LLama 3.2 3B Instruct)?

All factors must be true before you mark it as relevant. Otherwise, mark it as irrelevant.

Now, evaluate this response if its relevant and accurate:
"""

CHOOSE_BEST_RESPONSE_PROMPT="""You are an expert at choosing the best answer based on the question provided.
Given the following topic **{topic}**, subtopic **{subtopic}**, and the
question {question}, choose the best amongst the answers. Here are some factors to consider:
    Helpfulness: Overall helpfulness of the response to the prompt.
    Correctness: Inclusion of all pertinent facts without errors.
    Coherence: Consistency and clarity of expression.
    Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
    Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.
Choose the best answer that will benefit in instruct-tuning LLama 3.2 3B
Instruct Model.

Here are the responses:
"""

GENERATE_ANSWER_PROMPT="""You are an expert at generating data for fine-tuning
LLMs such as LLama 3.2 3B Instruct. Your goal is to fine-tune LLama 3.2 3B
Instruct in AWS SageMaker Jumpstart (Instruction-tune enabled and chat dataset
disabled). Given the following topic {topic} and subtopic {subtopic},
answer the following question provided. When answering, ensure that the
answer is comprehensive and accurate. Most importantly, make sure that the answers
are formatted to what LLama 3.2 3B Instruct expects. Do not include emojis and
non-ascii characters. Do not cite your sources.

Here is the question:
"""
