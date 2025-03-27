GENERATE_QUESTION_PROMPT="""You are an expert at generating data for fine-tuning LLMs such as LLama 3.2 3B Instruct.
Given the following topic {topic} and subtopic {subtopic},
generate atleast 5 questions from this subtopic.
Here are some examples to guide you through:
    - What is the role of temperature setting in prompt engineering?
    - Explain the importance of prompt testing and iteration.
    - How do Foundational Models address the problem of data efficiency?
    - What are the key architectural components of a typical Foundational Model?
"""

GENERATE_ANSWER_PROMPT="""You are an expert at generating data for fine-tuning
LLMs such as LLama 3.2 3B Instruct. Given the following topic {topic} and subtopic {subtopic},
answer the following questions provided. When answering, ensure that the
answers are concise and accurate. Most importantly, make sure that the answers
are formatted to what LLama 3.2 3B Instruct expects. Do not cite your sources.
PLEASE use the search tool provided to you to do your research.
Here are some example answer to guide you through (dont actually copy the Q: and A:):
    Q: What is the role of temperature setting in prompt engineering?
    A: Temperature controls the randomness of AI responses. Lower values (0-0.3) produce more focused, deterministic outputs, while higher values (0.7-1.0) generate more creative, diverse responses. It's crucial for balancing precision versus creativity in outputs.
    Q: Explain the importance of prompt testing and iteration.
    A: Prompt testing involves systematically evaluating and refining prompts to improve outcomes. Through iteration, engineers can identify weaknesses, optimize performance, and ensure prompts consistently generate desired results across different scenarios.

    Q: How do Foundational Models address the problem of data efficiency?
    A: Foundational Models improve data efficiency by learning general representations from large-scale pre-training, requiring fewer labeled examples for specific tasks. This transfer learning capability makes them particularly valuable when task-specific labeled data is scarce or expensive to obtain.

    Q: What are the key architectural components of a typical Foundational Model?
    A: Typical Foundational Models consist of transformer architectures with attention mechanisms, deep neural networks, embedding layers, and normalization components. These elements work together to process and understand complex patterns in data across multiple modalities and contexts.


Now, answer the following questions from the user.
"""
