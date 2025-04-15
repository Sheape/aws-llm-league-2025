# Dataset Generator for AWS LLM League 2025
The goal of the competition was to fine-tune LLama 3.2 3B Instruct model with your own custom dataset. 
The theme of the competition includes: Agentic AI, Responsible AI, Prompt Engineering, and Foundational Models.
Creating manually crafted dataset is tedious and almost impossible given the time constraint (3 weeks). Thus, we resort to AI-generated dataset.
However, the challenge arise with the following factors that can affect the dataset generation:
- Quality of the Responses
- Quality of the Questions
- Preventing Hallucination at all costs
- Format and Accuracy of the Responses

The solution I came up with was instead of utilizing [AWS PartyRock](https://partyrock.aws/), I crafted my own agentic AI to help me generate
question-answer dataset for LLama 3.2 3B Instruct. I used [Langgraph](https://www.langchain.com/langgraph) made by Langchain in order to 
execute operations. These operations include generating subtopics, high quality questions and answers, exporting to csv (for further data analysis) 
and jsonl (training dataset file format).

Initially, I tried proprietary options like [GretelAI](https://gretel.ai/) in order to generate high quality synthetic dataset. However, its not suited
for generating synthetic dataset for fine-tuning LLMs.

The highest win rate I've gotten was only 45%. With some optimized prompt engineering, you can probably go as far as 70%.
Feel free to experiment with the prompts or modifying the nodes.

## Tech Stack
- Langgraph
- SQLite3 (for saving subtopics, questions, and answers)
- GPT 4o (for generating questions and answers) and GPT 4o-mini (for decision nodes)

## Setup
In order to setup the agent, you need to follow this steps:
1. Install [Python 3.12](https://www.python.org/downloads/release/python-3120/)
2. Install [uv](https://docs.astral.sh/uv/) as the package manager
3. Setup environment variables in `.env`.
4. Create a virtual environment using `uv venv`.
5. Activate venv using `source .venv/bin/activate`.
6. Sync and install the required packages using `uv sync`.
7. Run langgraph through `langgraph dev`.

## Do you have a question?
If you have any questions regarding the competition or this project, feel free to reach out by opening an issue or sending an email.
