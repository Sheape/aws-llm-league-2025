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
be related to AI.
Here are some examples to guide you through:
    Topic: Responsible AI | Subtopic: Ethical Principles
    Topic: Responsible AI | Subtopic: Identifying & Mitigating Bias
    Topic: Responsible AI | Subtopic: Transparency Role
    Topic: Responsible AI | Subtopic: AI Accountability
    Topic: Responsible AI | Subtopic: Privacy Protection
    Topic: Responsible AI | Subtopic: Human-Centered AI
    Topic: Responsible AI | Subtopic: AI Governance Frameworks
    Topic: Responsible AI | Subtopic: Inclusivity & Accessibility
    Topic: Responsible AI | Subtopic: Risk & Failure Management
    Topic: Responsible AI | Subtopic: Stakeholder Engagement
    Topic: Agentic AI | Subtopic: Definition & Autonomy
    Topic: Agentic AI | Subtopic: Reward Functions
    Topic: Agentic AI | Subtopic: Goal-Directed Behavior
    Topic: Agentic AI | Subtopic: Ethical Challenges
    Topic: Agentic AI | Subtopic: Emergent Behavior
    Topic: Agentic AI | Subtopic: Multi-Agent Systems
    Topic: Agentic AI | Subtopic: Uncertainty in Decisions
    Topic: Agentic AI | Subtopic: Bounded Rationality
    Topic: Agentic AI | Subtopic: Environmental Learning
    Topic: Agentic AI | Subtopic: Decision-Making Framework
    Topic: Agentic AI | Subtopic: Social Awareness
    Topic: Agentic AI | Subtopic: Memory Role
    Topic: Agentic AI | Subtopic: Scalable Autonomy
    Topic: Prompt Engineering | Subtopic: Definition & Importance
    Topic: Prompt Engineering | Subtopic: Few-Shot Prompting
    Topic: Prompt Engineering | Subtopic: Temperature Setting
    Topic: Prompt Engineering | Subtopic: Chain-of-Thought Prompting
    Topic: Prompt Engineering | Subtopic: Effective Prompt Elements
    Topic: Prompt Engineering | Subtopic: Zero-Shot Prompting
    Topic: Prompt Engineering | Subtopic: Role Prompting
    Topic: Prompt Engineering | Subtopic: Context Length
    Topic: Prompt Engineering | Subtopic: Prompt Templates
    Topic: Prompt Engineering | Subtopic: Testing & Iteration
    Topic: Prompt Engineering | Subtopic: Explicit vs Implicit
    Topic: Prompt Engineering | Subtopic: Prompt Chaining
    Topic: Prompt Engineering | Subtopic: Common Pitfalls
    Topic: Foundational Models | Subtopic: Definition & Comparison
    Topic: Foundational Models | Subtopic: Transfer Learning
    Topic: Foundational Models | Subtopic: Self-Supervision
    Topic: Foundational Models | Subtopic: Data Efficiency
    Topic: Foundational Models | Subtopic: Architecture Components
    Topic: Foundational Models | Subtopic: Emergent Abilities
    Topic: Foundational Models | Subtopic: Deployment Challenges
    Topic: Foundational Models | Subtopic: Multi-modal Inputs
    Topic: Foundational Models | Subtopic: Ethical Considerations
    Topic: Foundational Models | Subtopic: Zero/Few-Shot Learning
    Topic: Foundational Models | Subtopic: Scaling Impact
    Topic: Foundational Models | Subtopic: Contextual Understanding
    Topic: Foundational Models | Subtopic: Open vs Closed Models
    Topic: Foundational Models | Subtopic: Chain-of-Thought Reasoning

These subtopics must be relevant to the main topic and should also be useful for
fine-tuning LLama 3.2 3B Instruct model.
Generate {num_subtopics} subtopics for the following topic:
"""

GENERATE_ANSWER_PROMPT="""You are an expert at generating data for fine-tuning
LLMs such as LLama 3.2 3B Instruct. Your goal is to fine-tune LLama 3.2 3B
Instruct in AWS SageMaker Jumpstart (Instruction-tune enabled and chat dataset
disabled). Given the following topic {topic} and subtopic {subtopic},
answer the following question provided. When answering, ensure that the
answer is concise and accurate. Most importantly, make sure that the answers
are formatted to what LLama 3.2 3B Instruct expects. Do not include emojis and
non-ascii characters. Do not cite your sources.

Here are some examples of Q&A:
    Question: What are the key ethical principles that should guide the development of responsible AI systems?
    Answer: Responsible AI development should be guided by principles including
    transparency, fairness, accountability, privacy protection, human
    oversight, non-maleficence (avoiding harm), beneficence (promoting good),
    and respect for human autonomy. These principles ensure AI systems serve
    humanity's best interests while minimizing potential risks.

    Question: How does multi-agent AI interaction differ from single-agent systems?
    Answer: Multi-agent AI systems involve multiple autonomous agents
    interacting, competing, or cooperating to achieve goals. This creates
    complex dynamics, requiring coordination, negotiation, and strategic
    decision-making, unlike single-agent systems that operate in isolation.

    Question: Explain the concept of 'zero-shot prompting'.
    Answer: Zero-shot prompting is when an AI model performs tasks without
    prior examples or training. It relies on the model's existing knowledge to
    understand and execute instructions, demonstrating its ability to
    generalize across different contexts.

    Question: How does prompt chaining work in complex tasks?
    Answer: Prompt chaining connects multiple prompts sequentially, where each
    prompt's output becomes input for the next. This technique helps break down
    complex tasks into manageable steps, improving accuracy and maintaining
    logical flow.

Here is the question:
"""

RANK_SUBTOPICS_PROMPT="""You are an expert at scoring the subtopic from its topic.
This is the scoring criteria for determining the best subtopics:
    * Relevance:
    How closely related is the subtopic to the main topic ("Foundational Models")?
    Score: 1 (low relevance) to 5 (high relevance).

    * Usefulness for Fine-Tuning:
    How valuable is the subtopic for improving the Llama 3.2 3B Instruct model's performance?
    Score: 1 (low usefulness) to 5 (high usefulness).

    * Complexity/Nuance:
    How much depth and interesting information can be generated by this subtopic?
    Score: 1 (basic) to 5 (complex).

    * Practical Application:
    How likely is this subtopic to be useful in real-world applications?
    Score: 1 (theoretical) to 5 (highly practical).

    * Clarity and Conciseness:
    How well does the subtopic fit the 1-5 word constraint while still being clear?
    Score: 1 (unclear) to 5 (very clear).

The subtopic's score must be at minimum 17, if its not, then you'll have to
generate a new subtopic that has the minimum score of 17.
Given the following topic **{topic}**, calculate the scores of each subtopic and rank them.
Here are the subtopics:
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
Instruct Model. Do NOT compare the answers. JUST OUTPUT THE BEST ANSWER AND COPY THE RESPONSE.
No need to reason out why you chose that answer.

Here are the responses:
"""
