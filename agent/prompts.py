GENERATE_QUESTION_PROMPT="""You are an expert at generating data for fine-tuning instruct models such as LLama 3.2 3B Instruct.
Given the following topic and subtopic, generate questions from this subtopic.
Remember, your question must be clear, concise, and relevant to the subtopic.

Here are some examples to guide you through:
    Topic: Responsible AI | Subtopic: Ethical Principles | Question: What are the key ethical principles that should guide the development of responsible AI systems?
    Topic: Responsible AI | Subtopic: Identifying & Mitigating Bias | Question: How can bias be identified and mitigated in AI systems?
    Topic: Responsible AI | Subtopic: Transparency Role | Question: What role does transparency play in responsible AI development?
    Topic: Responsible AI | Subtopic: AI Accountability | Question: Explain the concept of AI accountability and why it's important.
    Topic: Responsible AI | Subtopic: Privacy Protection | Question: How can organizations ensure privacy protection when developing AI systems?
    Topic: Responsible AI | Subtopic: Human-Centered AI | Question: What are the key considerations for ensuring AI systems remain human-centered?
    Topic: Responsible AI | Subtopic: AI Governance Frameworks | Question: How can organizations implement responsible AI governance frameworks?
    Topic: Responsible AI | Subtopic: Inclusivity & Accessibility | Question: What measures can be taken to ensure AI systems are inclusive and accessible?
    Topic: Responsible AI | Subtopic: Risk & Failure Management | Question: How should organizations handle AI-related risks and failures?
    Topic: Responsible AI | Subtopic: Stakeholder Engagement | Question: What role does stakeholder engagement play in responsible AI development?
    Topic: Agentic AI | Subtopic: Definition & Autonomy | Question: What is agentic AI and how does it differ from traditional AI systems?
    Topic: Agentic AI | Subtopic: Reward Functions | Question: Explain the role of reward functions in training agentic AI systems.
    Topic: Agentic AI | Subtopic: Goal-Directed Behavior | Question: How does goal-directed behavior manifest in agentic AI systems?
    Topic: Agentic AI | Subtopic: Ethical Challenges | Question: What are the key challenges in ensuring ethical behavior in agentic AI?
    Topic: Agentic AI | Subtopic: Emergent Behavior | Question: Describe the concept of emergent behavior in agentic AI systems.
    Topic: Agentic AI | Subtopic: Multi-Agent Systems | Question: How does multi-agent AI interaction differ from single-agent systems?
    Topic: Agentic AI | Subtopic: Uncertainty in Decisions | Question: What role does uncertainty play in agentic AI decision-making?
    Topic: Agentic AI | Subtopic: Bounded Rationality | Question: Explain the importance of bounded rationality in agentic AI design.
    Topic: Agentic AI | Subtopic: Environmental Learning | Question: How do agentic AI systems learn from their environment?
    Topic: Agentic AI | Subtopic: Decision-Making Framework | Question: What are the key components of an agentic AI's decision-making framework?
    Topic: Agentic AI | Subtopic: Social Awareness | Question: How does social awareness impact agentic AI behavior?
    Topic: Agentic AI | Subtopic: Memory Role | Question: Describe the role of memory in agentic AI systems.
    Topic: Agentic AI | Subtopic: Scalable Autonomy | Question: What are the implications of scalable autonomy in agentic AI?
    Topic: Prompt Engineering | Subtopic: Definition & Importance | Question: What is prompt engineering and why is it important in AI interactions?
    Topic: Prompt Engineering | Subtopic: Few-Shot Prompting | Question: Explain the concept of 'few-shot prompting' in prompt engineering.
    Topic: Prompt Engineering | Subtopic: Temperature Setting | Question: What is the role of temperature setting in prompt engineering?
    Topic: Prompt Engineering | Subtopic: Chain-of-Thought Prompting | Question: How does 'chain-of-thought' prompting work?
    Topic: Prompt Engineering | Subtopic: Effective Prompt Elements | Question: What are the key elements of an effective prompt?
    Topic: Prompt Engineering | Subtopic: Zero-Shot Prompting | Question: Explain the concept of 'zero-shot prompting'.
    Topic: Prompt Engineering | Subtopic: Role Prompting | Question: What is role prompting and how is it used?
    Topic: Prompt Engineering | Subtopic: Context Length | Question: How does context length affect prompt engineering?
    Topic: Prompt Engineering | Subtopic: Prompt Templates | Question: What are prompt templates and why are they useful?
    Topic: Prompt Engineering | Subtopic: Testing & Iteration | Question: Explain the importance of prompt testing and iteration.
    Topic: Prompt Engineering | Subtopic: Explicit vs Implicit | Question: What is the difference between explicit and implicit prompting?
    Topic: Prompt Engineering | Subtopic: Prompt Chaining | Question: How does prompt chaining work in complex tasks?
    Topic: Prompt Engineering | Subtopic: Common Pitfalls | Question: What are common pitfalls in prompt engineering?
    Topic: Foundational Models | Subtopic: Definition & Comparison | Question: What is a Foundational Model in AI, and how does it differ from traditional ML models?
    Topic: Foundational Models | Subtopic: Transfer Learning | Question: Explain the concept of transfer learning in the context of Foundational Models.
    Topic: Foundational Models | Subtopic: Self-Supervision | Question: What role does self-supervision play in training Foundational Models?
    Topic: Foundational Models | Subtopic: Data Efficiency | Question: How do Foundational Models address the problem of data efficiency?
    Topic: Foundational Models | Subtopic: Architecture Components | Question: What are the key architectural components of a typical Foundational Model?
    Topic: Foundational Models | Subtopic: Emergent Abilities | Question: Explain the concept of emergent abilities in Foundational Models.
    Topic: Foundational Models | Subtopic: Deployment Challenges | Question: What are the main challenges in deploying Foundational Models in production environments?
    Topic: Foundational Models | Subtopic: Multi-modal Inputs | Question: How do Foundational Models handle multi-modal inputs?
    Topic: Foundational Models | Subtopic: Ethical Considerations | Question: What are the ethical considerations in developing and deploying Foundational Models?
    Topic: Foundational Models | Subtopic: Zero/Few-Shot Learning | Question: How do Foundational Models handle zero-shot and few-shot learning tasks?
    Topic: Foundational Models | Subtopic: Scaling Impact | Question: What role does scaling play in Foundational Model performance?
    Topic: Foundational Models | Subtopic: Contextual Understanding | Question: How do Foundational Models maintain contextual understanding across different domains?
    Topic: Foundational Models | Subtopic: Open vs Closed Models | Question: What are the key differences between open and closed Foundational Models?
    Topic: Foundational Models | Subtopic: Chain-of-Thought Reasoning | Question: Explain how DeepSeek's chain-of-thought reasoning differs from traditional language models and its advantages in problem-solving tasks.

Now, here is the topic and subtopic and I want you to generate {num_questions} questions:
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

CHOOSE_BEST_QUESTION="""You are an expert at determining the best set of questions for a dataset.
These questions will be used to fine-tune LLama 3.2 3B Instruct (Instruct-tune instead of chat dataset).
Only return 1 if the first set of questions is the best set of questions.
Return 2 if the second set of questions is the best.
Here are some criteria to consider when choosing the best set of questions:
    1. Relevance:
    Topic Alignment: The question must directly relate to the overarching topic.
    Subtopic Specificity: It should focus on the specific nuances and details of the given subtopic.
    Avoid Tangents: Steer clear of questions that stray too far from the core concepts.

    2. Clarity and Conciseness:
    Unambiguous Language: Use clear, straightforward language that leaves no room for misinterpretation.
    Concise Phrasing: Keep questions short and to the point, avoiding unnecessary jargon or verbosity.
    Targeted Focus: Ensure the question has a clear, singular focus.

    3. Depth and Complexity:
    Varied Difficulty: Include a mix of simple, factual questions and more complex, analytical questions.
    Nuanced Understanding: Encourage the model to demonstrate a deeper understanding of the subtopic's intricacies.
    Avoid Triviality: Steer clear of questions that are too basic or easily answered with superficial knowledge.

    4. Diversity and Coverage:
    Comprehensive Scope: Cover a wide range of aspects within the subtopic.
    Different Question Types: Include questions that require different types of responses (e.g., definitions, explanations, comparisons, examples).
    Avoid Repetition: Ensure each question is unique and doesn't overlap significantly with others.

    5. Practicality and Real-World Application:
    Applicable Scenarios: Frame questions that relate to real-world applications or scenarios.
    Problem-Solving Focus: Encourage the model to demonstrate its ability to apply knowledge to solve practical problems.
    Ethical Considerations: If applicable, include questions that address ethical implications or considerations.

    6. Model-Friendly Format:
    Direct Question Structure: Use a clear question format (e.g., "What is...", "How does...", "Explain...").
    Avoid Ambiguous Pronouns: Be specific about the entities or concepts being referred to.
    Consistent Terminology: Use consistent terminology throughout the questions.

    7. Testable and Evaluatable:
    Objective Answers: Aim for questions with answers that can be objectively evaluated.
    Specific Criteria: Ensure there are clear criteria for judging the accuracy and quality of the model's responses.
    Avoid Open-Endedness: While some open-endedness can be useful, prioritize questions that allow for measurable evaluation.

Given the following topic **{topic}** and **{subtopic}**, determine the best set of questions.
Here are the set of questions:
"""

CHECK_QUESTIONS_RELEVANCE_PROMPT="""You are an expert at checking the relevance
and accuracy of a set of questions from the topic and subtopic.
Given the following topic **{topic}** and subtopic **{subtopic}**,
check the relevance and accuracy of the set of questions.
Here are some factors to consider:
    - Are the set of questions helpful in answering the subtopic?
    - Are the set of questions related to the main topic and its subtopic?
    - Are the set of questions helpful in providing a good quality dataset for fine-tuning instruct models (LLama 3.2 3B Instruct)?

All factors must be true before you mark it as relevant. Otherwise, mark it as irrelevant.

Now, evaluate the set of questions if its relevant and accurate:
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
