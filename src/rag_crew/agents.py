from crewai import Agent, LLM

from src.rag_crew.tools.retrieval_tool import document_retrieval_tool
import os

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = LLM(
    model="ollama/gemma3:4b",  # 👈 no need for ollama/ prefix now
    base_url=ollama_base_url,
    temperature=0.1,
    timeout=1000,
    verbose=True,
    max_tokens=4096,  # realistic for gemma3:1b
    num_ctx=4096,
)

document_researcher = Agent(
    llm=llm,
    max_iter=3,
    max_execution_time=1000,
    role="Document Researcher",
    goal="Retrieve relevant context from the RAG system for a given question.",
    backstory=(
        "You are a document retrieval agent. Your responsibilities are strictly limited to:"
        "1) Parse and understand the user’s query."
        "2) Use the Document Retrieval Tool to fetch the most relevant text chunks."
        "3) Return only the raw retrieved context, exactly as it appears."

        "DO NOT generate answers from your own knowledge."
        "DO NOT summarize, explain, or interpret the text."
        "ONLY output the retrieved text chunks for the next agent to process."

    ),
    tools=[document_retrieval_tool],
    verbose=True,
    allow_delegation=False,
)

# Synthesis agent - NO tools, only reads context from researcher
insight_synthesizer = Agent(
    llm=llm,
    max_iter=2,
    max_execution_time=1000,
    role="Policy Insight Synthesizer",
    goal="Produce a concise, faithful answer using retrieved context and conversation history.",
    backstory=(
        "You are an expert at answering questions from documents. "
        "You receive context from the researcher containing:\n\n"

        "1. **Retrieved Context**: Fresh document passages relevant to the current question.\n"
        "2. **Conversation History**: Previous questions and answers from this session.\n\n"

        "⚠️ IMPORTANT: You do NOT have access to retrieval tools. "
        "You ONLY analyze the context provided by the researcher. "
        "Never try to retrieve documents yourself.\n\n"

        "How to use context:\n"
        "- For NEW questions: Use 'Retrieved Context' section\n"
        "- For FOLLOW-UPS ('summarize', 'tell me more', 'what about', 'and X?'): "
        "Use 'Conversation History' section\n"
        "- For 'summarize your last response': Look at the most recent assistant message in history\n\n"

        "Response rules:\n"
        "1. Stay completely faithful to the source text — never guess\n"
        "2. Include article/clause numbers when citing documents\n"
        "3. For tables, interpret in plain language (no markdown)\n"
        "4. Be concise: 1-3 sentences maximum\n"
        "5. If info is missing, explicitly state it\n"
        "6. Use natural, official-sounding language\n\n"

        "Example responses:\n"
        "- New question: 'According to Article 110(3), employees receive 30 days annual leave.'\n"
        "- Follow-up: 'To summarize: the previous answer explained that annual leave is 30 days per Article 110(3).'\n"
        "- Missing info: 'The retrieved documents do not specify sick leave duration.'"
    ),
    verbose=True,
    allow_delegation=False,
    tools=[]  # ✅ No tools - prevents trying to call retrieval tool
)
