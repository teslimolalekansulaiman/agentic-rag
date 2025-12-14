from crewai import Task

from src.rag_crew.agents import document_researcher, insight_synthesizer

research_task = Task(
    description=(
        "Use the Document Retrieval Tool to fetch context for: '{query}'\n\n"
        " ALWAYS call the tool for EVERY question, even if it seems like a follow-up.\n\n"

        "Why? Because:\n"
        "- Follow-ups like 'what about sick leave?' need NEW document retrieval\n"
        "- The chat engine automatically uses conversation context to improve retrieval\n"
        "- Only questions like 'summarize that' or 'repeat your answer' use pure history\n\n"

        "Instructions:\n"
        "1. Call the Document Retrieval Tool with the exact question\n"
        "2. Return ALL context from the tool:\n"
        "   - Retrieved Context (fresh document passages)\n"
        "   - Conversation History (previous Q&A for context)\n"
        "3. Do not modify or interpret the results"
    ),
    expected_output=(
        "Raw context from the Document Retrieval Tool, including:\n"
        "- Retrieved Context section (new document passages)\n"
        "- Conversation History section (previous exchanges)"
    ),
    agent=document_researcher,
)

# Task 2: Synthesizer analyzes context and answers
synthesis_task = Task(
    description=(
        "Read the context from the researcher and answer: '{query}'\n\n"

        "⚠️ DO NOT call any tools. You only analyze the provided context.\n\n"

        "Context structure:\n"
        "1. **Retrieved Context**: Fresh document passages (researcher ALWAYS retrieves)\n"
        "2. **Conversation History**: Previous Q&A for reference\n\n"

        "How to answer different question types:\n\n"

        "   **New questions** (e.g., 'What is the leave policy?'):\n"
        "   → Use Retrieved Context section\n"
        "   → Include article/clause references\n\n"

        "   **Follow-ups needing new info** (e.g., 'What about sick leave?'):\n"
        "   → Use Retrieved Context (researcher already got new docs)\n"
        "   → The chat engine handled conversational context automatically\n\n"

        "   **Meta questions** (e.g., 'summarize your last response', 'repeat that'):\n"
        "   → Use Conversation History section only\n"
        "   → Find the most recent assistant message\n"
        "   → Summarize or clarify it\n\n"

        "   **Hybrid questions** (e.g., 'Is that the same as sick leave?'):\n"
        "   → Check Conversation History for 'that'\n"
        "   → Use Retrieved Context for 'sick leave'\n"
        "   → Compare both\n\n"

        "Response rules:\n"
        "- Stay faithful to source text — never guess\n"
        "- Include article/clause numbers from documents\n"
        "- Be concise: 1-3 sentences maximum\n"
        "- If info is missing, explicitly state it"
    ),
    expected_output=(
        "A 1-3 sentence answer that:\n"
        "- For new/follow-up questions: Extracts facts from Retrieved Context\n"
        "- For meta questions: Summarizes/clarifies from Conversation History\n"
        "- For hybrid questions: Combines both sources appropriately\n"
        "Entirely faithful to provided context."
    ),
    agent=insight_synthesizer,
    context=[research_task],
)
