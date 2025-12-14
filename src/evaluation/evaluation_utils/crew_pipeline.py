from src.rag_crew.crews import build_document_crew

def run_crew_pipeline(query: str):
    """
    Executes the CrewAI RAG pipeline for a given query.
    """
    try:
        crew = build_document_crew()
        result = crew.kickoff(inputs={"query": query})
        answer = str(result)
        return {"answer": answer, "contexts": [answer]}
    except Exception as e:
        print(f"❌ Crew error: {e}")
        return {"answer": "Error", "contexts": []}
