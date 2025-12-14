from src.guardrail.rails.guarded_agent import guardrails
from src.guardrail.rails.guarded_crew import GuardedCrew
from src.rag_crew.crews import build_document_crew


def run(question: str):
    crew = build_document_crew()
    guarded_crew = GuardedCrew(crew=crew, guardrails=guardrails)
    result = guarded_crew.kickoff(inputs={"query": question})
    return result
