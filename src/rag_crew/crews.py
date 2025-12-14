from crewai import Crew, Process

from src.rag_crew.agents import document_researcher, insight_synthesizer
from src.rag_crew.tasks import research_task, synthesis_task

def build_document_crew():
    return Crew(
    agents=[document_researcher, insight_synthesizer],
    tasks=[research_task, synthesis_task],
    process=Process.sequential,  # Researcher runs first, Synthesizer second
    #memory=True,
    verbose=True
)
