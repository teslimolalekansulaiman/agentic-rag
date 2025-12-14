from dataclasses import dataclass

@dataclass
class EvalContext:
    """
    Holds reusable dependencies and configuration for the RAGAS evaluation process.
    """
    llm: any
    embeddings: any
    excel_path: str
    sheet_name: str
    run_crew_pipeline: callable
    evaluate_single_question: callable
    print_metrics: callable
