from src.common.config import settings
from src.evaluation.evaluation_utils.crew_pipeline import run_crew_pipeline
from src.evaluation.evaluation_utils.evaluation_models import EvalContext
from src.evaluation.evaluation_utils.postprocess import update_results_and_summary, print_metrics, \
    process_single_question
from src.evaluation.evaluation_utils.preprocess import load_questions_to_process
from src.evaluation.evaluation_utils.evaluation_factory import get_llm, get_embeddings, evaluate_single_question
from dotenv import load_dotenv



def run_ragas(excel_path, sheet_name="Sheet1", question_ids=None, use_row_numbers=True):

    load_dotenv()
    print(f"📖 Starting RAGAS evaluation on {excel_path}")

    df, df_to_process, _ = load_questions_to_process(excel_path, sheet_name, question_ids, use_row_numbers)
    if df_to_process.empty:
        print("No questions to process.")
        return

    # 🔧 Prepare shared evaluation context
    ctx = EvalContext(
        llm=get_llm(),
        embeddings=get_embeddings(),
        excel_path=excel_path,
        sheet_name=sheet_name,
        run_crew_pipeline=run_crew_pipeline,
        evaluate_single_question=evaluate_single_question,
        print_metrics=print_metrics,
    )

    # 🔁 Process each question
    total = len(df_to_process)
    for i, (idx, row) in enumerate(df_to_process.iterrows(), 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{total}] ⚙️ Evaluating: {str(row['question'])[:100]}{'...' if len(str(row['question'])) > 100 else ''}")
        print(f"{'=' * 70}")

        process_single_question(df, idx, row, ctx)

    # 📊 Final summary
    update_results_and_summary(df, df_to_process, excel_path, sheet_name)


if __name__ == "__main__":
    run_ragas(
        excel_path=f"{settings.EVALUATION_DATASET_DIR}/golden.xlsx",
        question_ids=  [1],
        sheet_name="Sheet1",
    )
