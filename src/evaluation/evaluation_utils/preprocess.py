import pandas as pd

def load_questions_to_process(excel_path, sheet_name, question_ids, use_row_numbers):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    required_cols = ["question", "ground_truth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    metric_cols = ["answer", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = None

    id_column = "id"
    if id_column not in df.columns:
        df[id_column] = range(1, len(df) + 1)

    # Determine subset
    if question_ids:
        if use_row_numbers:
            indices = [i - 1 for i in question_ids if 1 <= i <= len(df)]
            df_to_process = df.iloc[indices]
        else:
            df_to_process = df[df[id_column].isin(question_ids)]
    else:
        df_to_process = df.copy()

    print(f"🧩 Loaded {len(df)} total, selected {len(df_to_process)} rows for processing")
    return df, df_to_process, id_column
