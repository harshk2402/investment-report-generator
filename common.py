import pandas as pd


def write_df_to_excel(df, file_path):
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        print(f"DataFrame successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing the DataFrame to Excel: {e}")


def chunk_text_from_es_results(es_results, chunk_size=12000, overlap=5000):
    full_text = " ".join(d.get("text", "") for d in es_results if "text" in d)

    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i : i + chunk_size])
    return chunks
