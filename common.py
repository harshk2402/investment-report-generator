import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch


def write_df_to_excel(df, file_path):
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        print(f"DataFrame successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing the DataFrame to Excel: {e}")


def chunk_text_from_es_results(es_results, chunk_size=400000, overlap=50000):
    full_text = " ".join(d.get("text", "") for d in es_results if "text" in d)
    normalized_text = normalize_text(full_text)

    # filename
    # meta -> symbol

    chunks = []
    for i in range(0, len(normalized_text), chunk_size - overlap):
        chunks.append(normalized_text[i : i + chunk_size])
    print(len(normalized_text))
    return chunks


def rechunk(chunks: list[str], chunk_size=500000, overlap=50000):
    """
    Rechunk the text chunks to ensure they are within the specified size limits.
    """
    normalized_text = normalize_text(" ".join(chunks))
    new_chunks = []
    for i in range(0, len(normalized_text), chunk_size - overlap):
        new_chunks.append(normalized_text[i : i + chunk_size])
    return new_chunks


def temp_data_chunks():
    import requests
    from bs4 import BeautifulSoup

    urls = [
        "https://www.sec.gov/Archives/edgar/data/1689548/000168954825000058/prax-20250331.htm",
        "https://www.sec.gov/Archives/edgar/data/1689548/000168954825000040/prax-20241231.htm",
        "https://www.sec.gov/Archives/edgar/data/1689548/000168954824000101/prax-20240930.htm",
        "https://www.sec.gov/Archives/edgar/data/1689548/000168954824000088/prax-20240630.htm",
    ]
    all_text = ""
    headers = {"User-Agent": "BiotechCatalystBot/1.0 (your.email@domain.com)"}
    for url in urls:
        response_html = requests.get(url, headers=headers)
        if response_html.status_code != 200:
            print(f"Failed to retrieve {url}. Status: {response_html.status_code}")
            continue

        soup = BeautifulSoup(response_html.text, "html.parser")
        all_text += soup.get_text(separator="\n") + "\n"

    normalized_text = normalize_text(all_text)
    norm_text_len = len(normalized_text)

    chunks = []
    chunk_size = 50000
    overlap = 10000

    for i in range(0, len(normalized_text), chunk_size - overlap):
        chunks.append(all_text[i : i + chunk_size])
    return chunks


def get_relevant_chunks(chunks: list[str], search_metric: str, top_k=9):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_embeddings = model.encode(
        chunks, show_progress_bar=True, convert_to_tensor=True
    )
    search_embedding = model.encode(search_metric, convert_to_tensor=True)

    cos_scores = util.cos_sim(chunk_embeddings, search_embedding).squeeze()

    n_closest = min(cos_scores.shape[0], top_k)

    top_results = torch.topk(cos_scores, k=n_closest, largest=True, sorted=True)

    top_chunks = [chunks[idx] for idx in top_results.indices]
    total_top_length = sum(len(chunk) for chunk in top_chunks)
    return top_chunks


# def normalize_text(text: str) -> str:
#     # Replace multiple newlines with a double newline (paragraph breaks)
#     text = re.sub(r"\n\s*\n+", "\n\n", text)

#     # Replace single newlines within sentences with a space
#     text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

#     # Remove excessive spaces
#     text = re.sub(r"[ \t]+", " ", text)

#     # Strip leading/trailing whitespace
#     return text.strip()


def normalize_text(text: str) -> str:
    # Step 0: Standardize all common newline representations to '\n'
    # This is crucial for cross-platform compatibility and consistent regex matching later.
    text = text.replace("\r\n", "\n")  # Windows newlines
    text = text.replace("\r", "\n")  # Old Mac newlines

    # Step 1: Replace multiple newlines with a double newline (paragraph breaks)
    # This now operates on a consistent '\n' base, making it more reliable.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Step 2: Replace single newlines within sentences with a space
    # Also benefits from the standardized '\n' base.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Step 3: Remove excessive spaces (including tabs and non-breaking spaces)
    # \xa0 is the Unicode for non-breaking space, common in web/document text.
    text = re.sub(r"[ \t\xa0]+", " ", text)

    # Step 4: Strip leading/trailing whitespace from the entire text
    return text.strip()
