import pandas as pd
import re


def write_df_to_excel(df, file_path):
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        print(f"DataFrame successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing the DataFrame to Excel: {e}")


def chunk_text_from_es_results(es_results, chunk_size=110000, overlap=20000):
    full_text = " ".join(d.get("text", "") for d in es_results if "text" in d)

    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i : i + chunk_size])
    return chunks


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

    chunks = []
    chunk_size = 110000  # ~8k tokens if text is dense
    overlap = 20000

    for i in range(0, len(all_text), chunk_size - overlap):
        chunks.append(all_text[i : i + chunk_size])
    return chunks


def normalize_text(text: str) -> str:
    # Replace multiple newlines with a double newline (paragraph breaks)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Replace single newlines within sentences with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()
