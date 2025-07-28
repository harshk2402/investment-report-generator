import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import time
import adtiam
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
from datetime import datetime


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


def get_filing_sections(ticker="PRAX", start_date="2025-01-01") -> tuple:
    sec_api_key = adtiam.creds["sources"]["secapid2v"]["key"]

    company_data = {
        "MNMD": {"cik": "0001813814", "name": "Mind Medicine (MindMed) Inc"},
        "PTCT": {"cik": "0001070081", "name": "PTC Therapeutics Inc"},
        "BIIB": {"cik": "0000875045", "name": "Biogen Inc"},
        "GILD": {"cik": "0000882095", "name": "Gilead Sciences Inc"},
        "VRTX": {"cik": "0000875320", "name": "Vertex Pharmaceuticals Inc"},
        "PRAX": {"cik": "0001689548", "name": "Praxis Precision Medicines Inc"},
    }

    company_name = company_data.get(ticker, {}).get("name", ticker)

    today = datetime.today()

    print(f"\nFetching 10-Q filings for {ticker} (since {start_date})...")

    # Find 10-K/10-Q filings
    query_url = f"https://api.sec-api.io?token={sec_api_key}"
    query_payload = {
        "query": f'ticker:{ticker} AND formType:"10-Q" AND filedAt:[{start_date} TO {today.strftime("%Y-%m-%d")}]',
        "from": 0,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    # query_url = f"https://api.sec-api.io?token={sec_api_key}"
    # query_payload = {
    #     "query": f'ticker:{ticker} AND (formType:"10-K" OR formType:"10-Q")',
    #     "from": 0,
    #     "size": 10,
    #     "sort": [{"filedAt": {"order": "desc"}}],
    # }
    # query_payload = {
    #     "query": f"ticker:{ticker}",
    #     "from": 0,
    #     "size": 100,
    #     "sort": [{"filedAt": {"order": "desc"}}],
    # }

    query_response = requests.post(
        query_url, json=query_payload, headers={"Content-Type": "application/json"}
    )
    metadata = []
    query_data = query_response.json()

    documents_texts = []
    documents_metadata = []

    print(
        "Retrieved", len(query_data.get("filings", [])), "filings for ticker:", ticker
    )

    for filing in query_data.get("filings", []):
        form_type = filing.get("formType", "")
        if form_type not in ("10-K", "10-Q"):
            continue

        filing_url = filing["linkToFilingDetails"]

        # Extract sections
        filing_metadata = {
            "accession": filing.get("accessionNo", ""),
            "filing_date": filing.get("filedAt", ""),
            "company_name": company_name,
            "form_type": form_type,
            "ticker": ticker,
            "cik": company_data.get(ticker, {}).get("cik", ""),
        }
        metadata.append(filing_metadata)
        if form_type == "10-K":
            sections_to_extract = ["1", "1A", "2", "3", "7"]
        elif form_type == "10-Q":
            sections_to_extract = [
                "part1item2",  # MD&A ✅
                "part2item1",  # Legal Proceedings ✅
                "part2item1a",  # Risk Factors ✅
                "part2item5",  # Other Information ✅
                "part1item1",  # (Optional) Financials, for spending patterns
            ]  # Use alias to distinguish MD&A
        extractor_base_url = f"https://api.sec-api.io/extractor?token={sec_api_key}"

        filing_text = ""
        for section in sections_to_extract:
            section_params = {"url": filing_url, "item": section, "type": "text"}
            section_response = requests.get(extractor_base_url, params=section_params)
            if section_response.status_code == 200:
                section_text = section_response.text
                if section_text and section_text.strip():
                    filing_text += f"\n\n=== ITEM {section} ===\n{section_text}"
            else:
                print(
                    f"Failed to extract section {section} from {filing_url}. Status: {section_response.status_code}"
                )

            time.sleep(0.5)

        # Clean text for this filing
        filing_text = re.sub(r"<[^>]+>", " ", filing_text)
        filing_text = re.sub(r"\s+", " ", filing_text)
        filing_text = normalize_text(filing_text)

        documents_texts.append(filing_text)
        documents_metadata.append(filing_metadata)

    print(f"Extracted {len(documents_texts)} sections from {ticker} filings.")

    return documents_texts, documents_metadata


def format_documents_for_prompt(
    documents: list[Document], chunk_size: int = 900000, chunk_overlap: int = 0
) -> list[str]:
    grouped = defaultdict(list)

    for doc in documents:
        meta = doc.metadata
        key = (
            meta.get("company_name", ""),
            meta.get("form_type", ""),
            meta.get("filing_date", ""),
            meta.get("accession", ""),
        )
        grouped[key].append(doc.page_content)

    output_lines = []
    for key, chunks in grouped.items():
        company_name, form_type, filing_date, accession = key
        output_lines.append(f"Company Name: {company_name}")
        output_lines.append(f"Form Type: {form_type}")
        output_lines.append(f"Filing Date: {filing_date}")
        output_lines.append(f"Accession: {accession}")
        output_lines.append("")
        for chunk in chunks:
            output_lines.append("--- TEXT START ---")
            output_lines.append(chunk)
            output_lines.append("--- TEXT END ---")
            output_lines.append("")

    output = "\n".join(output_lines)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(output)

    return chunks
