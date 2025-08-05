import hashlib
import pandas as pd
import re
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


def normalize_text(text: str) -> str:
    # Step 0: Standardize all common newline representations to '\n'
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
        "PRAX": {"cik": "0001689548", "name": "Praxis Precision Medicines Inc"},
        "LLY": {"cik": "0000059478", "name": "Eli Lilly and Company"},
        "MNMD": {"cik": "0001813814", "name": "Mind Medicine (MindMed) Inc"},
        "PTCT": {"cik": "0001070081", "name": "PTC Therapeutics Inc"},
        "VRTX": {"cik": "0000875320", "name": "Vertex Pharmaceuticals Inc"},
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


def event_identity_key(event) -> str:
    """Returns a hash of stable identifying fields to detect semantically duplicate events."""
    key_str = (
        f"{event.accession_number}|{event.drug}|{event.study}|{event.phase}".lower()
    )
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()
