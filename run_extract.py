import cfg
import adtdatasources.es
import extract_kpi2
import common
import pandas as pd
from faiss_manager import FAISSManager
import press_release
from datetime import datetime
import time


def company():
    search_metric = "all clinical trial activity, study results, and regulatory events"

    vector_store = FAISSManager()

    today = datetime.today()

    # Go back 3 years
    three_years_ago = today.replace(year=today.year - 3)

    # Return Jan 1 of that year
    start_date = datetime(year=three_years_ago.year, month=1, day=1).strftime(
        "%Y-%m-%d"
    )

    companies = ["PRAX", "LLY", "MNMD", "PTCT", "VRTX"]

    for company in companies:
        filings, metadatas = common.get_filing_sections(company, start_date)
        vector_store.add_filings(filings, metadatas)

        filings, metadatas = press_release.get_press_releases([company], start_date)
        vector_store.add_filings(filings, metadatas, isPressRelease=True)

    k = 20

    documents = vector_store.similarity_search_with_context(
        search_metric, k=k, window=2
    )

    all_results = []
    chunks = common.format_documents_for_prompt(
        documents, chunk_size=900000, chunk_overlap=0
    )

    last_request_time = time.time() - 60

    for idx, chunk in enumerate(chunks):
        time_since_last = time.time() - last_request_time

        if time_since_last < 60:
            time.sleep(60 - time_since_last)

        last_request_time = time.time()

        print(f"Processing chunk {idx + 1}/{len(chunks)}")
        result = extract_kpi2.extract_kpi(search_metric, chunk)  # Via gemini api
        if result.size > 0:
            all_results.append(result)

    df_final = pd.concat(all_results, ignore_index=True)
    df_final.sort_values(by="company")
    df_final.drop_duplicates(inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    try:
        common.write_df_to_excel(df_final, "./output/kpi_validated.xlsx")
    except Exception as e:
        print(f"Error writing DataFrame to Excel: {e}")


company()
