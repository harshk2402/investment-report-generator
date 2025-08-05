import cfg
import adtdatasources.es
import extract_kpi2
import common
import pandas as pd
from faiss_manager import FAISSManager
import press_release
from datetime import datetime
import time
import os


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

    company_results = {}

    companies = ["PRAX", "LLY", "MNMD", "PTCT", "VRTX"]

    os.makedirs("output", exist_ok=True)

    writer_raw = pd.ExcelWriter(
        "./output/kpi_output.xlsx",
        engine="openpyxl",
        mode="w",
    )

    writer_val = pd.ExcelWriter(
        "./output/kpi_output_validated.xlsx",
        engine="openpyxl",
        mode="w",
    )

    last_request_time = time.time() - 60

    for company in companies:
        filings, metadatas = common.get_filing_sections(company, start_date)
        vector_store.add_filings(filings, metadatas, company)

        filings, metadatas = press_release.get_press_releases([company], start_date)
        vector_store.add_filings(filings, metadatas, company, isPressRelease=True)

        k = 20

        documents = vector_store.similarity_search_with_context(
            company, search_metric, k=k, window=2
        )

        all_results = []
        chunks = common.format_documents_for_prompt(
            documents, chunk_size=900000, chunk_overlap=0
        )

        for idx, chunk in enumerate(chunks):
            time_since_last = time.time() - last_request_time

            if time_since_last < 60:
                print(f"Waiting {60 - time_since_last:.2f}s to avoid rate limit...")
                time.sleep(60 - time_since_last)

            last_request_time = time.time()

            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            result = extract_kpi2.extract_kpi(
                search_metric, chunk, writer_raw, company
            )  # Via gemini api
            if result.size > 0:
                all_results.append(result)

        if len(all_results) > 0:
            df_company = pd.concat(all_results, ignore_index=True)
            df_company.sort_values(by="company")
            df_company.drop_duplicates(inplace=True)
            df_company.reset_index(drop=True, inplace=True)
            company_results[company] = df_company
        else:
            print(f"No KPIs extracted for {company}")
            company_results[company] = pd.DataFrame()

    try:
        print("Writing validated DataFrame to Excel...")
        for company, df in company_results.items():
            if not df.empty:
                df.to_excel(writer_val, sheet_name=company, index=False)
            else:
                print(f"No data to write for {company}")

    except Exception as e:
        print(f"Error writing DataFrame to Excel: {e}")

    writer_raw.close()
    writer_val.close()


company()
