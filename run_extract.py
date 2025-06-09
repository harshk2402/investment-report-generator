import cfg
import adtdatasources.es
import extract_kpi2
import common
import pandas as pd


def dis():
    no = "0001744489-24-000276"
    search_metric = "Average Monthly Revenue Per Paid Subscriber"
    r = adtdatasources.es.ES(cfg.es_index).query_phrase(
        search_metric, "text", filter={"filename": no}
    )

    df_metrics = extract_kpi2.extract_kpi(search_metric, [d["text"] for d in r])


def dis2():
    search_metric = "Average Monthly Revenue Per Paid Subscriber"
    r = adtdatasources.es.ES(cfg.es_index).query_phrase(
        search_metric,
        "text",
        filter={"meta.symbol": "DIS", "meta.date_fiscal": "2024-09-28"},
    )

    df_metrics = extract_kpi2.extract_kpi(search_metric, [d["text"] for d in r])


def adbe():
    no = "0000796343-25-000059"
    search_metric = "Revenue by geographic area"
    # search_metric = "GMV"
    r = adtdatasources.es.ES(cfg.es_index).query_phrase(
        search_metric, "text", filter={"filename": no}
    )
    df_metrics = extract_kpi2.extract_kpi(search_metric, [d["text"] for d in r])

    search_metric = "Subscription revenue by segment"
    r = adtdatasources.es.ES(cfg.es_index).query_phrase(
        search_metric, "text", filter={"filename": no}
    )
    df_metrics = extract_kpi2.extract_kpi(search_metric, [d["text"] for d in r])
    a = 1


def bkng():
    no = "0001075531-25-000024"
    search_metric = "global average daily rates ADRs "
    # search_metric = "GMV"
    r = adtdatasources.es.ES(cfg.es_index).query_phrase(
        search_metric, "text", filter={"filename": no}
    )

    df_metrics = extract_kpi2.extract_kpi(search_metric, [d["text"] for d in r])


def company():

    q = {
        "bool": {
            "must": {
                "span_near": {
                    "clauses": [
                        {"span_term": {"text": "ulixacaltamide"}},
                        {"span_term": {"text": "expected"}},
                    ],
                    "slop": 10000,
                    "in_order": True,
                }
            },
            "filter": {"match": {"meta.symbol": "PRAX"}},
        }
    }
    r = adtdatasources.es.ES(cfg.es_index).query_raw_query(q)

    chunks = common.chunk_text_from_es_results(r)

    all_results = []

    for idx, chunk in enumerate(chunks):
        normalized_chunk = common.normalize_text(chunk)
        print(f"Processing chunk {idx + 1}/{len(chunks)}")
        result = extract_kpi2.extract_kpi(
            "Ulixacaltamide expected", normalized_chunk, {"company": "PRAX", "cik": ""}
        )
        if result.size > 0:
            all_results.append(result)

    df_final = pd.concat(all_results, ignore_index=True)
    df_final.sort_values(by="company")
    df_final.drop_duplicates(inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    try:
        common.write_df_to_excel(df_final, "kpi.xlsx")
    except Exception as e:
        print(f"Error writing DataFrame to Excel: {e}")


company()

# "must": {
#     "span_near": {
#         "clauses": [
#             {"span_term": {"text": "ulixacaltamide"}},
#             {"span_term": {"text": "expected"}},
#         ],
#         "slop": 1000,
#         "in_order": False,
#     }
# },
