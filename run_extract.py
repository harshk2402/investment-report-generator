import cfg
import adtdatasources.es
import extract_kpi2
import common


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
            "filter": {"match": {"meta.symbol": "PRAX"}},
        }
    }

    r = adtdatasources.es.ES("utest-edu").query_raw_query(q)

    df_metrics = extract_kpi2.extract_kpi(
        "Ulixacaltamide expected", [d["_source"] for d in r]
    )
    try:
        common.write_df_to_excel(df_metrics, "metrics.xlsx")
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
