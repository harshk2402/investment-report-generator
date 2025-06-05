import cfg
import elasticsearch.helpers as eshelpers
import adtdatasources.es


es = adtdatasources.es.ES('utest-edu')

r = es.query_phrase('Average Monthly Revenue Per Paid Subscriber','text',filter={'meta.symbol':'DIS','meta.date_fiscal':'2024-09-28'})

# Example of inserting multiple documents
test_docs = [
    {
        'text': 'This is a test document about revenue metrics for Apple',
        'meta': {
            'symbol': 'AAPL',
            'date_fiscal': '2024-03-31'
        }
    },
    {
        'text': 'Quarterly revenue analysis for Microsoft',
        'meta': {
            'symbol': 'MSFT',
            'date_fiscal': '2024-03-31'
        }
    },
    {
        'text': 'Annual revenue report for Google',
        'meta': {
            'symbol': 'GOOGL',
            'date_fiscal': '2024-03-31'
        }
    }
]

# Prepare bulk actions
actions = [
    {
        "_index": 'utest-edu',
        "_source": doc
    }
    for doc in test_docs
]

# Bulk insert documents
eshelpers.bulk(es.cnxn, actions)

# Query all documents from the same fiscal date
r2 = es.query_phrase('revenue', 'text', filter={'meta.date_fiscal': '2024-03-31'})

# Query specific company
r3 = es.query_phrase('revenue', 'text', filter={'meta.symbol': 'AAPL', 'meta.date_fiscal': '2024-03-31'})


