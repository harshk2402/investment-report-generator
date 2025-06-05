from datetime import datetime
import adtdatasources.es
from adtdatasources.es import ES

es_instance = ES("utest-edu")  # your index name here

es_client = es_instance.cnxn_es  # this is the raw elasticsearch-py client
doc = {
    "meta": {"test": True},
    "text": "Test document",
    "timestamp": datetime.now().isoformat(),
}

try:
    res = es_client.index(index="utest-edu", body=doc)
    print("✅ Indexed:", res)
except Exception as e:
    print("❌ Write failed:", e)
