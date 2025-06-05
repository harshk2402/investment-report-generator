import adtiam

adtiam.load_creds('adt-sources')
adtiam.load_creds('adt-llm')
adtiam.load_creds('adt-db')
r = adtiam.load_creds('adt-db')


adtiam.check_keys_loaded(['sources.secapid2v','llm.openai','db'])

if adtiam.env=='utest':
    es_index = 'utest-edu'
else:
    es_index = 'filings-sec-textonly'

import sec_api

cnxn_sec_idx = sec_api.QueryApi(api_key=adtiam.creds['sources']['secapid2v']['key'])
cnxn_sec_docs = sec_api.RenderApi(api_key=adtiam.creds['sources']['secapid2v']['key'])

import d6tflow2.settings.es

d6tflow2.settings.es.init(adtiam.creds['db']['elastics']['cloudid'],adtiam.creds['db']['elastics']['key'])
