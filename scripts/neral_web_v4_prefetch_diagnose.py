"""READ ONLY: read fresh primary pages from URLs, print source proof status, no keys or golds."""
import json
from neral_page_reader import read_entity_pages, read_top_pages
from neral_evidence_agent import _V4_BROWSER_REFERENCE_URLS, _V4_IOS_REFERENCE_URLS
from neral_v32_verifier import _property_evidence

for platform,sources in (('desktop',_V4_BROWSER_REFERENCE_URLS),('ios',_V4_IOS_REFERENCE_URLS)):
    for name,urls in sources.items():
        def ok(page):
            return _property_evidence(page.get('page_text',''),name,'engine',platform=platform) is not None
        found=read_entity_pages([{'url':u} for u in urls],name+' browser rendering engine '+platform,name,
                                  reader=read_top_pages,max_pages=1,predicate=ok)
        value=_property_evidence(found[0]['page_text'],name,'engine',platform=platform)[0] if found else ''
        host=found[0]['url'].split('/')[2] if found else ''
        print('NERAL_V4_URL_SOURCE_STATUS '+json.dumps({'subject':name,'platform':platform,
             'literal_property_present':bool(found),'literal_property':value,'verified_source_host':host}),flush=True)
print('NERAL_V4_URL_PREFETCH_DIAG_FINISHED',flush=True)
