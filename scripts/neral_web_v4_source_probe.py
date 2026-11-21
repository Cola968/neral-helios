"""Read-only source reachability/provenance probe; outputs no page or user text."""
import json
from neral_page_reader import read_top_pages
from neral_v32_verifier import _property_evidence
from neral_web_search import WebSearch

SOURCES={
 'Firefox':['https://developer.mozilla.org/en-US/docs/Glossary/Mozilla_Firefox',
            'https://firefox-source-docs.mozilla.org/overview/gecko.html'],
 'Chrome':['https://en.wikipedia.org/wiki/Google_Chrome',
           'https://blog.chromium.org/2013/04/blink-rendering-engine-for-chromium.html'],
 'Safari':['https://en.wikipedia.org/wiki/Safari_(web_browser)'],
 'Brave':['https://en.wikipedia.org/wiki/Brave_(web_browser)',
          'https://brave.com/learn/fastest-web-browser/'],
 'Edge':['https://en.wikipedia.org/wiki/Microsoft_Edge'],
 'Vivaldi':['https://vivaldi.com/blog/ios/happy-3rd-birthday-vivaldi-on-ios/'],
 'Opera':['https://en.wikipedia.org/wiki/Opera_(web_browser)'],
 'Tor Browser':['https://onionbrowser.com/faqs',
                'https://en.wikipedia.org/wiki/Tor_Browser'],
 'Waterfox':['https://en.wikipedia.org/wiki/Waterfox'],
 'LibreWolf':['https://en.wikipedia.org/wiki/LibreWolf'],
}
for name,urls in SOURCES.items():
    pages=read_top_pages([{'url':url} for url in urls],name+' browser rendering engine')
    for p in pages:
        proof=_property_evidence(p['page_text'],name,'engine')
        print('NERAL_V4_SOURCE_PROBE '+json.dumps({
            'subject':name,'host':p['url'].split('/')[2],
            'page_read':p['page_read'],'text_chars':len(p['page_text']),
            'property_proven':proof is not None},sort_keys=True),flush=True)
for subject in ('Firefox','Chrome','Safari'):
    try:
        hits=WebSearch(max_per_minute=6).search(subject+' browser rendering engine desktop')
        count=len(hits.get('results',[]))
        print('NERAL_V4_PROVIDER_PROBE '+subject+' hits='+str(count),flush=True)
    except Exception as exc:
        print('NERAL_V4_PROVIDER_PROBE '+subject+' error='+type(exc).__name__,flush=True)
print('NERAL_V4_SOURCE_DIAG_COMPLETE',flush=True)
