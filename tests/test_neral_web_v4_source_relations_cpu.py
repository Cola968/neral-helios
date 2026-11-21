"""Source-first browser-engine inference from complete literal documentation statements."""
import unittest
from neral_v32_verifier import _property_evidence

class SourceRelations(unittest.TestCase):
    def test_reverse_list_claim_from_primary_browser_engine_documentation(self):
        text=('Blink serves as the rendering engine for Chromium-based browsers, including '
              'Chrome, Android WebView, Microsoft Edge, Opera, and Brave.')
        for product in ('Chrome','Microsoft Edge','Opera','Brave'):
            with self.subTest(product=product):
                self.assertEqual(_property_evidence(text,product,'engine')[0],'Blink')
        self.assertIsNone(_property_evidence(text,'Safari','engine'))

    def test_direct_source_relationships_are_valid_even_without_uses_engine_phrasing(self):
        samples=(
          ('Brave','Brave is built on top of the Chromium project and the Blink rendering engine.','Blink'),
          ('Vivaldi','Vivaldi is based on Chromium and uses the Blink browser engine for pages.','Blink'),
          ('Tor Browser','Tor Browser on desktop is built upon Mozilla Firefox / Gecko components.','Gecko'),
          ('Chromium','Blink serves as the rendering engine for Chromium-based browsers.','Blink'),
        )
        for subject,line,expected in samples:
            with self.subTest(subject=subject):
                self.assertEqual(_property_evidence(line,subject,'engine')[0],expected)

    def test_ios_webkit_is_literal_but_not_valid_desktop_proof(self):
        samples=(
          ('Brave','On iOS, the Brave browser is powered by Apple WebKit for rendering.','WebKit'),
          ('Vivaldi','Vivaldi for iOS is built on top of Apple WebKit.','WebKit'),
          ('Firefox','Firefox for iOS uses WKWebKit to render web pages.','WKWebKit'),
        )
        for subject,line,expected in samples:
            with self.subTest(subject=subject):
                self.assertIsNone(_property_evidence(line,subject,'engine'))
                self.assertEqual(_property_evidence(line,subject,'engine',platform='ios')[0],expected)

    def test_unrelated_mention_is_not_subject_property_proof(self):
        self.assertIsNone(_property_evidence(
          'Firefox is a browser. Chrome uses the Blink rendering engine.',
          'Firefox','engine'))
        self.assertIsNone(_property_evidence(
          'Brave mentions that Safari uses WebKit as an engine.',
          'Brave','engine'))

if __name__=='__main__':unittest.main()

class AuthoritativeLivePageCandidates(unittest.TestCase):
    def test_primary_page_is_only_a_url_candidate_and_requires_fresh_page_proof(self):
        from neral_evidence_agent import run_evidence_agent
        class Search:
            def search(self,query):
                raise AssertionError('Do not search if fresh official source already proves each fact')
        def reader(hits,query):
            pages={
                'developer.chrome.com': 'Blink serves as the rendering engine for Chromium-based browsers, including Chrome, Microsoft Edge, Opera and Brave.',
                'developer.mozilla.org': 'Firefox uses the Gecko rendering engine to display webpages.'
            }
            result=[]
            for h in hits:
                import urllib.parse
                host=urllib.parse.urlsplit(h['url']).hostname
                if host in pages:
                    result.append({'url':h['url'],'page_read':True,'page_text':pages[host]})
            return result
        r=run_evidence_agent('Vergleiche Chrome vs Firefox: Browser-Engine.',
              lambda *a:None,lambda x:None,web_query='Chrome Firefox Browser-Engine',
              search=Search(),page_reader=reader,fast_web=True)
        self.assertTrue(r['tool_verified'])
        self.assertIn('Chrome: Blink',r['text'])
        self.assertIn('Firefox: Gecko',r['text'])
        self.assertTrue(all(s['page_read'] for s in r['sources']))

    def test_invalid_official_page_never_creates_a_verified_source(self):
        from neral_evidence_agent import run_evidence_agent
        class Search:
            def search(self,q):return {'results':[]}
        def reader(hits,query):
            return [{'url':h['url'],'page_read':True,
                     'page_text':'Chrome has a browser. Safari uses WebKit rendering engine.'} for h in hits]
        r=run_evidence_agent('Vergleiche Chrome vs Firefox: Browser-Engine.',
              lambda *a:None,lambda x:None,web_query='Chrome Firefox Browser-Engine',
              search=Search(),page_reader=reader,fast_web=True)
        self.assertFalse(r['tool_verified'])
