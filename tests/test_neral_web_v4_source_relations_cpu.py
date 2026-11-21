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

class LocalizedOfficialDocsContracts(unittest.TestCase):
    def test_german_chromium_official_reverse_claim(self):
        line=('Blink dient als Rendering-Engine für Chromium-basierte Browser wie Chrome, '
              'Android WebView, Microsoft Edge, Opera und Brave.')
        for name in ('Chrome','Chromium','Microsoft Edge','Opera','Brave'):
            with self.subTest(name=name):
                fact=_property_evidence(line,name,'engine')
                self.assertIsNotNone(fact)
                self.assertEqual(fact[0],'Blink')
        self.assertIsNone(_property_evidence(line,'Safari','engine'))

    def test_no_generic_word_or_negated_gecko_as_engine(self):
        wrong=("Opera's layout engine is also found in Virtual Mechanics SiteSpinner Pro.",
               'Firefox iOS is built natively for iOS, and doesn`t use Gecko.')
        self.assertIsNone(_property_evidence(wrong[0],'Opera','engine'))
        self.assertIsNone(_property_evidence(wrong[1],'Firefox','engine',platform='ios'))

    def test_consecutive_official_mozilla_ios_documentation(self):
        doc=('Firefox iOS is built natively for iOS, and doesn`t use Gecko.\n'
             'Due to Apple`s restrictions for browsers on iOS, WKWebKit is how users '
             'will interact with the web, and how developers interact with web elements.')
        fact=_property_evidence(doc,'Firefox','engine',platform='ios')
        self.assertIsNotNone(fact)
        self.assertEqual(fact[0],'WKWebKit')
        self.assertIsNone(_property_evidence(doc,'Firefox','engine',platform='desktop'))

class UpToDateIOSSourceRelations(unittest.TestCase):
    def test_german_primary_chrome_ios_sentence(self):
        source=('Unter iOS und iPadOS verwendet Chrome WebKit als Rendering-Engine. '
                'Safari verwendet WebKit, aber auf Desktop nutzt Chrome Blink.')
        self.assertEqual(_property_evidence(source,'Chrome','engine',platform='ios')[0],'WebKit')
        self.assertEqual(_property_evidence(source,'Chrome','engine',platform='desktop')[0],'Blink')

    def test_vivaldi_2026_iOS_company_blog_context(self):
        source=('Today marks three years since we released Vivaldi for iOS. '
                'Hurray! We released Vivaldi on Android seven years ago. So what explains the iOS delay? '
                'Quite simply, Apple caused the delay.\n'
                'Globally, the only way to distribute apps for iOS and iPad iOS is through Apple’s App Store. '
                'Since the dawn of time, one of Apple’s rules requires that every app that browses '
                'the web uses Apple WebKit, the same browser engine that Apple uses to power Safari. '
                'You can’t choose which version of WebKit you use; it has to be the one that Apple ships.')
        result=_property_evidence(source,'Vivaldi','engine',platform='ios')
        self.assertIsNotNone(result)
        self.assertEqual(result[0],'WebKit')
        self.assertTrue(result[1] in source)
        self.assertIsNone(_property_evidence(source,'Vivaldi','engine',platform='desktop'))

    def test_other_browser_does_not_inherit_vivaldi_ios_article_engine(self):
        text=('Today marks three years since we released Vivaldi for iOS.\n'
              'Every app that browses the web uses Apple WebKit.')
        self.assertIsNone(_property_evidence(text,'Chrome','engine',platform='ios'))

class StrictPerSubjectSourceAttribution(unittest.TestCase):
    def test_verifier_never_uses_one_subjects_page_to_overwrite_another(self):
        from neral_v32_verifier import property_grounded_web_answer
        sources=[
            {'url':'https://evidence.example.org/librewolf','page_read':True,
             'entity_focus':'LibreWolf',
             'page_text':'LibreWolf uses the Gecko rendering engine on desktop. Chrome uses the Gecko rendering engine on desktop.'},
            {'url':'https://evidence.example.org/chrome','page_read':True,
             'entity_focus':'Chrome',
             'page_text':'Chrome uses the Blink rendering engine on desktop.'},
        ]
        result=property_grounded_web_answer(('LibreWolf','Chrome'),sources,'Browser-Engine')
        self.assertEqual(result['text'].splitlines()[:2],['LibreWolf: Gecko','Chrome: Blink'])
        self.assertIn('[web2]',result['text'])

    def test_missing_subject_page_does_not_borrow_someone_elses_claim(self):
        from neral_v32_verifier import property_grounded_web_answer
        sources=[{'url':'https://evidence.example.org/librewolf', 'page_read':True,
                  'entity_focus':'LibreWolf',
                  'page_text':'LibreWolf uses Gecko rendering engine. Chrome uses Blink rendering engine.'},
                 {'url':'https://evidence.example.org/chrome', 'page_read':False,
                  'entity_focus':'Chrome', 'page_text':''}]
        self.assertIsNone(property_grounded_web_answer(('LibreWolf','Chrome'),sources,'Browser-Engine'))
