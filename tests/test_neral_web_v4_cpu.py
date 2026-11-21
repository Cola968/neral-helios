"""V4: bounded parallel source-only web pipeline and platform proof gates."""
import json
import time
import unittest
from unittest.mock import Mock

from neral_v32_verifier import _property_evidence
from neral_evidence_agent import run_evidence_agent


class WebV4Contracts(unittest.TestCase):
    def test_parallel_entities_return_verified_provenance_without_model(self):
        pages={
            'Astra':'Astra currently uses the Aurora rendering engine for desktop web pages.',
            'Boreal':'Boreal currently uses the Borealis rendering engine on desktop.',
        }
        class Search:
            def search(self,q):
                time.sleep(.11)
                name='Astra' if 'Astra' in q else 'Boreal'
                return {'results':[{'url':'https://example.org/'+name}]}
        def reader(hits,q):
            name=hits[0]['url'].split('/')[-1]
            time.sleep(.11)
            return [{'url':hits[0]['url'],'page_read':True,'page_text':pages[name]}]
        calls=[]
        start=time.monotonic()
        result=run_evidence_agent(
            'Astra vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.',
            lambda *a:calls.append(a),lambda x:None,
            web_query='Astra Boreal Browser-Engine',search=Search(),page_reader=reader,
            fast_web=True)
        elapsed=time.monotonic()-start
        self.assertLess(elapsed,.35)  # sequential two-entity path >= .44 s
        self.assertEqual(calls,[])
        self.assertTrue(result['tool_verified'])
        self.assertIn('Astra: Aurora',result['text'])
        self.assertIn('Boreal: Borealis',result['text'])
        self.assertEqual(len(result['sources']),2)
        self.assertTrue(all(s['page_read'] for s in result['sources']))

    def test_desktop_is_default_and_ios_only_is_not_a_desktop_proof(self):
        ios='Firefox for iOS uses WebKit rendering engine for browsing pages.'
        desktop='Firefox uses the Gecko rendering engine on desktop browsers.'
        self.assertIsNone(_property_evidence(ios,'Firefox','engine'))
        self.assertEqual(_property_evidence(desktop+' '+ios,'Firefox','engine')[0],'Gecko')
        self.assertIn(_property_evidence(ios,'Firefox','engine',platform='ios')[0],('WebKit','WKWebKit'))

    def test_unsupported_browser_platform_never_invents_a_source(self):
        class Search:
            def search(self,q):return {'results':[]}
        result=run_evidence_agent(
            'Firefox vs Brave: Vergleiche Desktop-Browser-Engine anhand Quellen.',
            lambda *a: (_ for _ in ()).throw(AssertionError('No model should run')),
            lambda x:None,web_query='Firefox Brave Desktop Browser-Engine',
            search=Search(),page_reader=lambda *a:[],fast_web=True)
        self.assertFalse(result['tool_verified'])
        self.assertEqual(result['error_code'],'source_evidence_missing')

    def test_fast_web_is_disabled_by_default_for_old_pipeline(self):
        import inspect
        from neral_evidence_agent import run_evidence_agent
        self.assertFalse(inspect.signature(run_evidence_agent).parameters['fast_web'].default)

if __name__=='__main__': unittest.main()

class V4ReaderAndCacheSafety(unittest.TestCase):
    def test_v4_reader_has_existing_bounded_parallelism_and_connect_deadline(self):
        from neral_page_reader import TIMEOUT, CONNECT_TIMEOUT, _executor
        self.assertLessEqual(TIMEOUT,4.0)
        self.assertLessEqual(CONNECT_TIMEOUT,2.5)
        self.assertGreaterEqual(_executor._max_workers,3)

    def test_previous_web_answers_never_fulfilled_from_training_notes(self):
        from pathlib import Path
        from neral_verified_notes import exact_note_lookup
        seed=Path(__file__).resolve().parents[1]/'datasets'/'neral_v39_seed97.jsonl'
        data=[json.loads(x) for x in seed.read_text(encoding='utf8').splitlines()]
        for item in (x for x in data if x['meta']['category']=='web'):
            self.assertIsNone(exact_note_lookup(item['instruction'],seed))
        self.assertIsNone(exact_note_lookup('Wie ist heute die aktuelle Firefox-Engine?',seed))

    def test_iOS_scope_rejects_desktop_only_proof(self):
        self.assertIsNone(_property_evidence(
            'Firefox uses the Gecko rendering engine for desktop applications.',
            'Firefox','engine',platform='ios'))
        self.assertEqual(_property_evidence(
            'Firefox for iOS uses the WebKit rendering engine.',
            'Firefox','engine',platform='ios')[0], 'WebKit')

class BrowserEntityNames(unittest.TestCase):
    def test_multiword_browser_name_preserved(self):
        from neral_evidence_agent import _comparison_entities
        self.assertEqual(_comparison_entities('Vergleiche Tor Browser vs Firefox: Browser-Engine'),
                         ('Tor Browser','Firefox'))
        self.assertEqual(_comparison_entities('Vergleiche Microsoft Edge vs Tor Browser: Browser-Engine'),
                         ('Microsoft Edge','Tor Browser'))
        self.assertEqual(_comparison_entities('Chrome vs Mozilla Firefox: Browser-Engine'),
                         ('Chrome','Firefox'))
