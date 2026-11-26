import unittest
from unittest.mock import Mock
from neral_web_search import live_topic_matches, live_topic_hits, WebSearch
from neral_page_reader import read_top_pages
from neral_v43_live_web import search_live_sources

class UniversalSearchRegression(unittest.TestCase):
    def test_plain_query_preserves_subject_across_domains(self):
        cases = [
            ('suche nach dem flughafen leipzig im internet', 'Flughafen Hannover', 'Leipzig/Halle Airport'),
            ('Suche nach Universität Potsdam', 'Universität Hamburg', 'Universität Potsdam'),
            ('Suche nach dem Computerspiel PEAK', 'Computerspiel Minecraft', 'PEAK game'),
            ('search for the airline Air Baltic', 'Air France airline', 'Air Baltic airline'),
            ('Suche nach dem Planeten Neptun', 'Planet Mars', 'Planet Neptune Neptun'),
            ('Suche nach der Firma Siemens', 'Firma Bosch', 'Siemens company'),
        ]
        for query, wrong, right in cases:
            with self.subTest(query=query):
                self.assertFalse(live_topic_matches(query, wrong))
                self.assertTrue(live_topic_matches(query, right))
    def test_reader_rejects_wrong_body_after_relevant_title(self):
        q='suche nach dem flughafen leipzig im internet'
        hit={'url':'https://example.org/leipzig','title':'Flughafen Leipzig'}
        fetch=Mock(return_value={'url':hit['url'],'text':'Flughafen Hannover. Informationen zum Flughafen Hannover.'})
        self.assertEqual(read_top_pages([hit],q,fetch=fetch),[])
    def test_reader_keeps_relevant_multilingual_body(self):
        q='suche nach dem flughafen leipzig im internet'
        hit={'url':'https://example.org/leipzig','title':'Leipzig/Halle Airport'}
        fetch=Mock(return_value={'url':hit['url'],'text':'Leipzig/Halle Airport serves the region.'})
        out=read_top_pages([hit],q,fetch=fetch)
        self.assertEqual(len(out),1)
        self.assertTrue(out[0]['page_read'])
    def test_live_uses_next_source_when_first_body_is_wrong(self):
        q='suche nach dem flughafen leipzig im internet'
        hits=[{'url':'https://example.org/'+str(i),'title':'Flughafen Leipzig'} for i in range(4)]
        engine=Mock(); engine.search.return_value={'results':hits}
        def reader(batch,query):
            return [{**h,'page_read':True,'page_text':('Leipzig/Halle Airport serves the region.' if h==hits[3] else 'Flughafen Hannover ist geöffnet.')} for h in batch]
        out=search_live_sources(engine,q,reader=reader)
        self.assertEqual([s['url'] for s in out['sources']],[hits[3]['url']])
    def test_wrong_first_provider_triggers_other_provider_for_plain_query(self):
        from unittest.mock import MagicMock, patch
        q='Flughafen Leipzig'
        wrong={'url':'https://example.org/hannover','title':'Flughafen Hannover'}
        good={'url':'https://example.org/leipzig','title':'Leipzig/Halle Airport'}
        engine=WebSearch(opener=MagicMock())
        response=engine.opener.open.return_value.__enter__.return_value
        response.status=200; response.headers={'Content-Type':'text/html, application/rss+xml'}
        response.read.return_value=b'<html></html>'
        with patch('neral_web_search.parse_search_results',return_value=[wrong]), patch('neral_web_search.parse_bing_rss',return_value=[good]):
            self.assertEqual(engine.search(q)['results'],[good])
    def test_live_recovers_when_provider_pages_cannot_be_read(self):
        from unittest.mock import MagicMock, patch
        q='Flughafen Leipzig'
        first={'url':'https://example.org/first','title':'Flughafen Leipzig'}
        second={'url':'https://example.org/second','title':'Flughafen Leipzig'}
        engine=WebSearch(opener=MagicMock())
        with patch.object(engine,'search',return_value={'provider':'DuckDuckGo HTML','results':[first]}), patch.object(engine,'search_bing',return_value=[second]), patch.object(engine,'search_wikipedia',return_value=[]):
            def reader(batch,query):
                return [{**h,'page_read':True,'page_text':('Flughafen Leipzig ist ein Verkehrsflughafen.' if h==second else 'Flughafen Hannover ist geöffnet.')} for h in batch]
            out=search_live_sources(engine,q,reader=reader)
        self.assertEqual([x['url'] for x in out['sources']],[second['url']])
    def test_metadata_gate_is_not_only_fresh_news(self):
        q='suche nach dem flughafen leipzig im internet'
        wrong={'url':'https://example.org/hannover','title':'Flughafen Hannover'}
        good={'url':'https://example.org/leipzig','title':'Leipzig/Halle Airport'}
        self.assertEqual(live_topic_hits([wrong,good],q),[good])

if __name__=='__main__': unittest.main()
