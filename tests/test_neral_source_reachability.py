"""Outbound link verification never contacts private services or trusts search hits."""
import unittest
from unittest.mock import patch
from flask import Flask
from neral_web_api import register_web_search_api
from neral_web_search import verify_public_https_url

PUBLIC_IP=[(2,1,6,'',('93.184.215.14',443))]
PRIVATE_IP=[(2,1,6,'',('127.0.0.1',443))]

class SourceReachability(unittest.TestCase):
    def test_rejects_private_dns_before_network(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=PRIVATE_IP),patch('neral_web_search._probe_https') as probe:
            self.assertIsNone(verify_public_https_url('https://example.org/article'))
            probe.assert_not_called()
    def test_rejects_unsafe_network_urls(self):
        with patch('neral_web_search.socket.getaddrinfo') as dns:
            for url in ('http://example.org', 'https://127.0.0.1/','https://localhost/',
                        'https://example.org:8080/', 'https://me:secret@example.org/'):
                self.assertIsNone(verify_public_https_url(url))
            dns.assert_not_called()
    def test_successful_public_response_keeps_real_url(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=PUBLIC_IP),patch('neral_web_search._probe_https',return_value=(200,None)) as probe:
            self.assertEqual(verify_public_https_url('https://example.org/article'), 'https://example.org/article')
            self.assertTrue(probe.called)
    def test_unreachable_source_is_not_promoted(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=PUBLIC_IP),patch('neral_web_search._probe_https',return_value=(404,None)):
            self.assertIsNone(verify_public_https_url('https://example.org/not-found'))
    def test_redirect_to_localhost_fails_closed(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=PUBLIC_IP),patch('neral_web_search._probe_https',return_value=(302,'https://127.0.0.1/private')):
            self.assertIsNone(verify_public_https_url('https://example.org/go'))
    def test_explicit_checked_search_marks_only_live_links(self):
        class FakeSearch:
            def search(self, query):
                return {'query':query,'results':[
                    {'title':'Good','url':'https://example.org/a','snippet':'about BER'},
                    {'title':'Dead','url':'https://example.org/404','snippet':'about BER'}]}
        app=Flask(__name__);register_web_search_api(app,FakeSearch())
        def checker(url): return url if url.endswith('/a') else None
        with patch('neral_web_api.verify_public_https_url',side_effect=checker):
            result=app.test_client().post('/api/web/search',json={'query':'BER','verify_links':True})
        self.assertEqual(result.status_code,200)
        content=result.get_json();self.assertEqual(len(content['results']),1)
        self.assertTrue(content['results'][0]['reachable'])
        self.assertEqual(content['results'][0]['url'],'https://example.org/a')
    def test_default_search_compatibility_without_verification(self):
        class FakeSearch:
            def search(self,query):return {'results':[{'title':'Raw','url':'https://example.org'}]}
        app=Flask(__name__);register_web_search_api(app,FakeSearch())
        out=app.test_client().post('/api/web/search',json={'query':'Test'})
        self.assertEqual(out.status_code,200)
        self.assertNotIn('reachable',out.get_json()['results'][0])

if __name__=='__main__':unittest.main()
