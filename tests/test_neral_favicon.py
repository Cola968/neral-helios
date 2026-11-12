"""Verified-site favicons: actual icon bytes, same-origin delivery, no SSRF."""
import unittest
from unittest.mock import patch
from flask import Flask
from neral_web_api import register_web_search_api
from neral_web_search import fetch_public_favicon

DNS_PUBLIC=[(2,1,6,'',('93.184.215.14',443))]
DNS_PRIVATE=[(2,1,6,'',('127.0.0.1',443))]
PNG=b"\x89PNG\r\n\x1a\n"+b"\x00"*32
ICO=b"\x00\x00\x01\x00"+b"\x00"*28

class FaviconSecurityTests(unittest.TestCase):
    def setUp(self):
        app=Flask(__name__);register_web_search_api(app,service=object())
        self.client=app.test_client()
    def test_invalid_host_never_uses_dns(self):
        with patch('neral_web_search.socket.getaddrinfo') as dns:
            for host in ('localhost','127.0.0.1','example.org:8080','example.org/fav',
                         'example.org@localhost','foo.internal','example..org',
                         'example.org\nX-Header:evil','[::1]'):
                self.assertIsNone(fetch_public_favicon(host),host)
            dns.assert_not_called()
    def test_private_dns_never_calls_network(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PRIVATE):
            with patch('neral_web_search._request_favicon') as fetch:
                self.assertIsNone(fetch_public_favicon('example.org'))
                fetch.assert_not_called()
    def test_public_icon_is_pinned_to_resolved_public_address(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            with patch('neral_web_search._request_favicon',return_value=(PNG,'image/png')) as fetch:
                self.assertEqual(fetch_public_favicon('example.org'),(PNG,'image/png'))
                self.assertEqual(fetch.call_args.args,('example.org','93.184.215.14','/favicon.ico'))
    def test_only_small_real_image_formats_accepted(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            with patch('neral_web_search._request_favicon',return_value=(b'<svg onload=alert(1)>','image/svg+xml')):
                self.assertIsNone(fetch_public_favicon('example.org'))
            with patch('neral_web_search._request_favicon',return_value=(b'<html>','image/png')):
                self.assertIsNone(fetch_public_favicon('example.org'))
            with patch('neral_web_search._request_favicon',return_value=(PNG,'image/png')):
                self.assertEqual(fetch_public_favicon('example.org'),(PNG,'image/png'))
            with patch('neral_web_search._request_favicon',return_value=(ICO,'image/x-icon')):
                self.assertEqual(fetch_public_favicon('example.org'),(ICO,'image/x-icon'))
    def test_route_delivers_image_with_no_tracking_redirects(self):
        with patch('neral_web_api.fetch_public_favicon',return_value=(PNG,'image/png')) as fetch:
            response=self.client.get('/api/web/favicon?host=github.com')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.data,PNG)
            self.assertEqual(response.mimetype,'image/png')
            self.assertEqual(response.headers.get('X-Content-Type-Options'),'nosniff')
            fetch.assert_called_once_with('github.com')
        with patch('neral_web_api.fetch_public_favicon',return_value=None):
            self.assertEqual(self.client.get('/api/web/favicon?host=unavailable.org').status_code,404)
    def test_route_does_not_accept_unrelated_url_parameters(self):
        with patch('neral_web_api.fetch_public_favicon') as fetch:
            self.assertEqual(self.client.get('/api/web/favicon?url=https://localhost/favicon.ico').status_code,400)
            self.assertEqual(self.client.get('/api/web/favicon?host=example.org&path=/admin').status_code,400)
            fetch.assert_not_called()


    def test_missing_ico_uses_fixed_png_path_with_same_pinned_ip(self):
        def receive(host,address,path="/favicon.ico"):
            if path=="/favicon.ico":
                return b"", ""
            return PNG,"image/png"
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            with patch('neral_web_search._request_favicon',side_effect=receive) as fetch:
                self.assertEqual(fetch_public_favicon('example.org'),(PNG,'image/png'))
                self.assertEqual([call.args[2] for call in fetch.call_args_list],
                    ["/favicon.ico","/favicon-32x32.png"])
                self.assertTrue(all(call.args[1]=='93.184.215.14' for call in fetch.call_args_list))
    def test_html_disguised_as_ico_does_not_prevent_fixed_png_fallback(self):
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            with patch('neral_web_search._request_favicon',
                       side_effect=[(b"<html>blocked</html>","text/html"),(PNG,"image/png")]):
                self.assertEqual(fetch_public_favicon('example.org'),(PNG,'image/png'))


    def test_safe_public_svg_icon_is_served_without_scripting(self):
        vector=(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                b'<path d="M1 1L20 20" fill="#111"/></svg>')
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            def site_icon(host,address,path="/favicon.ico"):
                return (vector,"image/svg+xml") if path=="/images/favicon.svg" else (b"","")
            with patch('neral_web_search._request_favicon',side_effect=site_icon) as get:
                icon=fetch_public_favicon("example.org")
                self.assertIsNotNone(icon)
                self.assertEqual(icon[1],"image/svg+xml")
                from xml.etree import ElementTree
                self.assertIsNotNone(ElementTree.fromstring(icon[0]).find("{http://www.w3.org/2000/svg}path"))
                self.assertTrue(all(call.args[1]=="93.184.215.14" for call in get.call_args_list))
                self.assertIn("/images/favicon.svg",[c.args[2] for c in get.call_args_list])

    def test_svg_icons_reject_scripts_events_external_resources_and_entities(self):
        bad=[
          b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>',
          b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
          b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://tracker.example/pixel"/></svg>',
          b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg xmlns="http://www.w3.org/2000/svg">&x;</svg>',
          b'<svg xmlns="http://www.w3.org/2000/svg"><path fill="url(https://attacker.example/x)" d="M0 0"/></svg>',
        ]
        with patch('neral_web_search.socket.getaddrinfo',return_value=DNS_PUBLIC):
            for vector in bad:
                with self.subTest(vector=vector[:70]),patch('neral_web_search._request_favicon',return_value=(vector,'image/svg+xml')):
                    self.assertIsNone(fetch_public_favicon('example.org'))


    def test_figma_svg_is_reduced_to_safe_static_paths_without_html_or_filters(self):
        from neral_web_search import sanitize_favicon_svg
        raw=(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             b'<defs><filter id="blur"><feGaussianBlur stdDeviation="3"/></filter></defs>'
             b'<foreignObject width="24" height="24">'
             b'<div xmlns="http://www.w3.org/1999/xhtml">blur effect</div>'
             b'</foreignObject><path d="M2 2L20 20" style="fill:#fff" fill="#ffffff"/>'
             b'</svg>')
        sanitized=sanitize_favicon_svg(raw)
        self.assertIsNotNone(sanitized)
        self.assertIn(b"path",sanitized)
        self.assertNotIn(b"foreignObject",sanitized)
        self.assertNotIn(b"filter",sanitized)
        self.assertNotIn(b"style=",sanitized)

if __name__=='__main__':unittest.main()
