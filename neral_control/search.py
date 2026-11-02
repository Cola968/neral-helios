"""Opt-in free Wikipedia retrieval. All returned content is untrusted data.

No query occurs until an explicitly enabled instance's search() is called. Only
fixed English/German Wikipedia HTTPS API endpoints are fetched. The default
stdlib transport retains environment proxy policy and never follows redirects.
Cancellation is cooperative before/between socket operations; Python urllib
cannot hard-cancel an in-progress DNS lookup or blocking socket operation.
Socket timeouts bound I/O waits, not a guaranteed wall-clock interruption.

The injectable transport contract is (url, *, timeout, max_bytes) -> bytes;
custom transports must honor those bounds and the no-redirect requirement.
Deadlines are absolute time.monotonic() values. published_at, when supplied,
is the source's latest revision timestamp, not its original publication date.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
import urllib.request
from urllib.parse import urlencode, urlsplit


class SearchUnavailable(RuntimeError):
    """Internal retrieval failure, reported as a structured search result."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _https_get(url, *, timeout, max_bytes):
    # Default ProxyHandler is intentional: keep the session's network policy.
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, headers={
        'User-Agent': 'NeralControl/1.0 (opt-in Wikipedia reference retrieval)',
        'Accept': 'application/json', 'Accept-Encoding': 'identity',
    })
    end = time.monotonic() + timeout
    with opener.open(request, timeout=timeout) as response:
        if response.geturl() != url or response.status != 200:
            raise SearchUnavailable('Unexpected HTTP response or redirect')
        size = response.headers.get('Content-Length')
        if size is not None and int(size) > max_bytes:
            raise SearchUnavailable('Response exceeds byte limit')
        chunks, size = [], 0
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise SearchUnavailable('Request deadline exceeded')
            # HTTPResponse.read1 uses at most one underlying raw read. Updating
            # this socket's timeout keeps each wait within the remaining budget.
            sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
            if sock is not None:
                sock.settimeout(remaining)
            chunk = response.read1(min(16384, max_bytes + 1 - size))
            if not chunk:
                return b''.join(chunks)
            size += len(chunk)
            if size > max_bytes:
                raise SearchUnavailable('Response exceeds byte limit')
            chunks.append(chunk)


class WikipediaSearch:
    """Replaceable provider returning retrieved extracts, never instructions."""

    provider = 'wikipedia'

    def __init__(self, *, enabled=False, language='en', transport=None,
                 timeout=5.0, max_response_bytes=262144):
        if language not in ('en', 'de'):
            raise ValueError('Supported Wikipedia languages: en, de')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 10:
            raise ValueError('timeout must be positive and at most 10 seconds')
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 1048576:
            raise ValueError('max_response_bytes must be between 1 and 1048576')
        self.enabled = enabled is True
        self.host = {'en': 'en.wikipedia.org', 'de': 'de.wikipedia.org'}[language]
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes
        self.transport = transport or _https_get

    def _validate_url(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.netloc != self.host or
                parsed.hostname != self.host or parsed.username or parsed.password or
                parsed.fragment or not parsed.path.startswith('/')):
            raise SearchUnavailable('Unapproved Wikipedia source URL')
        return url

    @staticmethod
    def _check(cancel_event, deadline):
        if cancel_event is not None and cancel_event.is_set():
            raise SearchUnavailable('Search cancelled')
        if time.monotonic() >= deadline:
            raise SearchUnavailable('Search deadline exceeded')

    def _request(self, params, cancel_event, deadline):
        self._check(cancel_event, deadline)
        url = self._validate_url('https://' + self.host + '/w/api.php?' + urlencode({
            'action': 'query', 'format': 'json', 'formatversion': '2', **params}))
        data = self.transport(url, timeout=min(self.timeout, deadline-time.monotonic()),
                              max_bytes=self.max_response_bytes)
        self._check(cancel_event, deadline)
        if not isinstance(data, bytes) or len(data) > self.max_response_bytes:
            raise SearchUnavailable('Invalid or oversized response')
        parsed = json.loads(data)
        if not isinstance(parsed, dict) or 'error' in parsed or not isinstance(parsed.get('query'), dict):
            raise SearchUnavailable('Invalid Wikipedia API response')
        return parsed['query'], url

    def search(self, query, limit=3, cancel_event=None, deadline=None):
        result = {'provider': self.provider, 'query': query if isinstance(query, str) else '',
                  'available': False, 'results': []}
        try:
            if not self.enabled:
                raise SearchUnavailable('Search is disabled; explicit opt-in required')
            if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                raise SearchUnavailable('Query must contain 1 to 2000 characters')
            if type(limit) is not int or not 1 <= limit <= 5:
                raise SearchUnavailable('Search limit must be an integer from 1 to 5')
            if deadline is None:
                deadline = time.monotonic() + 15.0
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                raise SearchUnavailable('Invalid search deadline')
            listing, _ = self._request({'list': 'search', 'srsearch': query,
                                       'srlimit': limit}, cancel_event, deadline)
            hits = listing.get('search')
            if not isinstance(hits, list):
                raise SearchUnavailable('Invalid Wikipedia search results')
            errors = []
            for hit in hits[:limit]:
                self._check(cancel_event, deadline)
                try:
                    pageid = hit.get('pageid') if isinstance(hit, dict) else None
                    if type(pageid) is not int or pageid <= 0:
                        raise SearchUnavailable('Invalid Wikipedia page identifier')
                    article, fetched_via = self._request({
                        'pageids': pageid, 'prop': 'extracts|info|revisions',
                        'explaintext': 1, 'exintro': 1, 'exchars': 1200,
                        'inprop': 'url', 'rvprop': 'timestamp', 'rvlimit': 1,
                    }, cancel_event, deadline)
                    pages = article.get('pages')
                    if not isinstance(pages, list) or len(pages) != 1:
                        raise SearchUnavailable('Missing Wikipedia article')
                    page = pages[0]
                    if not isinstance(page, dict) or page.get('pageid') != pageid or 'missing' in page:
                        raise SearchUnavailable('Missing Wikipedia article')
                    title, extract, url = page.get('title'), page.get('extract'), page.get('fullurl')
                    if not all(isinstance(value, str) and value.strip() for value in (title, extract, url)):
                        raise SearchUnavailable('Wikipedia article has no usable extract')
                    self._validate_url(url)
                    source = {'title': title[:500], 'url': url,
                              'snippet': extract[:1200], 'untrusted': True,
                              'retrieved_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                              'fetched_via': fetched_via}
                    revisions = page.get('revisions')
                    if isinstance(revisions, list) and revisions and isinstance(revisions[0], dict):
                        stamp = revisions[0].get('timestamp')
                        if isinstance(stamp, str):
                            try:
                                datetime.fromisoformat(stamp.replace('Z', '+00:00'))
                            except ValueError:
                                pass
                            else:
                                source['published_at'] = stamp
                    result['results'].append(source)
                except (OSError, ValueError, TypeError, SearchUnavailable) as exc:
                    errors.append(str(exc)[:200])
            self._check(cancel_event, deadline)
            result['available'] = bool(result['results']) or not hits
            if errors:
                result['error'] = '; '.join(errors)[:600]
            return result
        except (OSError, ValueError, TypeError, SearchUnavailable) as exc:
            result['error'] = str(exc)[:600]
            return result
