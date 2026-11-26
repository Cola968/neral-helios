"""Public official news index: only bounded, dated same-origin discoveries."""
import unittest
from datetime import datetime,timezone,timedelta
from email.utils import format_datetime
from http.client import BadStatusLine, IncompleteRead
from unittest.mock import MagicMock, Mock, patch
from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch
from neral_official_news import parse_official_news_index, discover_official_news

BASE="https://corporate.berlin-airport.de/de/unternehmen-presse/presseportal/pressemitteilungen.html"
CLOCK=datetime(2026,11,25,tzinfo=timezone.utc)

class OfficialNewsTests(unittest.TestCase):
    def test_colloquial_request_routes_to_web(self):
        self.assertTrue(route_live_query("Was gibt es Neues am Flughafen MUC?"))

    def test_index_only_discovers_recent_same_origin_dated_articles(self):
        html='<a href="/presse/2026-11-23-news.html">23. November 2026 Neue Verbindungen am BER</a><a href="/presse/2026-10-10-old.html">10. Oktober 2026 BER alt</a><a href="https://other.example.com/news">24. November 2026 BER fremd</a><a href="/undatiert">BER</a>'
        rows=parse_official_news_index(html,BASE,now=CLOCK)
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]["url"].endswith("2026-11-23-news.html"))
        self.assertEqual(rows[0]["published_at"],"2026-11-23T00:00:00+00:00")

    def test_conflicting_dates_and_future_entries_rejected(self):
        html='<a href="/2026-10-01-old.html">24. November 2026 BER</a><a href="/2026-11-30-future.html">30. November 2026 BER</a>'
        self.assertEqual(parse_official_news_index(html,BASE,now=CLOCK),[])

    def test_mixed_official_index_keeps_topic_candidates_before_the_cap(self):
        index = "https://press.example.org/releases/index.html"
        other = "".join(f'<a href="/press/2026-11-24-rail-{i}.html">Rail station renovation {i}</a>'
                        for i in range(12))
        wanted = "/press/2026-11-21-ocean.html"
        markup = other + f'<a href="{wanted}">Coral reefs survey results</a>'
        self.assertEqual(len(parse_official_news_index(markup, index, now=CLOCK)), 10)
        relevant = parse_official_news_index(markup, index, now=CLOCK,
                                            query="Latest news about coral reefs")
        self.assertEqual([row["url"] for row in relevant], ["https://press.example.org" + wanted])

    def test_live_reader_reaches_topical_article_beyond_unrelated_index_cap(self):
        now = datetime.now(timezone.utc)
        index = "https://press.example.org/releases/index.html"
        unrelated_day = (now - timedelta(days=1)).date().isoformat()
        topical_day = (now - timedelta(days=2)).date().isoformat()
        unrelated = "".join(f'<a href="/press/{unrelated_day}-rail-{i}.html">Zugverkehr wird ausgebaut {i}</a>'
                            for i in range(12))
        path = f"/press/{topical_day}-airport.html"
        title = "Flughafen BER meldet neue Regionalverbindungen"
        candidate_url = "https://press.example.org" + path
        pages = {
            index: {"url": index, "markup": unrelated + f'<a href="{path}">{title}</a>'},
            candidate_url: {"url": candidate_url, "markup": (
                f"<article><h1>{title}</h1><p>Der Flughafen Berlin Brandenburg erweitert den Regionalflugplan. "
                "Die beteiligten Fluggesellschaften haben Details zu den geplanten Verbindungen veröffentlicht.</p>"
                "<p>Das neue Angebot berücksichtigt Nachfrage und Umsteigemöglichkeiten. Reisende können die "
                "Abflugzeiten in den veröffentlichten Unterlagen einsehen und ihre Anreise entsprechend planen.</p></article>"
            )},
        }
        fetch = Mock(side_effect=lambda url, *args, **kwargs: pages.get(url))
        engine = WebSearch()
        query = "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"BER": index}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {}), \
                patch("neral_page_reader.fetch_and_extract", fetch), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]):
            result = search_live_sources(engine, query, include_diagnostics=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["url"] for row in result["sources"]], [candidate_url])
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [index, candidate_url])
        self.assertEqual(result["stage_counts"]["official_candidates"], 1)

    def test_structured_card_requires_explicit_recent_date_and_unique_topic_permalink(self):
        index = "https://press.example.org/releases/index.html"
        query = "Welche aktuellen Nachrichten gibt es zum Flughafen MUC?"
        title = "Flughafen München ändert Terminalzugang"
        observed = '<time itemprop="datePublished" datetime="2026-11-24">24.11.2026</time>'
        def link(path, headline=title, marked=True):
            prop = ' itemprop="url"' if marked else ""
            return f'<a{prop} href="{path}"><span itemprop="headline">{headline}</span></a>'
        def card(parts, tag="div", kind="Article"):
            return f'<{tag} itemscope itemtype="https://schema.org/{kind}">{parts}</{tag}>'
        markup = "".join([
            card(link("/secondary") + observed + link("/another", "Weitere Links", marked=False)),
            card(link("/only-modified") + observed.replace("datePublished", "dateModified")),
            card(link("/unmarked", marked=False) + observed),
            card(link("/unrelated", "Terminalroute der Kreisverwaltung") + observed),
            card(link("https://other.example.org/releases/cross-host") + observed),
            card(link("/queried?latest=1") + observed),
            card(link("/newsletter.pdf") + observed),
            card(link("https://[invalid/article") + observed),
            card(link("/unrelated-scope") + observed, kind="Event"),
            card(link("/nested-scope") + '<span itemscope itemtype="https://schema.org/Event">' + observed + '</span>'),
            card(link("/stale", "24. November 2026 " + title) + observed.replace("2026-11-24", "2026-11-02")),
            card(link("/future") + observed.replace("2026-11-24", "2026-11-30")),
            card(link("/invalid-date", "24. November 2026 " + title) + observed.replace("2026-11-24", "2026-11-31")),
            card(link("/inferred-date") + '<time itemprop="datePublished">24. November 2026</time>'),
            card(link("/invalid-offset") + observed.replace("2026-11-24", "2026-11-24T12:00:00+00:99")),
            card(link("/2026-11-23-path-conflict") + observed),
            card(link("/duplicate-conflict") + observed + observed.replace("2026-11-24", "2026-11-23")),
            card(link("/valid") + '<div>' + observed + '</div>' + link("/valid", marked=False)),
            card(link("/valid-with-time") + '<meta itemprop="datePublished" content="2026-11-23T10:00:00Z">',
                 tag="article", kind="NewsArticle"),
            card(link("/legacy", "23. November 2026 " + title)),
            '<li>24. November 2026 ' + card(link("/stale-nested") + observed.replace("2026-11-24", "2026-11-02")) + '</li>',
            '<li>22. November 2026 ' + card(link("/legacy-nested")) + '</li>',
        ])
        rows = parse_official_news_index(markup, index, query=query, now=CLOCK)
        self.assertEqual([(row["url"], row["published_at"]) for row in rows], [
            ("https://press.example.org/valid", "2026-11-24T00:00:00+00:00"),
            ("https://press.example.org/valid-with-time", "2026-11-23T10:00:00+00:00"),
            ("https://press.example.org/legacy", "2026-11-23T00:00:00+00:00"),
            ("https://press.example.org/legacy-nested", "2026-11-22T00:00:00+00:00"),
        ])
        self.assertTrue(all(row["date_origin"] == "official_index" for row in rows))

    def test_official_html_diagnostics_distinguish_link_date_freshness_and_topic(self):
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(days=1)).date().isoformat()
        old = (now - timedelta(days=20)).date().isoformat()
        host = "https://press.example.org"
        index = host + "/releases/index.html"
        title = "Munich Airport confirms new flight schedule"
        off_topic = "Municipal library opens new study area"
        headline = {
            "url": "https://news.google.com/rss/articles/opaque-fixture",
            "title": title, "published_at": (now - timedelta(days=1)).isoformat(),
        }
        markup = (
            f'<a href="https://foreign.example.org/{fresh}-airport">{title}</a>'
            f'<a href="/assets/{fresh}-notice.pdf">{title}</a>'
            f'<a href="/news/{fresh}-query?id=1">{title}</a>'
            '<a href="/releases/undated">Munich Airport modernizes terminal</a>'
            f'<a href="/releases/{old}-schedule">Munich Airport publishes old schedule</a>'
            f'<a href="/releases/{fresh}-library">{off_topic}</a>'
        )
        fetch = Mock(return_value={"url": index, "markup": markup})
        engine = WebSearch()
        query = "Was gibt es Neues am Flughafen MUC?"
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"MUC": index}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {}), \
                patch("neral_page_reader.fetch_and_extract", fetch), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[headline]):
            normal = search_live_sources(engine, query)
            measured = search_live_sources(engine, query, include_diagnostics=True)

        counters = measured.pop("stage_counts")
        self.assertEqual(normal, measured)
        self.assertEqual(measured["status"], "headlines_only")
        self.assertEqual(measured["context"], "")
        self.assertEqual([row["url"] for row in measured["sources"]], [headline["url"]])
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [index, index])
        self.assertEqual({key: counters[key] for key in (
            "official_html_bodies", "official_html_link_rows", "official_html_dated_rows",
            "official_html_fresh_rows", "official_html_topic_rows", "official_candidates",
            "reader_candidates",
        )}, {
            "official_html_bodies": 1, "official_html_link_rows": 3, "official_html_dated_rows": 2,
            "official_html_fresh_rows": 1, "official_html_topic_rows": 0, "official_candidates": 0,
            "reader_candidates": 0,
        })
        self.assertNotIn(off_topic, str(counters))
        self.assertNotIn(host, str(counters))

    def test_live_reader_reaches_article_with_separately_marked_publication(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(days=1)).replace(microsecond=0).isoformat()
        index = "https://press.example.org/releases/index.html"
        candidate = "https://press.example.org/releases/regional-routes"
        title = "Flughafen BER veröffentlicht neue Regionalverbindungen"
        card = (
            '<div itemscope itemtype="https://schema.org/Article">'
            f'<div><a itemprop="url" href="/releases/regional-routes"><span itemprop="headline">{title}</span></a></div>'
            f'<time itemprop="datePublished" datetime="{published}">Gestern</time>'
            '</div>'
        )
        article = (
            f"<article><h1>{title}</h1><p>Der Flughafen Berlin Brandenburg erweitert den Regionalflugplan. "
            "Die beteiligten Fluggesellschaften haben Details zu den geplanten Verbindungen veröffentlicht.</p>"
            "<p>Das neue Angebot berücksichtigt Nachfrage und Umsteigemöglichkeiten. Reisende können die "
            "Abflugzeiten in den veröffentlichten Unterlagen einsehen und ihre Anreise entsprechend planen.</p></article>"
        )
        fetch = Mock(side_effect={
            index: {"url": index, "markup": card},
            candidate: {"url": candidate, "markup": article},
        }.get)
        # Other discovery providers return no candidates; use the real bounded
        # live search and article extraction against local markup.
        engine = WebSearch()
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"BER": index}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {}), \
                patch("neral_page_reader.fetch_and_extract", side_effect=lambda url, *args, **kwargs: fetch(url)), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]):
            result = search_live_sources(engine, "Welche aktuellen Nachrichten gibt es zum Flughafen BER?",
                                         include_diagnostics=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["url"] for row in result["sources"]], [candidate])
        self.assertEqual(result["sources"][0]["published_at"], published)
        self.assertIn(f"Veröffentlichung laut offizieller Übersicht: {published}", result["context"])
        self.assertEqual(result["sources"][0]["date_origin"], "official_index")
        self.assertNotIn("Feed-Veröffentlichung", result["context"])
        self.assertEqual([result["stage_counts"][key] for key in (
            "official_html_link_rows", "official_html_dated_rows",
            "official_html_fresh_rows", "official_html_topic_rows", "official_candidates",
        )], [1, 1, 1, 1, 1])
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [index, candidate])

    def test_transport_failure_in_optional_provider_cannot_discard_official_article(self):
        now = datetime.now(timezone.utc)
        day = (now - timedelta(days=1)).date().isoformat()
        index = "https://press.example.org/releases/index.html"
        path = f"/press/{day}-regional-routes.html"
        title = "Flughafen BER veröffentlicht neue Regionalverbindungen"
        article = "https://press.example.org" + path
        pages = {
            index: {"url": index, "markup": f'<a href="{path}">{title}</a>'},
            article: {"url": article, "markup": (
                f"<article><h1>{title}</h1><p>Der Flughafen Berlin Brandenburg erweitert den Regionalflugplan. "
                "Die beteiligten Fluggesellschaften haben Details zu den geplanten Verbindungen veröffentlicht.</p>"
                "<p>Das neue Angebot berücksichtigt Nachfrage und Umsteigemöglichkeiten. Reisende können die "
                "Abflugzeiten in den veröffentlichten Unterlagen einsehen und ihre Anreise entsprechend planen.</p></article>"
            )},
        }
        opaque = (f'<rss><channel><item><title>{title}</title>'
                  '<link>https://news.google.com/rss/articles/transport-fixture</link>'
                  f'<pubDate>{format_datetime(now)}</pubDate>'
                  '<source url="https://publisher.example.org/">Fixture</source>'
                  '</item></channel></rss>').encode()
        for failed_stage in ("all_feeds", "publisher_lookup", "independent_index"):
            with self.subTest(failed_stage=failed_stage):
                failed = []
                def open_provider(request, timeout):
                    url = request.full_url
                    response = MagicMock(status=200)
                    response.__enter__.return_value = response
                    if url.startswith("https://api.gdeltproject.org/"):
                        response.headers = {"Content-Type": "application/json"}
                        response.read.return_value = b'{"articles":[]}'
                        if failed_stage == "independent_index":
                            failed.append("independent_index")
                            response.read.side_effect = IncompleteRead(b'{"articles":[]}', 7)
                    else:
                        response.headers = {"Content-Type": "application/rss+xml"}
                        response.read.return_value = b"<rss><channel/></rss>"
                        if failed_stage == "all_feeds":
                            failed.append("feed")
                            response.read.side_effect = IncompleteRead(b"<rss>", 20)
                        elif failed_stage == "publisher_lookup":
                            if url.startswith("https://news.google.com/"):
                                response.read.return_value = opaque
                            elif url.startswith("https://www.bing.com/search?"):
                                failed.append("publisher_lookup")
                                raise BadStatusLine("local transport sentinel")
                    return response
                engine = WebSearch()
                engine.opener = Mock(open=Mock(side_effect=open_provider))
                fetch = Mock(side_effect=lambda url, *args, **kwargs: pages.get(url))
                query = "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
                with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"BER": index}), \
                        patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {}), \
                        patch("neral_page_reader.fetch_and_extract", fetch), \
                        patch("neral_public_news._LAST", None):
                    result = search_live_sources(engine, query, include_diagnostics=True)
                self.assertTrue(failed)
                if failed_stage == "all_feeds":
                    self.assertGreaterEqual(len(failed), 2)
                self.assertEqual(result["status"], "ok")
                self.assertEqual([row["url"] for row in result["sources"]], [article])
                official_fetches = [call.args[0] for call in fetch.call_args_list
                                    if call.args[0].startswith("https://press.example.org/")]
                self.assertEqual(official_fetches, [index, article])
                self.assertLessEqual(engine.opener.open.call_count, 6)
                self.assertEqual(result["stage_counts"]["official_candidates"], 1)
                self.assertNotIn("local transport sentinel", str(result))

    def test_authority_atom_uses_same_host_https_and_explicit_publication(self):
        feed = "https://press.example.org/feeds/civic.xml"
        stamp = "2026-11-22T12:00:00+00:00"
        def entry(href, title, date_tag="published", date=stamp):
            return (f'<entry><title>{title}</title><link href="{href}"/>'
                    f'<{date_tag}>{date}</{date_tag}></entry>')
        markup = '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join([
            entry("http://press.example.org/news/terminal", "Flughafen München ändert Terminalzugang"),
            entry("http://other.example.org/news/external", "Flughafen München fremd"),
            entry("/news/undated", "Flughafen München undatiert", "updated"),
            entry("/news/past", "Flughafen München alt", date="2026-11-10T12:00:00+00:00"),
            entry("/news/street", "Die Stadt erneuert eine Straße"),
        ]) + '</feed>'
        fetch = Mock(return_value={"url": feed, "markup": markup})
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}, create=True):
            rows = discover_official_news("Was gibt es Neues am Flughafen MUC?", fetch=fetch, now=CLOCK)
        self.assertEqual([row["url"] for row in rows], ["https://press.example.org/news/terminal"])
        self.assertEqual(rows[0]["published_at"], stamp)
        self.assertNotIn("page_text", rows[0])
        fetch.assert_called_once_with(feed)

    def test_distinct_official_feed_remains_in_bounded_live_read_budget(self):
        now = datetime.now(timezone.utc)
        index = "https://operator.example.org/releases/index.html"
        feed = "https://authority.example.org/feeds/transport.xml"
        older = now - timedelta(days=2)
        html_day = (now - timedelta(days=1)).date().isoformat()
        paths = [f"/press/{html_day}-flight-{i}.html" for i in range(10)]
        html = "".join(f'<a href="{path}">Flughafen München meldet Update {i}</a>'
                       for i, path in enumerate(paths))
        candidate = "https://authority.example.org/press/terminal-routes.html"
        title = "Flughafen München veröffentlicht neue Regionalverbindungen"
        atom = (f'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>{title}</title>'
                f'<link href="{candidate}"/><published>{older.isoformat()}</published></entry></feed>')
        article = (
            f"<article><h1>{title}</h1><p>Der Flughafen München erweitert den Regionalflugplan. "
            "Die beteiligten Fluggesellschaften haben Details zu den geplanten Verbindungen veröffentlicht.</p>"
            "<p>Das neue Angebot berücksichtigt Nachfrage und Umsteigemöglichkeiten. Reisende können die "
            "Abflugzeiten in den veröffentlichten Unterlagen einsehen und ihre Anreise entsprechend planen.</p></article>"
        )
        pages = {
            index: {"url": index, "markup": html},
            feed: {"url": feed, "markup": atom},
            candidate: {"url": candidate, "markup": article},
        }
        fetch = Mock(side_effect=lambda url, *args, **kwargs: pages.get(url))
        engine = WebSearch()
        query = "Welche aktuellen Nachrichten gibt es zum Flughafen MUC?"
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"MUC": index}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}), \
                patch("neral_page_reader.fetch_and_extract", fetch), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]):
            result = search_live_sources(engine, query, include_diagnostics=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["url"] for row in result["sources"]], [candidate])
        self.assertEqual(result["sources"][0]["published_at"], older.isoformat())
        self.assertEqual(result["sources"][0]["date_origin"], "official_feed")
        self.assertIn(f"Veröffentlichung laut offiziellem Feed: {older.isoformat()}", result["context"])
        self.assertEqual(result["stage_counts"]["official_candidates"], 10)
        self.assertEqual(result["stage_counts"]["reader_candidates"], 10)
        attempted = [call.args[0] for call in fetch.call_args_list]
        self.assertEqual(attempted[:2], [index, feed])
        self.assertEqual(attempted.count(candidate), 1)
        self.assertEqual(len(attempted), len(set(attempted)))
        self.assertLessEqual(len(attempted), 12)

    def test_official_rescue_uses_remaining_read_budget_only_after_no_original_is_readable(self):
        now = datetime.now(timezone.utc)
        stamp = (now - timedelta(days=1)).isoformat()
        day = (now - timedelta(days=1)).date().isoformat()
        host = "https://press.example.org"
        index = f"{host}/releases/index.html"
        rescue = f"{host}/press/{day}-public-regional-routes.html"
        title = "Flughafen BER veröffentlicht neue Regionalverbindungen"
        article = (
            f"<article><h1>{title}</h1><p>Der Flughafen Berlin Brandenburg erweitert den Regionalflugplan. "
            "Die beteiligten Fluggesellschaften haben Details zu den geplanten Verbindungen veröffentlicht.</p>"
            "<p>Das neue Angebot berücksichtigt Nachfrage und Umsteigemöglichkeiten. Reisende können die "
            "Abflugzeiten in den veröffentlichten Unterlagen einsehen und ihre Anreise entsprechend planen.</p></article>"
        )
        opaque = {
            "url": "https://news.google.com/rss/articles/unavailable-fixture",
            "publisher_url": "https://another.example.org/",
            "title": title, "published_at": stamp,
        }
        query = "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
        for direct_count, readable, expected_rescue, initial_official in (
            (3, False, True, False), (3, True, False, False),
            (10, False, False, False), (2, False, False, True),
        ):
            with self.subTest(direct_count=direct_count, readable=readable):
                originals = [f"{host}/press/{day}-original-{i}.html" for i in range(direct_count)]
                direct = [{"url": url, "title": title, "published_at": stamp} for url in originals]
                index_html = f'<a href="{originals[0]}">{title}</a><a href="{rescue}">{title}</a>'
                def fetch_url(url, *args, **kwargs):
                    if url == index:
                        return {"url": url, "markup": index_html}
                    if (url == rescue and expected_rescue) or (readable and url in originals):
                        return {"url": url, "markup": article}
                    if url in originals or (url == rescue and initial_official):
                        return {"url": url, "markup": "<p>Sign in to continue.</p>"}
                    raise AssertionError("unexpected request: " + url)
                fetch = Mock(side_effect=fetch_url)
                engine = WebSearch()
                with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"BER": index}), \
                        patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {}), \
                        patch("neral_page_reader.fetch_and_extract", fetch), \
                        patch("neral_public_news.discover_public_news", return_value=[]) as gdelt, \
                        patch.object(engine, "search_bing", return_value=[]) as targeted, \
                        patch.object(engine, "search_bing_news", return_value=direct), \
                        patch.object(engine, "search_news", return_value=[] if initial_official else [opaque]):
                    result = search_live_sources(engine, query, include_diagnostics=True)
                attempted = [call.args[0] for call in fetch.call_args_list]
                self.assertEqual(len(attempted), len(set(attempted)))
                official_attempted = expected_rescue or initial_official
                self.assertEqual(set(attempted), set(originals + ([index, rescue] if official_attempted else [])))
                self.assertEqual(result["stage_counts"]["reader_candidates"],
                                 direct_count + int(official_attempted))
                gdelt.assert_called_once()
                targeted.assert_not_called()
                self.assertEqual(engine.status()["last_status"], result["status"])
                self.assertEqual(engine.status()["last_error"], result["error"])
                if expected_rescue:
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual([row["url"] for row in result["sources"]], [rescue])
                    self.assertEqual(result["sources"][0]["date_origin"], "official_index")
                    self.assertIn("Veröffentlichung laut offizieller Übersicht:", result["context"])
                    self.assertIn("Official press index", result["provider"])
                    self.assertIn("Official press index", engine.status()["last_provider"])
                elif readable:
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(len(result["sources"]), 3)
                    self.assertEqual({row["date_origin"] for row in result["sources"]}, {"feed"})
                    self.assertIn(f"Veröffentlichung laut Nachrichtenfeed: {stamp}", result["context"])
                else:
                    self.assertEqual(result["status"], "headlines_only")
                    self.assertEqual(result["context"], "")

    def test_duplicate_official_html_and_atom_link_do_not_consume_extra_slot(self):
        index = "https://authority.example.org/releases/index.html"
        feed = "https://authority.example.org/feeds/transport.xml"
        common = "/press/2026-11-24-shared.html"
        only_html = "/press/2026-11-25-index-only.html"
        only_atom = "/press/2026-11-23-feed-only.html"
        html = (f'<a href="{only_html}">Flughafen München meldet mehr Verbindungen</a>'
                f'<a href="{common}">Flughafen München meldet neue Abfertigung</a>')
        atom = ('<feed xmlns="http://www.w3.org/2005/Atom">'
                f'<entry><title>Flughafen München meldet neue Abfertigung</title><link href="{common}"/>'
                '<published>2026-11-24T12:00:00+00:00</published></entry>'
                f'<entry><title>Flughafen München plant neue Routen</title><link href="{only_atom}"/>'
                '<published>2026-11-23T12:00:00+00:00</published></entry></feed>')
        fetch = Mock(side_effect=[{"url": index, "markup": html}, {"url": feed, "markup": atom}])
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"MUC": index}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}):
            rows = discover_official_news("Aktuelle Nachrichten Flughafen MUC", fetch=fetch, now=CLOCK)
        self.assertEqual([row["url"] for row in rows],
                         ["https://authority.example.org" + path for path in (only_html, only_atom, common)])
        self.assertEqual(rows[2]["date_origin"], "official_index")
        self.assertEqual(rows[2]["published_at"], "2026-11-24T00:00:00+00:00")
        self.assertEqual(fetch.call_count, 2)

    def test_probe_separates_official_fetch_failures_from_received_feed(self):
        engine = WebSearch()
        feed = "https://press.example.org/feeds/civic.xml"
        fetch = Mock(side_effect=[None, {"url": feed, "markup": '<feed xmlns="http://www.w3.org/2005/Atom"/>'}])
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"MUC": "https://press.example.org/"}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}), \
                patch("neral_page_reader.fetch_and_extract", fetch), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]):
            result = search_live_sources(engine, "Was gibt es Neues am Flughafen MUC?", include_diagnostics=True)
        self.assertEqual(result["status"], "empty")
        stages = result["stage_counts"]
        self.assertEqual([stages[k] for k in (
            "official_html_attempts", "official_html_bodies", "official_atom_attempts",
            "official_atom_bodies", "official_candidates")], [1, 0, 1, 1, 0])
        self.assertEqual(fetch.call_count, 2)

    def test_probe_classifies_public_fetch_failures_without_extra_requests(self):
        engine = WebSearch()
        feed = "https://press.example.org/feeds/civic.xml"
        address = Mock(side_effect=[OSError("private name sentinel"), "93.184.215.14"])
        connection = Mock()
        connection.request.side_effect = TimeoutError("private connection sentinel")
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {"MUC": "https://press.example.org/"}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}), \
                patch("neral_page_reader._public_address", address), \
                patch("neral_page_reader._PinnedConnection", return_value=connection), \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]):
            result = search_live_sources(engine, "Was gibt es Neues am Flughafen MUC?", include_diagnostics=True)
        stages = result["stage_counts"]
        self.assertEqual([stages[k] for k in (
            "official_fetch_address", "official_fetch_timeout", "official_fetch_http",
            "official_fetch_network", "official_fetch_policy_body")], [1, 1, 0, 0, 0])
        self.assertNotIn("sentinel", str(stages))
        self.assertEqual(address.call_count, 2)
        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_unknown_entity_does_not_contact_any_official_index(self):
        fetch=Mock()
        self.assertEqual(discover_official_news("Aktuelle Nachrichten zur NASA",fetch=fetch),[])
        fetch.assert_not_called()

    def test_discovery_does_not_treat_index_as_article(self):
        fetch=Mock(return_value={"url":BASE,"markup":'<a href="/presse/2026-11-23-news.html">23. November 2026 Neue Verbindungen am BER</a>'})
        rows=discover_official_news("Aktuelle Nachrichten am Flughafen BER",fetch=fetch,now=CLOCK)
        self.assertEqual(len(rows),1)
        self.assertNotEqual(rows[0]["url"],BASE)
        self.assertNotIn("page_text",rows[0])

if __name__=="__main__":
    unittest.main()
