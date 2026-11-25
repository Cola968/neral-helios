"""CPU-only boot contract: reproduce the live missing-profile state."""
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app
from neral_three_tier_runtime import LUNA, HELIOS, RAW_9B, ThreeTierRuntime
from neral_v43_verified_start import build_verified_app, require_web_profile


class VerifiedStartTests(unittest.TestCase):
    def test_isolated_port_validates_models_and_keeps_separate_cookie(self):
        legacy,luna,helios=self.backends()
        runtime,app=build_verified_app(legacy,luna,helios,port=9692)
        self.assertIsNone(runtime.legacy_backend)
        self.assertEqual(app.config["SESSION_COOKIE_NAME"],"neral-v43-9692")
        self.assertEqual(require_web_profile(app,9692)["quartz"],RAW_9B)
        with self.assertRaises(RuntimeError):
            require_web_profile(app)

    def backends(self, model=RAW_9B, adapter=None):
        return (
            OllamaLocalBackend("qwen3:8b"),
            Mock(model=LUNA, adapter_sha256=None, loaded=True),
            Mock(model=model, adapter_sha256=adapter, loaded=True),
        )

    def test_reject_observed_unconfigured_legacy_runtime(self):
        legacy, luna, helios = self.backends(HELIOS, "a" * 64)
        runtime = ThreeTierRuntime(luna, helios, legacy_backend=legacy,
                                   execute_python=Mock(), v43_profile=False)
        app = create_preview_app(legacy, three_tier_runtime=runtime)
        with self.assertRaisesRegex(RuntimeError, "Webprofil fehlt"):
            require_web_profile(app)

    def test_explicit_web_profile_is_visible_in_real_app(self):
        legacy, luna, helios = self.backends()
        runtime, app = build_verified_app(legacy, luna, helios, execute_python=Mock())
        self.assertTrue(runtime.v43_profile)
        status = require_web_profile(app)
        self.assertTrue(status["web_search"]["configured"])
        self.assertFalse(status["adapter_loaded"])
        helios.respond_chat.assert_not_called()

    def test_public_news_reaches_web_provider_with_fresh_session(self):
        legacy, luna, helios = self.backends()
        runtime, app = build_verified_app(legacy, luna, helios, execute_python=Mock())
        runtime._verified_source = Mock(return_value=None)
        helios.respond_chat.return_value = "BER: Testmeldung."
        url = "https://ber.berlin-airport.de/de.html"
        hit = {"url": url, "title": "BER Test", "published_at":datetime.now(timezone.utc).isoformat()}
        page = {**hit, "page_read": True, "page_text": "BER Testmeldung zur aktuellen Entwicklung."}
        with patch("neral_web_search.WebSearch.search", return_value={"provider":"fixture","results":[hit]}) as search:
            with patch("neral_page_reader.read_top_pages", return_value=[page]):
                with patch.object(legacy, "unload", return_value=None):
                    with app.test_client() as client:
                        base = "http://127.0.0.1:9690"
                        client.get("/", base_url=base)
                        receipt = client.get("/api/neral/preview/session", base_url=base).json
                        response = client.post(
                            "/api/neral/chat/three-tier", base_url=base,
                            headers={"X-Neral-CSRF": receipt["csrf"]},
                            json={"tier":"Quartz","prompt":"Welche aktuellen Nachrichten gibt es zum Flughafen BER?","history":[],"force_sandbox":False},
                        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.json["web_search_status"], "ok")
        self.assertTrue(response.json["web_sources"][0]["page_read"])
        search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
