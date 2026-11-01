"""Flask integration for opt-in local Qwen plan drafting; all model calls faked."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("HELIOS_PREPARATION_MODE","1")
import app as helios


class AgentPlanAPITests(unittest.TestCase):
    def setUp(self):
        self.previous=(helios.ASSISTANT_ENABLED,helios.PREPARATION_MODE)
        self.addCleanup(self.restore)
        helios.ASSISTANT_ENABLED=True
        helios.PREPARATION_MODE=False
        helios._hits.clear()
        self.client=helios.app.test_client()
        self.assertEqual(self.client.get("/").status_code,200)

    def restore(self):
        helios.ASSISTANT_ENABLED,helios.PREPARATION_MODE=self.previous

    def test_session_and_double_opt_in_before_model_call(self):
        endpoint="/api/assistant/agent-plan"
        with patch.object(helios,"propose_agent_plan") as model:
            self.assertEqual(helios.app.test_client().post(
                endpoint,json={"goal":"Recherchiere BER.","confirm":True}).status_code,401)
            self.assertEqual(self.client.post(
                endpoint,json={"goal":"Recherchiere BER.","confirm":False}).status_code,400)
            self.assertEqual(self.client.post(
                endpoint,json={"goal":"Recherchiere BER."}).status_code,400)
            model.assert_not_called()

    def test_explicit_plan_only_returns_unapproved_result(self):
        endpoint="/api/assistant/agent-plan"
        candidate={"schema":1,"status":"PROPOSED_UNAPPROVED",
                   "source":"external-model","base_model":"qwen3.5:9b",
                   "execution_enabled":False,"requires_human_approval":True,
                   "steps":[{"type":"research","instruction":"Search public sources."}]}
        with patch.object(helios,"propose_agent_plan",return_value=candidate) as model:
            response=self.client.post(endpoint,json={
                "goal":"Recherchiere öffentliche Luftverkehrszahlen am BER.",
                "confirm":True})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json["status"],"PROPOSED_UNAPPROVED")
            self.assertFalse(response.json["execution_enabled"])
            model.assert_called_once_with(
                "Recherchiere öffentliche Luftverkehrszahlen am BER.",
                model=helios.ASSISTANT_MODEL)

    def test_opt_out_and_preparation_mode_hard_block(self):
        endpoint="/api/assistant/agent-plan"
        with patch.object(helios,"propose_agent_plan") as model:
            helios.ASSISTANT_ENABLED=False
            self.assertEqual(self.client.post(endpoint,json={
                "goal":"Recherchiere öffentliches Thema BER.","confirm":True}).status_code,503)
            helios.ASSISTANT_ENABLED=True;helios.PREPARATION_MODE=True
            self.assertEqual(self.client.post(endpoint,json={
                "goal":"Recherchiere öffentliches Thema BER.","confirm":True}).status_code,503)
            model.assert_not_called()


if __name__=="__main__":
    unittest.main()
