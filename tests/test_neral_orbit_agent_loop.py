"""One-shot composed Orbit/Qwen process contract, no network in tests."""
import unittest
from neral_orbit_agent_loop import run_confirmed_research, AgentLoopError

PLAN={"status":"PROPOSED_UNAPPROVED","base_model":"qwen3.5:9b",
      "execution_enabled":False,"tools_executed":[],
      "steps":[{"type":"research","instruction":"Recherchiere öffentliche Flughafenfakten."},
               {"type":"review","instruction":"Prüfe den Quellenkontext."},
               {"type":"draft","instruction":"Erstelle einen Quellenbericht."}]}
REPORT={"provider":"Wikipedia (de)","source_count":1,"model_used":False,
        "requires_review":True,"sources":[{"url":"https://de.wikipedia.org/?curid=42",
        "title":"BER", "excerpt":"Öffentlicher enzyklopädischer Auszug."}]}


class FakeStore:
    def __init__(self):
        self.calls=[]
    def create(self,**kwargs):
        self.calls.append("create")
        return {"id":"a"*32,"revision":1,"goal":kwargs["goal"],"status":"DRAFT"}
    def transition(self,id,action,*,revision):
        self.calls.append(action)
        return {"id":id,"revision":2,"status":"APPROVED_NO_EXECUTOR"}
    def begin_research(self,id,*,revision):
        self.calls.append("begin")
        return {"id":id,"revision":3,"status":"RESEARCH_RUNNING"}
    def finish_research(self,id,*,revision,report=None,error=None):
        self.calls.append("finish")
        return {"id":id,"revision":4,"status":"REVIEW_READY","report":report}


class AgentLoopTests(unittest.TestCase):
    def test_no_consent_means_zero_model_or_network_calls(self):
        called=[]
        def f(*args,**kwargs):
            called.append(True)
        for model_consent,public_consent in [(False,True),(True,False),
                                             ("yes",True),(True,None)]:
            with self.subTest(a=model_consent,b=public_consent),self.assertRaises(AgentLoopError):
                run_confirmed_research(
                    "Flughafen Berlin Brandenburg",confirm_model=model_consent,
                    confirm_public=public_consent,plan=f,research=f,draft=f,store=FakeStore())
        self.assertEqual(called,[])

    def test_approved_pipeline_calls_one_public_research_and_preserves_review(self):
        store=FakeStore();called=[]
        def p(goal):
            called.append("plan")
            return PLAN
        def r(goal):
            called.append("research")
            return REPORT
        def d(sources):
            called.append("draft")
            self.assertEqual(len(sources),1)
            return "Draft generated, not verified."
        out=run_confirmed_research(
            "Flughafen Berlin Brandenburg",confirm_model=True,
            confirm_public=True,plan=p,research=r,draft=d,store=store)
        self.assertEqual(called,["plan","research","draft"])
        self.assertEqual(store.calls,["create","approve","begin","finish"])
        self.assertTrue(out["research_completed"])
        self.assertTrue(out["draft_generated"])
        self.assertTrue(out["requires_human_review"])
        self.assertEqual(out["status"],"REVIEW_READY")
        self.assertEqual(out["source_count"],1)
        self.assertNotIn("Draft generated",str(out))
        self.assertNotIn("Öffentlicher enzyklopädischer Auszug",str(out))
        self.assertFalse(out["auto_published"])
        self.assertFalse(out["model_weights_modified"])

    def test_model_cannot_add_unapproved_execution_tools(self):
        called=[]
        def illegal(_):
            return {**PLAN,"execution_enabled":True}
        with self.assertRaises(AgentLoopError):
            run_confirmed_research(
                "Flughafen Berlin Brandenburg",confirm_model=True,
                confirm_public=True,plan=illegal,
                research=lambda _:called.append("network"),draft=lambda _:None,store=FakeStore())
        self.assertEqual(called,[])


if __name__=="__main__":
    unittest.main()
