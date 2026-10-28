"""Safety and data-split contracts for one unattended bounded FP32 SFT job."""
import unittest
from frontier350.arctus350_unattended_curriculum import (
    permit_run, bounded_quotas, schedule_unique_rows, candidate_gate, stop_conditions
)
class UnattendedContracts(unittest.TestCase):
    def test_strict_time_and_gpu_admission(self):
        self.assertTrue(permit_run(True,10300,10100,61,"2026-10-28T16:00:00+00:00",3600,0.65))
        self.assertFalse(permit_run(False,10300,10100,61,"2026-10-28T16:00:00+00:00",3600,.65))
        self.assertFalse(permit_run(True,8000,10100,61,"2026-10-28T16:00:00+00:00",3600,.65))
        self.assertFalse(permit_run(True,10300,10100,80,"2026-10-28T16:00:00+00:00",3600,.65))
        self.assertFalse(permit_run(True,10300,10100,61,"2026-10-29T00:00:00+01:00",3600,.65))
        self.assertFalse(permit_run(True,10300,10100,61,"2026-10-28T16:00:00+00:00",20000,.65))
        self.assertFalse(permit_run(True,10300,10100,61,"2026-10-28T16:00:00+00:00",3600,.80))
        with self.assertRaises(ValueError):permit_run(True,10300,10100,float("nan"),"2026-10-28T16:00:00+00:00",3600,.65)

    def test_quotas_bounded_and_all_real_train_only(self):
        quotas=bounded_quotas()
        self.assertEqual(sum(quotas.values()),3630)
        self.assertEqual(quotas["exact"],260)
        self.assertEqual(quotas["small_math"],260)
        self.assertEqual(quotas["chat"],800)
        self.assertLessEqual(sum(quotas.values()),4096)

    def test_deterministic_unique_interleaved_curriculum(self):
        quotas={"exact":3,"small_math":2,"chat":2}
        groups={
            "exact":[("exact%d"%i,"e") for i in range(5)],
            "small_math":[("math%d"%i,"m") for i in range(4)],
            "chat":[("chat%d"%i,"c") for i in range(4)]}
        a=schedule_unique_rows(groups,quotas)
        b=schedule_unique_rows({k:list(reversed(v)) for k,v in groups.items()},quotas)
        self.assertEqual(a,b)
        self.assertEqual(len(a),7)
        self.assertEqual(len({r[0] for r in a}),7)
        self.assertEqual([r[0][:4] for r in a[:3]],["exac","math","chat"])
        groups["chat"]=[("exact0","c"),("chat0","c"),("chat1","c")]
        with self.assertRaises(ValueError):schedule_unique_rows(groups,quotas)
        groups["chat"]=[("chat0","c")]
        with self.assertRaises(ValueError):schedule_unique_rows(groups,quotas)

    def test_candidate_is_never_released_based_on_nll_only(self):
        self.assertFalse(candidate_gate(3.,1.,3.,3.40,2.0,2.0))
        self.assertFalse(candidate_gate(3.,1.,3.,3.0,2.0,2.35))
        self.assertFalse(candidate_gate(3.,2.99,3.,3.0,2.0,2.0))
        self.assertTrue(candidate_gate(3.,2.0,3.,3.05,2.0,2.1))

    def test_deadline_and_temperature_are_hard_stops(self):
        self.assertEqual(stop_conditions(60,4000,66,3000,False),None)
        self.assertEqual(stop_conditions(4001,4000,66,3000,False),"deadline")
        self.assertEqual(stop_conditions(60,4000,80,3000,False),"temperature")
        self.assertEqual(stop_conditions(60,4000,None,3000,False),"missing_gpu_telemetry")
        self.assertEqual(stop_conditions(60,4000,60,1800,False),"cuda_vram")
        self.assertEqual(stop_conditions(60,4000,60,3000,True),"user_gpu_busy")
if __name__=="__main__":
    unittest.main()
