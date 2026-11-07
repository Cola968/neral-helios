"""CPU contract for NERAL exact low-rank delta effective-update audit."""
import unittest
import numpy as np

from scripts.neral_lora_drift_audit import (
    matrix_stats, inventory_pairs, group_name, budget_diagnostics,
    V5_SHA, V61_SHA, expected_frozen,
)


class LowRankAuditTests(unittest.TestCase):
    def test_effective_weight_update_uses_b_times_a_not_factor_differences(self):
        A0=np.array([[1.,2.,-1.],[2.,1.,3.]])
        B0=np.array([[1.,2.],[3.,4.],[5.,6.]])
        A1=np.array([[0.9,2.2,-1.],[2.,1.1,3.]])
        B1=np.array([[1.2,2.1],[2.7,4.5],[5.1,6.]])
        observed=matrix_stats(A0,B0,A1,B1)
        effective0=B0@A0
        effective1=B1@A1
        self.assertAlmostEqual(observed["old_norm_sq"],float(np.sum(effective0**2)),places=6)
        self.assertAlmostEqual(observed["candidate_norm_sq"],float(np.sum(effective1**2)),places=6)
        self.assertAlmostEqual(observed["difference_norm_sq"],float(np.sum((effective1-effective0)**2)),places=6)
        self.assertAlmostEqual(observed["cross_inner"],float(np.sum(effective0*effective1)),places=6)

    def test_exactly_equal_matrices_have_no_functional_drift(self):
        A=np.array([[1.,2.],[0.,1.]])
        B=np.array([[2.,3.],[1.,0.]])
        report=matrix_stats(A,B,A.copy(),B.copy())
        self.assertAlmostEqual(report["difference_norm_sq"],0.,places=8)

    def test_lora_factor_changes_can_cancel_exactly(self):
        A=np.array([[1.,2.],[2.,-1.]])
        B=np.array([[3.,1.],[1.,5.]])
        # A'=2*A, B'=B/2: distinct LoRA tensors, identical *effective* update.
        report=matrix_stats(A,B,A*2,B/2)
        self.assertAlmostEqual(report["difference_norm_sq"],0.,places=8)

    def test_directed_sign_change_changes_real_effect(self):
        A=np.array([[1.,2.]])
        B=np.array([[3.],[4.]])
        r=matrix_stats(A,B,A,-B)
        self.assertAlmostEqual(r["difference_norm_sq"],4*r["old_norm_sq"],places=8)

    def test_invalid_nan_shapes_refused(self):
        A=np.array([[1.,2.]])
        B=np.array([[3.],[4.]])
        for candA,candB in ((np.array([[1.,2.,3.]]),B),
                            (np.array([[float("nan"),1.]]),B),
                            (A,np.array([[1.,2.]]))):
            with self.assertRaises(ValueError):
                matrix_stats(A,B,candA,candB)

    def test_lora_inventory_requires_complete_A_B_pairs(self):
        keys={"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
              "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"}
        pairs=inventory_pairs(keys,expected_pairs=1)
        self.assertEqual(len(pairs),1)
        with self.assertRaises(ValueError):
            inventory_pairs(keys-{"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"},expected_pairs=1)
        with self.assertRaises(ValueError):
            inventory_pairs(keys|{"irrelevant.weight"},expected_pairs=1)
        with self.assertRaises(ValueError):
            inventory_pairs(keys,expected_pairs=196)

    def test_tensors_grouped_by_real_modules_not_order(self):
        self.assertEqual(group_name("base_model.model.model.layers.1.mlp.up_proj"),"mlp")
        self.assertEqual(group_name("base_model.model.model.layers.27.self_attn.q_proj"),"attention")
        with self.assertRaises(ValueError):
            group_name("unsafe.layer")

    def test_replay_training_budget_does_not_cover_entire_dataset(self):
        d=budget_diagnostics(train_rows=800,steps=120,accumulation=4,batch_size=1)
        self.assertEqual(d["samples_processed"],480)
        self.assertAlmostEqual(d["nominal_epochs"],.6)
        self.assertFalse(d["full_epoch_completed"])
        self.assertEqual(d["samples_unseen_if_single_pass"],320)

    def test_predecessor_identity_immutable_and_release_always_false(self):
        self.assertEqual(len(V5_SHA),64)
        self.assertEqual(len(V61_SHA),64)
        self.assertNotEqual(V5_SHA,V61_SHA)
        self.assertTrue(expected_frozen(V5_SHA,V61_SHA))
        with self.assertRaises(ValueError):
            expected_frozen(V61_SHA,V5_SHA)
        with self.assertRaises(ValueError):
            expected_frozen(V5_SHA,V5_SHA)


if __name__=="__main__":
    unittest.main()
