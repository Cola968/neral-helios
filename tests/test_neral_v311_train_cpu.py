import math
import unittest
from scripts.neral_qlora_verified_v39 import validate_smoke_proof

class MultiEpochGate(unittest.TestCase):
    def test_2_epochs_requires_exact_26_optimizer_updates(self):
        m={'grad_norm':3.4,'training_loss':1.1,'global_step':26,'peak_allocated_gib':8.1,
           'peak_reserved_gib':9.3,'nvidia_peak_mib':10140,
           'adapter_before':'one','adapter_after':'two'}
        self.assertTrue(validate_smoke_proof(m,expected_steps=26))
        with self.assertRaises(ValueError):validate_smoke_proof(m,expected_steps=1)
        with self.assertRaises(ValueError):validate_smoke_proof({**m,'global_step':25},expected_steps=26)
        with self.assertRaises(ValueError):validate_smoke_proof({**m,'grad_norm':float('nan')},expected_steps=26)
        with self.assertRaises(ValueError):validate_smoke_proof({**m,'nvidia_peak_mib':11264},expected_steps=26)

    def test_epoch_limit_and_optimizer_configuration_are_explicit(self):
        from scripts.neral_qlora_verified_v39 import training_schedule
        self.assertEqual(training_schedule(97,2,1024),{'epochs':2,'expected_steps':26,'max_context':1024,'optimizer':'paged_adamw_8bit','rank':16})
        with self.assertRaises(ValueError):training_schedule(97,4,1024)
        with self.assertRaises(ValueError):training_schedule(97,2,2048)

if __name__=='__main__':unittest.main()
