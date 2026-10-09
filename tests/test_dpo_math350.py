from __future__ import annotations
import unittest
from frontier350.dpo_math350 import dpo_loss_value,dpo_grad_coeff
class DPOMathTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        for adv in (-20,-2,-.1,0,.1,2,20):
            for beta in (.05,.1,.5,1.0):
                h=1e-6
                fd=(dpo_loss_value(adv+h,beta)-dpo_loss_value(adv-h,beta))/(2*h)
                self.assertAlmostEqual(fd,dpo_grad_coeff(adv,beta),places=7)
    def test_signs(self):
        self.assertLess(dpo_grad_coeff(0,.1),0)
        self.assertGreater(-dpo_grad_coeff(0,.1),0)
if __name__=="__main__":unittest.main()
