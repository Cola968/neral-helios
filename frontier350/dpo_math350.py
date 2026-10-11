from __future__ import annotations
import math

def dpo_loss_value(advantage: float, beta: float) -> float:
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0,1]")
    x=beta*float(advantage)
    # softplus(-x), stable for both signs
    return max(-x,0.0)+math.log1p(math.exp(-abs(x)))

def dpo_grad_coeff(advantage: float, beta: float) -> float:
    """d(-log sigmoid(beta*adv))/d(adv), used as dL/dchosen; dL/drejected is its negative."""
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0,1]")
    x=beta*float(advantage)
    if x >= 0:
        e=math.exp(-x)
        sigmoid_neg=e/(1.0+e)
    else:
        e=math.exp(x)
        sigmoid_neg=1.0/(1.0+e)
    return -beta*sigmoid_neg
