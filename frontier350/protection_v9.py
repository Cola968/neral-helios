"""First-order replay constraints; capability gates remain mandatory."""
import torch


def project_descent(vector, references, max_passes=12, tolerance=1e-6):
    """Project an update direction onto g_ref.dot(direction) >= 0 halfspaces.

    Alternating projections may converge slowly for conflicting references. If
    the bounded solver cannot satisfy every constraint, return a zero direction.
    Never advertise a first-order training constraint as a capability guarantee.
    """
    if not torch.isfinite(vector).all():
        raise FloatingPointError('nonfinite proposed direction')
    out=vector.clone(); refs=[]
    for ref in references:
        if ref.shape != out.shape or not torch.isfinite(ref).all():
            raise ValueError('invalid reference gradient')
        norm=ref.norm()
        if float(norm)>0:refs.append(ref/norm)
    corrections=0
    for _ in range(max_passes):
        for ref in refs:
            dot=torch.dot(out,ref)
            if float(dot)<0:
                out.add_(ref,alpha=-float(dot));corrections+=1
        scale=max(float(out.norm()),1e-12)
        if all(float(torch.dot(out,ref))>=-tolerance*scale for ref in refs):
            return out,{'corrections':corrections,'zero_fallback':False}
    out.zero_()
    return out,{'corrections':corrections,'zero_fallback':True}


def bound_step(descent, before, relative_limit=0.0005):
    if relative_limit<=0:raise ValueError('positive trust radius required')
    norm=descent.norm();limit=before.norm()*relative_limit
    factor=min(1.,float(limit)/(float(norm)+1e-12))
    return descent*factor,factor


def configure_trainable(model, last_blocks=4):
    if not 0<last_blocks<=len(model.blocks):raise ValueError('invalid trainable block count')
    for parameter in model.parameters():parameter.requires_grad_(False)
    for block in model.blocks[-last_blocks:]:
        block.float()
        for parameter in block.parameters():parameter.requires_grad_(True)
    return [(name,p) for name,p in model.named_parameters() if p.requires_grad]
