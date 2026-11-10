"""Shared numerical training kernel; publication control belongs to callers."""
import torch


def execute(spec,graph,source,*,model=None,step=None,collector=None,check_slot_bounds=False,check_noise=False):
    # Lazy import avoids changing the dependency/order of existing production wrappers.
    from . import mechanism as base
    adapter=base.Adapter(spec,graph)
    if model is None:model=base.new_model()
    frames=[] if collector is not None else None
    for _ in range(100):
        batch=adapter.batch();signal,slots=base.center(model,batch)
        if check_slot_bounds and any(float(s.norm())>3. for s in slots):raise RuntimeError('slot bound violated')
        calls=source.calls
        if step is None:
            base.update(model.parameters(),signal,source);stats={}
        else:stats=step(model,signal,source)
        if check_noise and source.calls!=calls+1:raise RuntimeError('exactly one Gaussian vector required')
        if collector is not None:frames.append(collector(batch,slots,stats))
    values=tuple(float(v) for v in torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
    return values,frames,model
