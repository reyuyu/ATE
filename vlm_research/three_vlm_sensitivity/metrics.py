import math
import torch


def metrics(full, off):
    full, off = full.float(), off.float()
    assert full.ndim == off.ndim == 1 and full.shape == off.shape
    assert torch.isfinite(full).all() and torch.isfinite(off).all()
    target = int(full.argmax())
    p, q = full.log_softmax(-1), off.log_softmax(-1)
    mixture = torch.logaddexp(p, q) - math.log(2)
    js = float(.5 * ((p.exp() * (p - mixture)).sum() + (q.exp() * (q - mixture)).sum()))
    loss = float(p[target] - q[target])
    top2 = full.topk(2).values
    assert math.isfinite(js) and js >= -1e-6
    return dict(full_token_id=target, off_token_id=int(off.argmax()), full_top1_logp=float(p[target]),
        off_full_top1_logp=float(q[target]), signed_loss=loss, absolute_change=abs(loss),
        top1_flip=int(target != int(off.argmax())), js_divergence=js,
        full_top1_probability=float(p[target].exp()), full_top1_margin=float(top2[0] - top2[1]))
