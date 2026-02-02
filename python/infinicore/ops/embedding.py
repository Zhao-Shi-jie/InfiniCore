import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def embedding(
    input: Tensor,
    weight: Tensor,
    *,
    out=None,
) -> Tensor:
    r"""Performs embedding lookup."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.embedding(input, weight, out=out)

    if out is None:
        return Tensor(_infinicore.embedding(input._underlying, weight._underlying))
    _infinicore.embedding_(out._underlying, input._underlying, weight._underlying)
    return out
