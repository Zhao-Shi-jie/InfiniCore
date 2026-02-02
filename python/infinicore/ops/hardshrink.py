import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hardshrink(
    input: Tensor,
    lambd: float = 0.5,
    *,
    out=None,
) -> Tensor:
    r"""Applies the hardshrink function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.hardshrink(input, lambd)

    if out is None:
        return Tensor(_infinicore.hardshrink(input._underlying, lambd))
    _infinicore.hardshrink_(out._underlying, input._underlying, lambd)
    return out
