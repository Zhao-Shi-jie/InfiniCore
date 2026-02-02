import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
    out: Tensor = None,
) -> Tensor:
    r"""Performs cosine embedding loss lookup."""

    if infinicore.use_ntops and input1.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.cosine_embedding_loss(
            input1, input2, target, margin, reduction
        )

    if out is None:
        return Tensor(
            _infinicore.cosine_embedding_loss(
                input1._underlying,
                input2._underlying,
                target._underlying,
                margin,
                reduction,
            )
        )
    _infinicore.cosine_embedding_loss_(
        out._underlying,
        input1._underlying,
        input2._underlying,
        target._underlying,
        margin,
        reduction,
    )
    return out
