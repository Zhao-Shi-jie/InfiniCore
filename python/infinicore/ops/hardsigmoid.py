import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hardsigmoid(
    input: Tensor,
) -> Tensor:
    r"""Apply the hardsigmoid activation function element-wise.

    hardsigmoid(x) = max(0, min(1, (x + 3) / 6))

    Args:
        input: Input tensor
        inplace: If True, performs the operation in-place (default: False)

    Returns:
        Output tensor with hardsigmoid applied
    """

    # 使用 ntops 实现（如果可用）
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.hardsigmoid(input, inplace=False)
    else:
        return Tensor(_infinicore.hardsigmoid(input._underlying))
