import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def argmin(input: Tensor, dim: int = None, keepdim: bool = False) -> Tensor:
    r"""Return the indices of the minimum values along a given dimension.

    Args:
        input: Input tensor
        dim: The dimension to reduce (can be negative)
        keepdim: Whether to keep the reduced dimension (default: False)

    Returns:
        Output tensor containing the indices of minimum values
    """

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.argmin(input, dim, keepdim)
    else:
        if dim is None:
            # If dim is None, we flatten the input and perform argmin on the flattened tensor
            flattened = input.contiguous().view([input.numel()])
            scalar_index = Tensor(_infinicore.argmin(flattened._underlying, 0, False))
            if keepdim:
                output_shape = [1] * input.ndim
                return scalar_index.reshape(output_shape)
            else:
                return scalar_index
        else:
            return Tensor(_infinicore.argmin(input._underlying, dim, keepdim))
