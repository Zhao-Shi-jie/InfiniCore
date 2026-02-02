import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner

# Test cases format: (input1_shape, input2_shape, target_shape, input1_strides_or_None, input2_strides_or_None, target_strides_or_None, margin_or_None)
# infinicore.nn.functional.cosine_embedding_loss(x1, x2, y, margin=0.0, reduction='mean')

_TEST_CASES_DATA = [
    ((4, 3), (4, 3), (4,), None, None, None, None),
    ((8, 5), (8, 5), (8,), (40, 5), None, None, 0.5),
    ((1, 10), (1, 10), (1,), None, None, None, 0.2),
    ((16, 20), (16, 20), (16,), None, None, None, 0.0),
    ((3, 7), (3, 7), (3,), None, (21, 7), None, None),
    ((2, 2), (2, 2), (2,), None, None, None, 1.0),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for s1, s2, st, st1, st2, stt, margin in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a = TensorSpec.from_tensor(s1, st1, dtype)
            b = TensorSpec.from_tensor(s2, st2, dtype)
            y = TensorSpec.from_tensor(st, stt, dtype)

            kwargs = {}
            if margin is not None:
                kwargs["margin"] = margin

            test_cases.append(
                TestCase(
                    inputs=[a, b, y],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="cosine_embedding_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """cosine_embedding_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("cosine_embedding_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def prepare_pytorch_inputs_and_kwargs(self, test_case, device):
        """Override to properly initialize target tensor with 1 or -1 values."""
        inputs, kwargs = super().prepare_pytorch_inputs_and_kwargs(test_case, device)

        if len(inputs) >= 3 and isinstance(inputs[2], torch.Tensor):
            target = inputs[2]
            target_shape = target.shape
            target_dtype = target.dtype
            target_device = target.device

            new_target = torch.randint(0, 2, target_shape, device=target_device).to(
                target_dtype
            )
            new_target = new_target * 2 - 1

            inputs[2] = new_target

        return inputs, kwargs

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.cosine_embedding_loss(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore implementation"""
        # return infinicore.nn.functional.cosine_embedding_loss(*args, **kwargs)
        return infinicore.cosine_embedding_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
