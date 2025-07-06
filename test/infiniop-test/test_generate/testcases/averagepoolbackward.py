import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def averagepool_backward_1d(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """1D Average Pooling Forward + Backward using PyTorch"""
    # 设置requires_grad=True以便计算梯度
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    # 前向传播
    output = F.avg_pool1d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    
    # 创建与输出相同形状的梯度张量
    grad_output = torch.ones_like(output)
    
    # 反向传播
    output.backward(grad_output)
    
    return output, input_tensor.grad


def averagepool_backward_2d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """2D Average Pooling Forward + Backward using PyTorch"""
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    # 前向传播
    output = F.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    
    # 创建与输出相同形状的梯度张量
    grad_output = torch.ones_like(output)
    
    # 反向传播
    output.backward(grad_output)
    
    return output, input_tensor.grad


def averagepool_backward_3d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """3D Average Pooling Forward + Backward using PyTorch"""
    original_dtype = input_tensor.dtype

    # a transform as "RuntimeError: "avg_pool3d_out_frame" not implemented for 'Half'"
    if input_tensor.dtype in [torch.bfloat16, torch.float16]:
        input_tensor = input_tensor.float()

    input_tensor = input_tensor.clone().requires_grad_(True)
    
    # 前向传播
    output = F.avg_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # 创建与输出相同形状的梯度张量
    grad_output = torch.ones_like(output)
    # 反向传播
    output.backward(grad_output)

    # 梯度计算完成后，转换回原始类型
    if original_dtype in [torch.bfloat16, torch.float16]:
        output = output.to(original_dtype)
        grad_output = input_tensor.grad.to(original_dtype)
    else:
        grad_output = input_tensor.grad
    
    return output, grad_output


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class AveragePoolBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = None,
        padding: Union[int, Tuple] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
        pool_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("averagepool_backward")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.pool_dim = pool_dim
        
        # 随机生成输入张量
        self.input_tensor = random_tensor(input_size, dtype)
        
        # 计算前向输出和反向梯度
        if self.pool_dim == 1:
            self.forward_output, self.backward_output = averagepool_backward_1d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 2:
            self.forward_output, self.backward_output = averagepool_backward_2d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 3:
            self.forward_output, self.backward_output = averagepool_backward_3d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        else:
            raise ValueError(f"Unsupported pool dimension: {self.pool_dim}")

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 处理forward_output（反向传播的输入）
        if self.forward_output.dtype == torch.bfloat16:
            forward_output_numpy = self.forward_output.view(torch.uint16).detach().numpy()
            ggml_dtype_input = gguf.GGMLQuantizationType.BF16
        else:
            forward_output_numpy = self.forward_output.detach().numpy()
            ggml_dtype_input = np_dtype_to_ggml(forward_output_numpy.dtype)
        
        # Add forward_output tensor (input for backward pass)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            forward_output_numpy,
            raw_dtype=ggml_dtype_input,
        )
        
        # # Add input_size information (原始输入的尺寸)
        # test_writer.add_array(test_writer.gguf_key("input_size"), list(self.input_tensor.shape))
        
        # Add parameters
        if isinstance(self.kernel_size, int):
            test_writer.add_array(test_writer.gguf_key("kernel_size"), [self.kernel_size])
        else:
            test_writer.add_array(test_writer.gguf_key("kernel_size"), list(self.kernel_size))
            
        if self.stride is not None:
            if isinstance(self.stride, int):
                test_writer.add_array(test_writer.gguf_key("stride"), [self.stride])
            else:
                test_writer.add_array(test_writer.gguf_key("stride"), list(self.stride))
                
        if isinstance(self.padding, int):
            test_writer.add_array(test_writer.gguf_key("padding"), [self.padding])
        else:
            test_writer.add_array(test_writer.gguf_key("padding"), list(self.padding))
            
        test_writer.add_bool(test_writer.gguf_key("ceil_mode"), self.ceil_mode)
        test_writer.add_bool(test_writer.gguf_key("count_include_pad"), self.count_include_pad)
        
        if self.divisor_override is not None:
            test_writer.add_int32(test_writer.gguf_key("divisor_override"), self.divisor_override)
            
        # Add backward_output tensor (output of backward pass) - 使用float64精度
        backward_output_f64 = self.backward_output.double()
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            backward_output_f64.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("averagepool_backward.gguf")
    
    # Data types to test
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    test_cases = []
    
    # Generate comprehensive test cases for each data type and dimension
    for dtype in dtypes:
        
        # ============ 1D Average Pooling Backward Tests ============
        # Basic cases
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(4, 8, 128),
                kernel_size=3, stride=1, padding=0, pool_dim=1, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 16, 256),
                kernel_size=5, stride=2, padding=2, pool_dim=1, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(8, 4, 64),
                kernel_size=7, stride=3, padding=1, pool_dim=1, dtype=dtype,
            ),
        ])
        
        # ceil_mode variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 3, 99),
                kernel_size=4, stride=3, padding=1, ceil_mode=True, pool_dim=1, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(3, 2, 77),
                kernel_size=6, stride=4, padding=0, ceil_mode=True, pool_dim=1, dtype=dtype,
            ),
        ])
        
        # count_include_pad variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(2, 5, 48),
                kernel_size=5, stride=2, padding=2, count_include_pad=False, pool_dim=1, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 6, 32),
                kernel_size=3, stride=1, padding=1, count_include_pad=False, pool_dim=1, dtype=dtype,
            ),
        ])
        
        # ============ 2D Average Pooling Backward Tests ============
        # Basic cases with square kernels
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(2, 3, 64, 64),
                kernel_size=3, stride=1, padding=1, pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(4, 16, 128, 128),
                kernel_size=5, stride=2, padding=2, pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 8, 96, 96),
                kernel_size=7, stride=3, padding=0, pool_dim=2, dtype=dtype,
            ),
        ])
        
        # Rectangular kernels
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(2, 4, 80, 120),
                kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 6, 72, 48),
                kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(3, 2, 56, 84),
                kernel_size=(2, 4), stride=(2, 3), padding=(0, 2), pool_dim=2, dtype=dtype,
            ),
        ])
        
        # ceil_mode variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 33, 33),
                kernel_size=4, stride=3, padding=1, ceil_mode=True, pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 5, 77, 89),
                kernel_size=(5, 3), stride=(4, 2), padding=(2, 1), ceil_mode=True, pool_dim=2, dtype=dtype,
            ),
        ])
        
        # count_include_pad variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 3, 48, 48),
                kernel_size=5, stride=2, padding=2, count_include_pad=False, pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 1, 36, 52),
                kernel_size=(3, 7), stride=(1, 2), padding=(1, 3), count_include_pad=False, pool_dim=2, dtype=dtype,
            ),
        ])
        
        # divisor_override variations (only for 2D)
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 2, 32, 32),
                kernel_size=4, stride=2, padding=0, divisor_override=20, pool_dim=2, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 1, 24, 40),
                kernel_size=(2, 3), stride=(2, 2), padding=0, divisor_override=10, pool_dim=2, dtype=dtype,
            ),
        ])
        
        # ============ 3D Average Pooling Backward Tests ============
        # Basic cubic kernels
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 2, 32, 32, 32),
                kernel_size=3, stride=1, padding=1, pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 4, 48, 48, 48),
                kernel_size=5, stride=2, padding=2, pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 64, 64, 64),
                kernel_size=7, stride=3, padding=0, pool_dim=3, dtype=dtype,
            ),
        ])
        
        # Non-cubic kernels
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 3, 24, 36, 48),
                kernel_size=(2, 3, 4), stride=(1, 2, 2), padding=(0, 1, 2), pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 2, 40, 32, 56),
                kernel_size=(5, 3, 7), stride=(2, 1, 3), padding=(2, 1, 3), pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 28, 44, 36),
                kernel_size=(3, 5, 2), stride=(2, 3, 1), padding=(1, 2, 1), pool_dim=3, dtype=dtype,
            ),
        ])
        
        # ceil_mode variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 27, 27, 27),
                kernel_size=4, stride=3, padding=1, ceil_mode=True, pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(2, 2, 33, 45, 39),
                kernel_size=(5, 3, 4), stride=(3, 2, 3), padding=(2, 1, 1), ceil_mode=True, pool_dim=3, dtype=dtype,
            ),
        ])
        
        # count_include_pad variations
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 2, 24, 24, 24),
                kernel_size=5, stride=2, padding=2, count_include_pad=False, pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 30, 42, 36),
                kernel_size=(3, 7, 5), stride=(1, 2, 2), padding=(1, 3, 2), count_include_pad=False, pool_dim=3, dtype=dtype,
            ),
        ])
        
        # divisor_override variations (only for 3D)
        test_cases.extend([
            AveragePoolBackwardTestCase(
                input_size=(1, 1, 20, 20, 20),
                kernel_size=4, stride=2, padding=0, divisor_override=100, pool_dim=3, dtype=dtype,
            ),
            AveragePoolBackwardTestCase(
                input_size=(1, 2, 16, 24, 32),
                kernel_size=(2, 3, 4), stride=(2, 2, 2), padding=0, divisor_override=50, pool_dim=3, dtype=dtype,
            ),
        ])
    
    # Add some edge cases
    edge_cases = [
        # Very large kernels
        AveragePoolBackwardTestCase(
            input_size=(1, 2, 64),
            kernel_size=32, stride=16, padding=16, pool_dim=1, dtype=torch.float32,
        ),
        AveragePoolBackwardTestCase(
            input_size=(1, 1, 64, 64),
            kernel_size=32, stride=16, padding=16, pool_dim=2, dtype=torch.float16,
        ),
        # Kernel size equals input size
        AveragePoolBackwardTestCase(
            input_size=(1, 2, 16, 16),
            kernel_size=16, stride=1, padding=8, pool_dim=2, dtype=torch.bfloat16,
        ),
        # Large stride
        AveragePoolBackwardTestCase(
            input_size=(2, 3, 100, 100),
            kernel_size=5, stride=10, padding=2, pool_dim=2, dtype=torch.float32,
        ),
        # Complex 3D case
        AveragePoolBackwardTestCase(
            input_size=(1, 1, 16, 32, 48),
            kernel_size=(8, 4, 6), stride=(4, 8, 12), padding=(4, 2, 3), pool_dim=3, dtype=torch.float16,
        ),
    ]
    
    test_cases.extend(edge_cases)
    
    print(f"Generated {len(test_cases)} test cases")
    print(f"Data types: {len(dtypes)} types")
    print(f"Pool dimensions: 1D, 2D, 3D")
    
    test_writer.add_tests(test_cases)
    test_writer.save()