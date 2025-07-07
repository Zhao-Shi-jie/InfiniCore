import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def interpolate_nearest_1d(
    input_tensor: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """1D Interpolate Nearest using PyTorch"""
    return F.interpolate(
        input_tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def interpolate_nearest_2d(
    input_tensor: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """2D Interpolate Nearest using PyTorch"""
    return F.interpolate(
        input_tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def interpolate_nearest_3d(
    input_tensor: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float, float]]] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """3D Interpolate Nearest using PyTorch"""
    return F.interpolate(
        input_tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    if dtype == torch.int8:
        # For int8, generate random integers in the range [-128, 127]
        tensor = torch.randint(-128, 128, shape, dtype=dtype)
    else:
        tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class InterpolateNearestTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        size: Optional[Union[int, Tuple]] = None,
        scale_factor: Optional[Union[float, Tuple]] = None,
        mode: str = 'nearest',
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
        interp_dim: int = 2,  # 1, 2, or 3 (spatial dimensions)
    ):
        super().__init__("interpolate_nearest")
        self.input_tensor = input_tensor
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
        self.interp_dim = interp_dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Handle different data types for input
        if self.input_tensor.dtype == torch.bfloat16:
            input_numpy = self.input_tensor.view(torch.uint16).numpy()
            ggml_dtype = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input_tensor.numpy()
            ggml_dtype = np_dtype_to_ggml(input_numpy.dtype)
        
        # Add input tensor
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype,
        )
            
        # Compute expected output using PyTorch
        # For int8, we need to convert to float for interpolation, then back to int8
        if self.input_tensor.dtype == torch.int8:
            input_float = self.input_tensor.float()
            if self.interp_dim == 1:
                ans_float = interpolate_nearest_1d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 2:
                ans_float = interpolate_nearest_2d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 3:
                ans_float = interpolate_nearest_3d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            else:
                raise ValueError(f"Unsupported interpolation dimension: {self.interp_dim}")
            
            # Convert back to int8 and then to float64 for output
            ans = ans_float.round().clamp(-128, 127).to(torch.int8).double()
        else:
            # Use double precision for computation
            input_f64 = self.input_tensor.double()
            
            if self.interp_dim == 1:
                ans = interpolate_nearest_1d(
                    input_f64,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 2:
                ans = interpolate_nearest_2d(
                    input_f64,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 3:
                ans = interpolate_nearest_3d(
                    input_f64,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            else:
                raise ValueError(f"Unsupported interpolation dimension: {self.interp_dim}")
        
        # Add expected output size (spatial dimensions only)
        output_spatial_size = list(ans.shape[2:])  # Skip batch and channel dimensions
        test_writer.add_array(test_writer.gguf_key("output_size"), output_spatial_size)
            
        # Add expected output tensor
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("interpolate_nearest.gguf")
    
    # Data types to test
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    test_cases = []
    
    # Generate comprehensive test cases for each data type and dimension
    for dtype in dtypes:
        
        # ============ 1D Interpolation Tests ============
        # Using size parameter
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((4, 8, 32), dtype),
                size=64, mode='nearest', interp_dim=1,
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 16, 128), dtype),
                size=256, mode='nearest', interp_dim=1,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 4, 100), dtype),
                size=50, mode='nearest', interp_dim=1,
            ),
        ])
        
        # Converted from scale_factor to size
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((2, 6, 48), dtype),
                size=96, mode='nearest', interp_dim=1,  # 48 * 2.0
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 3, 64), dtype),
                size=32, mode='nearest', interp_dim=1,  # 64 * 0.5
            ),
            InterpolateNearestTestCase(
                random_tensor((3, 2, 80), dtype),
                size=120, mode='nearest', interp_dim=1,  # 80 * 1.5
            ),
        ])
        
        # Additional 1D cases
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((2, 4, 56), dtype),
                size=112, mode='nearest', interp_dim=1,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 8, 40), dtype),
                size=30, mode='nearest', interp_dim=1,  # 40 * 0.75
            ),
        ])
        
        # ============ 2D Interpolation Tests ============
        # Using size parameter - square outputs
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((2, 3, 32, 32), dtype),
                size=(64, 64), mode='nearest', interp_dim=2,
            ),
            InterpolateNearestTestCase(
                random_tensor((4, 16, 128, 128), dtype),
                size=(256, 256), mode='nearest', interp_dim=2,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 8, 96, 96), dtype),
                size=(48, 48), mode='nearest', interp_dim=2,
            ),
        ])
        
        # Using size parameter - rectangular outputs
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((2, 4, 40, 60), dtype),
                size=(80, 120), mode='nearest', interp_dim=2,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 6, 72, 48), dtype),
                size=(36, 96), mode='nearest', interp_dim=2,
            ),
            InterpolateNearestTestCase(
                random_tensor((3, 2, 56, 84), dtype),
                size=(28, 42), mode='nearest', interp_dim=2,
            ),
        ])
        
        # Converted from uniform scaling to size
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((2, 3, 24, 24), dtype),
                size=(48, 48), mode='nearest', interp_dim=2,  # 24 * 2.0
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 4, 64, 64), dtype),
                size=(32, 32), mode='nearest', interp_dim=2,  # 64 * 0.5
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 2, 48, 48), dtype),
                size=(60, 60), mode='nearest', interp_dim=2,  # 48 * 1.25
            ),
        ])
        
        # Converted from non-uniform scaling to size
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 3, 32, 48), dtype),
                size=(64, 72), mode='nearest', interp_dim=2,  # 32*2.0, 48*1.5
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 5, 60, 40), dtype),
                size=(45, 50), mode='nearest', interp_dim=2,  # 60*0.75, 40*1.25
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 1, 80, 60), dtype),
                size=(40, 120), mode='nearest', interp_dim=2,  # 80*0.5, 60*2.0
            ),
        ])
        
        # Additional 2D cases
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 2, 44, 44), dtype),
                size=(88, 88), mode='nearest', interp_dim=2,
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 3, 36, 48), dtype),
                size=(54, 36), mode='nearest', interp_dim=2,  # 36*1.5, 48*0.75
            ),
        ])
        
        # ============ 3D Interpolation Tests ============
        # Using size parameter - cubic outputs
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 2, 16, 16, 16), dtype),
                size=(32, 32, 32), mode='nearest', interp_dim=3,
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 4, 24, 24, 24), dtype),
                size=(48, 48, 48), mode='nearest', interp_dim=3,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 1, 40, 40, 40), dtype),
                size=(20, 20, 20), mode='nearest', interp_dim=3,
            ),
        ])
        
        # Using size parameter - non-cubic outputs
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 3, 12, 18, 24), dtype),
                size=(24, 36, 48), mode='nearest', interp_dim=3,
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 2, 20, 16, 28), dtype),
                size=(10, 32, 14), mode='nearest', interp_dim=3,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 1, 32, 24, 16), dtype),
                size=(16, 48, 32), mode='nearest', interp_dim=3,
            ),
        ])
        
        # Converted from uniform scaling to size
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 2, 20, 20, 20), dtype),
                size=(40, 40, 40), mode='nearest', interp_dim=3,  # 20 * 2.0
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 4, 32, 32, 32), dtype),
                size=(16, 16, 16), mode='nearest', interp_dim=3,  # 32 * 0.5
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 2, 24, 24, 24), dtype),
                size=(36, 36, 36), mode='nearest', interp_dim=3,  # 24 * 1.5
            ),
        ])
        
        # Converted from non-uniform scaling to size
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 1, 16, 24, 20), dtype),
                size=(32, 36, 25), mode='nearest', interp_dim=3,  # 16*2.0, 24*1.5, 20*1.25
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 3, 28, 20, 32), dtype),
                size=(21, 40, 16), mode='nearest', interp_dim=3,  # 28*0.75, 20*2.0, 32*0.5
            ),
            InterpolateNearestTestCase(
                random_tensor((2, 1, 36, 24, 18), dtype),
                size=(18, 30, 36), mode='nearest', interp_dim=3,  # 36*0.5, 24*1.25, 18*2.0
            ),
        ])
        
        # Additional 3D cases
        test_cases.extend([
            InterpolateNearestTestCase(
                random_tensor((1, 1, 22, 22, 22), dtype),
                size=(44, 44, 44), mode='nearest', interp_dim=3,
            ),
            InterpolateNearestTestCase(
                random_tensor((1, 2, 18, 24, 30), dtype),
                size=(27, 18, 24), mode='nearest', interp_dim=3,  # 18*1.5, 24*0.75, 30*0.8
            ),
        ])
    
    # Add some edge cases
    edge_cases = [
        # Very small inputs
        InterpolateNearestTestCase(
            random_tensor((1, 1, 4), torch.float32),
            size=16, mode='nearest', interp_dim=1,
        ),
        InterpolateNearestTestCase(
            random_tensor((1, 1, 4, 4), torch.float16),
            size=(16, 16), mode='nearest', interp_dim=2,
        ),
        InterpolateNearestTestCase(
            random_tensor((1, 1, 4, 4, 4), torch.bfloat16),
            size=(8, 8, 8), mode='nearest', interp_dim=3,
        ),
        
        # Large scaling
        InterpolateNearestTestCase(
            random_tensor((1, 2, 8), torch.int8),
            size=32, mode='nearest', interp_dim=1,  # 8 * 4.0
        ),
        InterpolateNearestTestCase(
            random_tensor((1, 1, 16, 16), torch.float32),
            size=(48, 48), mode='nearest', interp_dim=2,  # 16 * 3.0
        ),
        
        # Small scaling
        InterpolateNearestTestCase(
            random_tensor((2, 3, 120), torch.float16),
            size=30, mode='nearest', interp_dim=1,  # 120 * 0.25
        ),
        InterpolateNearestTestCase(
            random_tensor((1, 4, 80, 80), torch.bfloat16),
            size=(8, 8), mode='nearest', interp_dim=2,  # 80 * 0.1
        ),
        
        # Complex scaling
        InterpolateNearestTestCase(
            random_tensor((1, 1, 30, 40, 50), torch.int8),
            size=(18, 72, 20), mode='nearest', interp_dim=3,  # 30*0.6, 40*1.8, 50*0.4
        ),
        
        # Single pixel inputs
        InterpolateNearestTestCase(
            random_tensor((1, 3, 1), torch.float32),
            size=10, mode='nearest', interp_dim=1,
        ),
        InterpolateNearestTestCase(
            random_tensor((1, 1, 1, 1), torch.float16),
            size=(5, 7), mode='nearest', interp_dim=2,
        ),
    ]
    
    test_cases.extend(edge_cases)
    
    print(f"Generated {len(test_cases)} test cases")
    print(f"Data types: {len(dtypes)} types (f32, f16, bf16, int8)")
    print(f"Interpolation dimensions: 1D, 2D, 3D")
    print(f"Mode: nearest only")
    
    test_writer.add_tests(test_cases)
    test_writer.save()
