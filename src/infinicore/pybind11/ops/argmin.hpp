#pragma once

#include "infinicore/ops/argmin.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_argmin(py::module &m) {
    m.def("argmin",
          &op::argmin,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(Return the indices of the minimum values along a given dimension.
          
          Args:
              input: Input tensor
              dim: The dimension to reduce (can be negative)
              keepdim: Whether to keep the reduced dimension (default: False)
              
          Returns:
              Output tensor containing the indices of minimum values)doc");

    m.def("argmin_",
          &op::argmin_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(In-place argmin operation along a given dimension.
          
          Args:
              output: Output tensor to store the result
              input: Input tensor
              dim: The dimension to reduce (can be negative)
              keepdim: Whether to keep the reduced dimension (default: False))doc");
}

} // namespace infinicore::ops
