#pragma once

#include "infinicore/ops/hardshrink.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hardshrink(py::module &m) {
    m.def("hardshrink",
          &op::hardshrink,
          py::arg("input"),
          py::arg("lambd") = 0.5,
          R"doc(Apply the hardshrink activation function element-wise.
          
          hardshrink(x) = x if |x| > lambda, else 0
          
          Args:
              input: Input tensor
              lambda: Threshold value (default: 0.5)
              
          Returns:
              Output tensor with hardshrink applied)doc");

    m.def("hardshrink_",
          &op::hardshrink_,
          py::arg("output"),
          py::arg("input"),
          py::arg("lambd") = 0.5,
          R"doc(In-place hardshrink activation function.
          
          Args:
              output: Output tensor to store the result
              input: Input tensor
              lambda: Threshold value (default: 0.5))doc");
}

} // namespace infinicore::ops
