#pragma once

#include "infinicore/ops/hardsigmoid.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hardsigmoid(py::module &m) {
    m.def("hardsigmoid",
          &op::hardsigmoid,
          py::arg("input"),
          R"doc(Apply the hardsigmoid activation function element-wise.
          
          hardsigmoid(x) = max(0, min(1, (x + 3) / 6))
          
          Args:
              input: Input tensor
              
          Returns:
              Output tensor with hardsigmoid applied)doc");

    m.def("hardsigmoid_",
          &op::hardsigmoid_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place hardsigmoid activation function.
          
          Args:
              output: Output tensor to store the result
              input: Input tensor)doc");
}

} // namespace infinicore::ops
