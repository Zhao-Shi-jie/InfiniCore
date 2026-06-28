#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/axpby.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_axpby(py::module &m) {
    m.def("axpby",
          &op::axpby,
          py::arg("x"),
          py::arg("y"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 1.0f,
          R"doc(Out-of-place sparse axpby, returning alpha * x_sparse + beta * y.)doc");

    m.def("axpby_",
          &op::axpby_,
          py::arg("x"),
          py::arg("y"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 1.0f,
          R"doc(In-place sparse axpby, updating y to alpha * x_sparse + beta * y.)doc");
}

} // namespace infinicore::ops
