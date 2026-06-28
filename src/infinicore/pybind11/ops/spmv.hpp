#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spmv(py::module &m) {
    m.def("spmv",
          &op::spmv,
          py::arg("a"),
          py::arg("x"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(Sparse CSR matrix vector multiplication.)doc");

    m.def("spmv_",
          &op::spmv_,
          py::arg("y"),
          py::arg("a"),
          py::arg("x"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(In-place sparse CSR matrix vector multiplication.)doc");
}

} // namespace infinicore::ops
