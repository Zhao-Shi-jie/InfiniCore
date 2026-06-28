#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spmm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spmm(py::module &m) {
    m.def("spmm",
          &op::spmm,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(Sparse CSR matrix multiplication with a dense tensor.)doc");

    m.def("spmm_",
          &op::spmm_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(In-place sparse CSR matrix multiplication.)doc");
}

} // namespace infinicore::ops
