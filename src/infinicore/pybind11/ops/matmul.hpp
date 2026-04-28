#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/matmul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_matmul(py::module &m) {
    m.def("matmul",
          &op::matmul,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("matmul_",
          &op::matmul_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(In-place matrix multiplication.)doc");

    m.def("gemm",
          py::overload_cast<const SpMat &, const Tensor &, float, float>(&op::gemm),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(Matrix multiplication with a sparse CSR left-hand side.)doc");

    m.def("gemm_",
          py::overload_cast<Tensor, const SpMat &, const Tensor &, float, float>(&op::gemm_),
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(In-place matrix multiplication with a sparse CSR left-hand side.)doc");
}

} // namespace infinicore::ops
