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
          R"doc(Sparse Matrix × Dense Matrix multiplication (SpMM).

Computes ``C = alpha * A * B`` where A is a CSR sparse matrix and B is a
dense matrix.  The result C is a newly allocated dense matrix.

Parameters
----------
a : SpMat
    CSR sparse matrix of shape (m, k).
b : Tensor
    Dense input matrix of shape (k, n).
alpha : float, optional
    Scalar multiplier (default 1.0).

Returns
-------
Tensor
    Dense output matrix of shape (m, n).)doc");

    m.def("spmm_",
          &op::spmm_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(In-place Sparse Matrix × Dense Matrix multiplication.

Computes ``C = alpha * A * B + beta * C`` in-place.

Parameters
----------
c : Tensor
    Dense output matrix of shape (m, n) — modified in-place.
a : SpMat
    CSR sparse matrix of shape (m, k).
b : Tensor
    Dense input matrix of shape (k, n).
alpha : float, optional
    Scalar multiplier for A*B (default 1.0).
beta : float, optional
    Scalar multiplier for the existing C (default 0.0).)doc");
}

} // namespace infinicore::ops
