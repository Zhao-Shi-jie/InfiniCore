#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/sparse.hpp"

namespace py = pybind11;

namespace infinicore::sparse {

inline void bind(py::module &m) {
    py::class_<SpMat>(m, "SpMat",
                      R"doc(Sparse matrix in CSR (Compressed Sparse Row) format.

The matrix is stored internally as three dense Tensors:
  - row_offsets : shape [rows+1], dtype int32  — row-pointer array
  - col_indices : shape [nnz],   dtype int32  — column-index array
  - values      : shape [nnz],   dtype T      — non-zero value array

This mirrors the cuSPARSE ``cusparseSpMatDescr_t`` descriptor while reusing
the existing ``Tensor`` primitive so that no new memory management is needed.)doc")

        .def(py::init<Tensor, Tensor, Tensor, size_t, size_t>(),
             py::arg("row_offsets"),
             py::arg("col_indices"),
             py::arg("values"),
             py::arg("rows"),
             py::arg("cols"),
             R"doc(Construct a CSR SpMat from existing Tensors.

Parameters
----------
row_offsets : Tensor
    1-D tensor of shape [rows+1], dtype int32.
col_indices : Tensor
    1-D tensor of shape [nnz], dtype int32.
values : Tensor
    1-D tensor of shape [nnz] containing non-zero values.
rows : int
    Number of rows in the sparse matrix.
cols : int
    Number of columns in the sparse matrix.)doc")

        .def_property_readonly("row_offsets", &SpMat::row_offsets,
                               "Row-pointer tensor [rows+1], int32")
        .def_property_readonly("col_indices", &SpMat::col_indices,
                               "Column-index tensor [nnz], int32")
        .def_property_readonly("values", &SpMat::values,
                               "Non-zero value tensor [nnz]")
        .def_property_readonly("rows", &SpMat::rows,
                               "Number of rows")
        .def_property_readonly("cols", &SpMat::cols,
                               "Number of columns")
        .def_property_readonly("nnz", &SpMat::nnz,
                               "Number of non-zero elements")
        .def_property_readonly("device", &SpMat::device,
                               "Device on which the tensors reside")
        .def_property_readonly("dtype", &SpMat::dtype,
                               "Data type of the non-zero values");
}

} // namespace infinicore::sparse
