#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spmv(py::module &m) {
    m.def(
        "spmv",
        [](::infinicore::Tensor input_x, ::infinicore::Tensor values, ::infinicore::Tensor row_ptr, ::infinicore::Tensor col_indices, size_t rows, size_t cols, size_t nnzs) {
            return op::spmv(input_x, values, row_ptr, col_indices, rows, cols, nnzs);
        },
        py::arg("input_x"),
        py::arg("values"),
        py::arg("row_ptr"),
        py::arg("col_indices"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("nnzs"),
        R"doc(SpMV out-of-place.)doc");

    m.def(
        "spmv_",
        [](::infinicore::Tensor output, ::infinicore::Tensor input_x, ::infinicore::Tensor values, ::infinicore::Tensor row_ptr, ::infinicore::Tensor col_indices, size_t rows, size_t cols, size_t nnzs) {
            op::spmv_(output, input_x, values, row_ptr, col_indices, rows, cols, nnzs);
        },
        py::arg("output"),
        py::arg("input_x"),
        py::arg("values"),
        py::arg("row_ptr"),
        py::arg("col_indices"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("nnzs"),
        R"doc(SpMV in-place variant writing to provided output tensor.)doc");
}

} // namespace infinicore::ops
