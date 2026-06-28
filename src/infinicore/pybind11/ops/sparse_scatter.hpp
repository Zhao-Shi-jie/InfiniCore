#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sparse_scatter.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sparse_scatter(py::module &m) {
    m.def("sparse_scatter",
          &op::sparse_scatter,
          py::arg("input"),
          R"doc(Scatter COO sparse vector values into a dense vector.)doc");

    m.def("sparse_scatter_",
          &op::sparse_scatter_,
          py::arg("output"),
          py::arg("input"),
          R"doc(Explicit-output sparse scatter.)doc");
}

} // namespace infinicore::ops
