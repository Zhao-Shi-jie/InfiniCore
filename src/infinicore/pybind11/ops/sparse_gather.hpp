#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sparse_gather.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sparse_gather(py::module &m) {
    m.def("sparse_gather",
          &op::sparse_gather,
          py::arg("pattern"),
          py::arg("input"),
          R"doc(Gather dense vector values at COO sparse vector indices.)doc");

    m.def("sparse_gather_",
          &op::sparse_gather_,
          py::arg("output"),
          py::arg("pattern"),
          py::arg("input"),
          R"doc(Explicit-output sparse gather.)doc");
}

} // namespace infinicore::ops
