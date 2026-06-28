#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spvv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spvv(py::module &m) {
    m.def("spvv",
          &op::spvv,
          py::arg("a"),
          py::arg("x"),
          R"doc(Sparse COO vector dense vector dot product.)doc");

    m.def("spvv_",
          &op::spvv_,
          py::arg("y"),
          py::arg("a"),
          py::arg("x"),
          R"doc(In-place sparse COO vector dense vector dot product.)doc");
}

} // namespace infinicore::ops
