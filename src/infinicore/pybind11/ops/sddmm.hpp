#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sddmm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sddmm(py::module &m) {
    m.def("sddmm_",
          &op::sddmm_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(In-place sampled dense-dense matrix multiplication over a CSR pattern.)doc");
}

} // namespace infinicore::ops
