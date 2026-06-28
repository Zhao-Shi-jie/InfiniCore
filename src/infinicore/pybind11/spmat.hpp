#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::spmat {

inline void bind(py::module &m) {
    py::class_<SpMat>(m, "SpMat")
        .def_property_readonly("rows", [](const SpMat &spmat) { return spmat->rows(); })
        .def_property_readonly("cols", [](const SpMat &spmat) { return spmat->cols(); })
        .def_property_readonly("nnz", [](const SpMat &spmat) { return spmat->nnz(); })
        .def_property_readonly("dtype", [](const SpMat &spmat) { return spmat->dtype(); })
        .def_property_readonly("index_dtype", [](const SpMat &spmat) { return spmat->index_dtype(); })
        .def_property_readonly("device", [](const SpMat &spmat) { return spmat->device(); })
        .def_property_readonly("crow_indices", [](const SpMat &spmat) { return spmat->crow_indices(); })
        .def_property_readonly("col_indices", [](const SpMat &spmat) { return spmat->col_indices(); })
        .def_property_readonly("values", [](const SpMat &spmat) { return spmat->values(); });

    m.def("csr_spmat",
          &SpMat::csr,
          py::arg("crow_indices"),
          py::arg("col_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"));
}

} // namespace infinicore::spmat
