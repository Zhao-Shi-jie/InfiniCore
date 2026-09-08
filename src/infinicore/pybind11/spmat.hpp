#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::spmat {

inline void bind(py::module &m) {
    py::class_<SpMat>(m, "SpMat")
        .def_property_readonly("format", [](const SpMat &spmat) { return static_cast<int>(spmat->format()); })
        .def_property_readonly("rows", [](const SpMat &spmat) { return spmat->rows(); })
        .def_property_readonly("cols", [](const SpMat &spmat) { return spmat->cols(); })
        .def_property_readonly("nnz", [](const SpMat &spmat) { return spmat->nnz(); })
        .def_property_readonly("ell_width", [](const SpMat &spmat) { return spmat->ell_width(); })
        .def_property_readonly("slice_height", [](const SpMat &spmat) { return spmat->slice_height(); })
        .def_property_readonly("sigma", [](const SpMat &spmat) { return spmat->sigma(); })
        .def_property_readonly("dtype", [](const SpMat &spmat) { return spmat->dtype(); })
        .def_property_readonly("index_dtype", [](const SpMat &spmat) { return spmat->index_dtype(); })
        .def_property_readonly("device", [](const SpMat &spmat) { return spmat->device(); })
        .def_property_readonly("crow_indices", [](const SpMat &spmat) { return spmat->crow_indices(); })
        .def_property_readonly("col_indices", [](const SpMat &spmat) { return spmat->col_indices(); })
        .def_property_readonly("slice_offsets", [](const SpMat &spmat) { return spmat->slice_offsets(); })
        .def_property_readonly("row_indices", [](const SpMat &spmat) { return spmat->row_indices(); })
        .def_property_readonly("values", [](const SpMat &spmat) { return spmat->values(); });

    m.def("csr_spmat",
          &SpMat::csr,
          py::arg("crow_indices"),
          py::arg("col_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"));

    m.def("coo_spmat",
          &SpMat::coo,
          py::arg("row_indices"),
          py::arg("col_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"));

    m.def("ell_spmat",
          &SpMat::ell,
          py::arg("col_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("ell_width"),
          py::arg("nnz"));

    m.def("sell_spmat",
          &SpMat::sell,
          py::arg("slice_offsets"),
          py::arg("col_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("slice_height"),
          py::arg("nnz"));

    m.def("sell_sigma_c_spmat",
          &SpMat::sell_sigma_c,
          py::arg("slice_offsets"),
          py::arg("col_indices"),
          py::arg("row_indices"),
          py::arg("values"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("slice_height"),
          py::arg("sigma"),
          py::arg("nnz"));
}

} // namespace infinicore::spmat
