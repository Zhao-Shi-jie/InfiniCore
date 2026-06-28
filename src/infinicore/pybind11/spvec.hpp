#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::spvec {

inline void bind(py::module &m) {
    py::class_<SpVec>(m, "SpVec")
        .def_property_readonly("size", [](const SpVec &spvec) { return spvec->size(); })
        .def_property_readonly("nnz", [](const SpVec &spvec) { return spvec->nnz(); })
        .def_property_readonly("dtype", [](const SpVec &spvec) { return spvec->dtype(); })
        .def_property_readonly("index_dtype", [](const SpVec &spvec) { return spvec->index_dtype(); })
        .def_property_readonly("device", [](const SpVec &spvec) { return spvec->device(); })
        .def_property_readonly("indices", [](const SpVec &spvec) { return spvec->indices(); })
        .def_property_readonly("values", [](const SpVec &spvec) { return spvec->values(); });

    m.def("coo_spvec",
          &SpVec::coo,
          py::arg("indices"),
          py::arg("values"),
          py::arg("size"));
}

} // namespace infinicore::spvec
