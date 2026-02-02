#pragma once

#include "infinicore/ops/cosine_embedding_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cosine_embedding_loss(py::module &m) {

    m.def("cosine_embedding_loss",
          &op::cosine_embedding_loss,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("target"),
          py::arg("margin") = 0.0,
          py::arg("reduction") = "mean",
          R"doc(Compute cosine embedding loss between input1 and input2..)doc");

    m.def("cosine_embedding_loss_",
          &op::cosine_embedding_loss_,
          py::arg("out"),
          py::arg("input1"),
          py::arg("input2"),
          py::arg("target"),
          py::arg("margin") = 0.0,
          py::arg("reduction") = "mean",
          R"doc(Compute cosine embedding loss between input1 and input2..)doc");
}

} // namespace infinicore::ops
