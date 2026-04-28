#include "infinicore/ops/spmm.hpp"
#include "../../utils.hpp"

#include <cstring>
#include <stdexcept>

namespace infinicore::op {

namespace {

// ---------------------------------------------------------------------------
// Typed CPU kernel: C = alpha * CSR(A) * B + beta * C
// T must be a plain arithmetic type (float / double).
// ---------------------------------------------------------------------------
template <typename T>
void csr_spmm_cpu(T *C,
                  const int32_t *row_offsets,
                  const int32_t *col_indices,
                  const T *values,
                  const T *B,
                  size_t m, size_t n,
                  float alpha, float beta) {
    // Scale existing C by beta
    if (beta == 0.0f) {
        std::memset(C, 0, m * n * sizeof(T));
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < m * n; ++i) {
            C[i] = static_cast<T>(static_cast<float>(C[i]) * beta);
        }
    }

    // Accumulate alpha * A * B
    for (size_t row = 0; row < m; ++row) {
        int32_t row_start = row_offsets[row];
        int32_t row_end = row_offsets[row + 1];
        for (int32_t idx = row_start; idx < row_end; ++idx) {
            int32_t col = col_indices[idx];
            float val = alpha * static_cast<float>(values[idx]);
            for (size_t j = 0; j < n; ++j) {
                C[row * n + j] = static_cast<T>(
                    static_cast<float>(C[row * n + j]) +
                    val * static_cast<float>(B[col * n + j]));
            }
        }
    }
}

void dispatch_cpu(Tensor c, const SpMat &a, const Tensor &b,
                  float alpha, float beta) {
    const size_t m = a.rows();
    const size_t k = a.cols();
    const size_t n = b->size(1);

    INFINICORE_ASSERT(b->ndim() == 2 && b->size(0) == k);
    INFINICORE_ASSERT(c->ndim() == 2 && c->size(0) == m && c->size(1) == n);
    INFINICORE_ASSERT(b->is_contiguous());
    INFINICORE_ASSERT(c->is_contiguous());

    const auto *row_offsets =
        reinterpret_cast<const int32_t *>(a.row_offsets()->data());
    const auto *col_indices =
        reinterpret_cast<const int32_t *>(a.col_indices()->data());

    switch (a.dtype()) {
    case DataType::F32:
        csr_spmm_cpu(
            reinterpret_cast<float *>(c->data()),
            row_offsets, col_indices,
            reinterpret_cast<const float *>(a.values()->data()),
            reinterpret_cast<const float *>(b->data()),
            m, n, alpha, beta);
        break;
    case DataType::F64:
        csr_spmm_cpu(
            reinterpret_cast<double *>(c->data()),
            row_offsets, col_indices,
            reinterpret_cast<const double *>(a.values()->data()),
            reinterpret_cast<const double *>(b->data()),
            m, n, alpha, beta);
        break;
    default:
        throw std::runtime_error(
            "spmm: unsupported dtype (only F32 and F64 are supported on CPU)");
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Tensor spmm(const SpMat &a, const Tensor &b, float alpha) {
    INFINICORE_ASSERT(a.device() == b->device());
    auto c = Tensor::zeros({a.rows(), b->size(1)}, a.dtype(), a.device());
    spmm_(c, a, b, alpha, 0.0f);
    return c;
}

void spmm_(Tensor c, const SpMat &a, const Tensor &b,
           float alpha, float beta) {
    INFINICORE_ASSERT(a.device() == b->device());
    INFINICORE_ASSERT(a.device() == c->device());

    switch (a.device().getType()) {
    case Device::Type::CPU:
        dispatch_cpu(c, a, b, alpha, beta);
        break;
    default:
        throw std::runtime_error(
            "spmm: unsupported device — only CPU is currently supported");
    }
}

} // namespace infinicore::op
