#pragma once

#include "../sparse.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * spmm: Sparse Matrix × Dense Matrix Multiplication.
 *
 * Computes C = alpha * A * B where:
 *   A — SpMat, shape (m × k), CSR format
 *   B — dense Tensor, shape (k × n)
 *   C — dense Tensor, shape (m × n)  [output]
 *
 * @param a     CSR sparse matrix
 * @param b     Dense input matrix [k, n]
 * @param alpha Scalar multiplier (default 1.0)
 * @return      Dense output matrix [m, n]
 */
Tensor spmm(const SpMat &a, const Tensor &b, float alpha = 1.0f);

/**
 * spmm_: In-place variant — C = alpha * A * B + beta * C.
 *
 * @param c     Dense output matrix [m, n] (read-write)
 * @param a     CSR sparse matrix (m × k)
 * @param b     Dense input matrix [k, n]
 * @param alpha Scalar multiplier for A*B (default 1.0)
 * @param beta  Scalar multiplier for existing C  (default 0.0)
 */
void spmm_(Tensor c, const SpMat &a, const Tensor &b,
           float alpha = 1.0f, float beta = 0.0f);

} // namespace infinicore::op
