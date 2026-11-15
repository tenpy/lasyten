// Python-free tensor test (also runs with LASYTEN_WITH_PYTHON=OFF, where liblasyten is linked
// without libpython): `ElementarySpace` / `TensorProduct`, `SymmetricTensor` on the
// `NoSymmetryBackend` with the `EigenBlockBackend`, `tdot`, `compose`, `svd`, `qr`, `lq` and
// `norm`, each checked against a dense reference computed by plain loops.

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <lasyten/backends/no_symmetry.h>
#include <lasyten/block_backend/eigen.h>
#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/decompositions.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/symmetric_tensor.h>

#if LASYTEN_HAS_PYTHON
#include <lasyten/embed.h>
#endif

using namespace lasyten;

namespace {

#define TENSOR_TEST_CHECK(cond)                                                                   \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::ostringstream os_;                                                               \
            os_ << __FILE__ << ":" << __LINE__ << ": check failed: " #cond;                       \
            throw std::runtime_error(os_.str());                                                  \
        }                                                                                         \
    } while (false)

using Blk = EigenBlockBackend::Block;
using BlockPtr = BlockBackend::BlockPtr;

/// Deterministic, "generic" real entries.
std::vector<double>
pseudo_random(std::size_t n, double seed)
{
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::sin(seed + 1.7 * static_cast<double>(i)) + 0.3 * std::cos(0.37 * i * i);
    }
    return out;
}

BlockPtr
make_f64(std::vector<int64> shape, std::vector<double> const& values)
{
    auto b = std::make_shared<Blk>(Dtype::Float64, shape);
    TENSOR_TEST_CHECK(static_cast<std::size_t>(b->numel()) == values.size());
    std::copy(values.begin(), values.end(), b->data<double>());
    return b;
}

bool
dense_close(BlockBackend& bb, BlockPtr const& a, BlockPtr const& b, double atol = 1e-10)
{
    return a->shape() == b->shape() && bb.allclose(a, b, 0., atol);
}

SymmetricTensor::Ptr
as_symmetric(TensorPtr const& t)
{
    auto res = std::dynamic_pointer_cast<SymmetricTensor>(t);
    TENSOR_TEST_CHECK(res != nullptr);
    return res;
}

TensorPtr
as_tensor(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    TENSOR_TEST_CHECK(std::holds_alternative<TensorPtr>(v));
    return std::get<TensorPtr>(v);
}

} // namespace

int
test_tensor_no_python(int /*argc*/, char** /*argv*/)
{
#if LASYTEN_HAS_PYTHON
    // Match test_symmetric_blocks_no_python: Symmetry / factor attributes go through Python
    // when the interpreter is available, so register nanobind types first.
    ScopedInterpreter interp;
    py::module_::import_("lasyten._core");
    init_embedded_nanobind();
#endif
    auto block_backend = EigenBlockBackend::from_factory_shared("eigen");
    BlockBackend& bb = *block_backend;
    auto backend = std::make_shared<NoSymmetryBackend>(block_backend);
    auto symmetry = std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() });

    auto vec_space = [&](int64 dim, bool is_dual = false) -> Leg::Ptr {
        return ElementarySpace::from_trivial_sector(dim, symmetry, is_dual);
    };
    constexpr int64 d0 = 3;
    constexpr int64 d1 = 2;
    constexpr int64 d2 = 4;
    constexpr int64 d3 = 5;

    auto V0 = vec_space(d0);
    auto V1 = vec_space(d1);
    auto V2 = vec_space(d2);
    auto V3 = vec_space(d3);
    TENSOR_TEST_CHECK(V0.dim() == d0);
    TENSOR_TEST_CHECK(V2.dual().dim() == d2);

    auto cod_A = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ V0, V1 });
    auto dom_A = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ V2 });
    auto cod_B = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ V2 });
    auto dom_B = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ V3 });

    // dense references: A[i, j, k] with legs (V0, V1, V2*), B[k, l] with legs (V2, V3*)
    std::vector<double> a_vals = pseudo_random(d0 * d1 * d2, 0.3);
    std::vector<double> b_vals = pseudo_random(d2 * d3, 1.1);
    BlockPtr a_dense = make_f64({ d0, d1, d2 }, a_vals);
    BlockPtr b_dense = make_f64({ d2, d3 }, b_vals);

    auto A = SymmetricTensor::from_dense_block(
      a_dense, cod_A, dom_A, backend, LegLabelList{ "i", "j", "k" });
    auto B =
      SymmetricTensor::from_dense_block(b_dense, cod_B, dom_B, backend, LegLabelList{ "k*", "l" });
    A->test_sanity();
    B->test_sanity();
    TENSOR_TEST_CHECK(A->num_legs == 3);
    TENSOR_TEST_CHECK(A->num_codomain_legs() == 2);
    TENSOR_TEST_CHECK(A->dtype == Dtype::Float64);
    TENSOR_TEST_CHECK(dense_close(bb, A->to_dense_block(), a_dense));

    // ---------------- norm ----------------
    {
        double expect = 0.;
        for (double v : a_vals) {
            expect += v * v;
        }
        double got = norm(TensorCPtr(A)).as_float64();
        TENSOR_TEST_CHECK(std::abs(got - std::sqrt(expect)) < 1e-12);
    }

    // ---------------- tdot / compose ----------------
    {
        std::vector<double> c_vals(d0 * d1 * d3, 0.);
        for (int64 i = 0; i < d0 * d1; ++i) {
            for (int64 l = 0; l < d3; ++l) {
                for (int64 k = 0; k < d2; ++k) {
                    c_vals[i * d3 + l] += a_vals[i * d2 + k] * b_vals[k * d3 + l];
                }
            }
        }
        BlockPtr c_ref = make_f64({ d0, d1, d3 }, c_vals);

        auto C = as_symmetric(as_tensor(tdot(A, B, { LegRef{ 2 } }, { LegRef{ 0 } })));
        C->test_sanity();
        TENSOR_TEST_CHECK(C->num_legs == 3);
        TENSOR_TEST_CHECK(dense_close(bb, C->to_dense_block(), c_ref));

        // same contraction by labels
        auto C2 = as_symmetric(as_tensor(tdot(A, B, { LegRef{ "k" } }, { LegRef{ "k*" } })));
        TENSOR_TEST_CHECK(dense_close(bb, C2->to_dense_block(), c_ref));

        // compose: A is (V0 V1 <- V2), B is (V2 <- V3)
        auto C3 = as_symmetric(as_tensor(compose(A, B)));
        TENSOR_TEST_CHECK(C3->num_codomain_legs() == 2);
        TENSOR_TEST_CHECK(C3->num_domain_legs() == 1);
        TENSOR_TEST_CHECK(dense_close(bb, C3->to_dense_block(), c_ref));
    }

    // ---------------- svd ----------------
    {
        auto [U, S, Vh] = svd(A, LegLabelList{ "a", "b", "c", "d" });
        U->test_sanity();
        S->test_sanity();
        Vh->test_sanity();
        // economic SVD of a (d0 * d1) x d2 = 6 x 4 matrix
        TENSOR_TEST_CHECK(S->leg()->dim == d2);
        TENSOR_TEST_CHECK(U->num_codomain_legs() == 2);
        TENSOR_TEST_CHECK(Vh->num_domain_legs() == 1);
        auto const labels_U = U->labels.as_list();
        TENSOR_TEST_CHECK(labels_U.back() == "a");
        TENSOR_TEST_CHECK(Vh->labels.as_list().front() == "d");

        auto rec = as_tensor(compose(as_tensor(compose(U, S)), Vh));
        TENSOR_TEST_CHECK(dense_close(bb, rec->to_dense_block(), a_dense));

        // Vh Vh^dagger = 1 and U^dagger U = 1, checked on the dense matrices
        auto Vh_dense = Vh->to_dense_block(); // shape (d2, d2) here: new leg x d2
        TENSOR_TEST_CHECK(Vh_dense->shape() == (std::vector<int64>{ d2, d2 }));
        auto vv = bb.matrix_dot(Vh_dense, bb.permute_axes(Vh_dense, { 1, 0 }));
        auto eye = make_f64({ d2, d2 }, [] {
            std::vector<double> e(d2 * d2, 0.);
            for (int64 i = 0; i < d2; ++i) {
                e[i * d2 + i] = 1.;
            }
            return e;
        }());
        TENSOR_TEST_CHECK(dense_close(bb, vv, eye));

        // truncation-free `S` is non-negative
        auto s_blk = S->to_dense_block();
        TENSOR_TEST_CHECK(s_blk->shape() == (std::vector<int64>{ d2, d2 }));
        for (int64 i = 0; i < d2; ++i) {
            TENSOR_TEST_CHECK(bb.get_block_element(s_blk, { i, i }).as_float64() >= 0.);
        }
    }

    // ---------------- qr / lq ----------------
    {
        auto [Q, R] = qr(A, LegLabelList{ "m" });
        Q->test_sanity();
        R->test_sanity();
        auto rec = as_tensor(compose(Q, R));
        TENSOR_TEST_CHECK(dense_close(bb, rec->to_dense_block(), a_dense));
        TENSOR_TEST_CHECK(Q->labels.as_list().back() == "m");
        TENSOR_TEST_CHECK(R->labels.as_list().front() == "m*");

        auto [L, Q2] = lq(A, LegLabelList{ "m" });
        L->test_sanity();
        Q2->test_sanity();
        auto rec2 = as_tensor(compose(L, Q2));
        TENSOR_TEST_CHECK(dense_close(bb, rec2->to_dense_block(), a_dense));
    }

    std::cout << "NoSymmetry tensor stack on Eigen OK ("
              << A->backend->block_backend->get_backend_name() << ")\n";
    return 0;
}

#ifdef LASYTEN_TEST_STANDALONE_MAIN
// LASYTEN_WITH_PYTHON=OFF: no create_test_sourcelist driver, build a standalone executable.
int
main(int argc, char** argv)
{
    return test_tensor_no_python(argc, argv);
}
#endif
