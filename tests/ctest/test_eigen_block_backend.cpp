// Tests for the native Eigen/LAPACK BlockBackend.
//
// All computations are done by EigenBlockBackend itself; the embedded Python interpreter is only
// needed for the ``to_numpy`` / ``block_from_numpy`` round trip. With LASYTEN_WITH_PYTHON=OFF
// (LASYTEN_HAS_PYTHON=0) that part is compiled out and the test runs without Python at all.

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <algorithm>
#include <filesystem>

#include <lasyten/block_backend/eigen.h>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>

#if LASYTEN_HAS_PYTHON
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#endif
using namespace lasyten;

namespace {

#define EIGEN_TEST_CHECK(cond)                                                                    \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::ostringstream os_;                                                               \
            os_ << __FILE__ << ":" << __LINE__ << ": check failed: " #cond;                       \
            throw std::runtime_error(os_.str());                                                  \
        }                                                                                         \
    } while (false)

using Blk = EigenBlockBackend::Block;
using BlockPtr = BlockBackend::BlockPtr;
using BlockCPtr = BlockBackend::BlockCPtr;

const Blk&
eb(const BlockCPtr& b)
{
    auto* p = dynamic_cast<const Blk*>(b.get());
    EIGEN_TEST_CHECK(p != nullptr);
    return *p;
}

BlockPtr
make_f64(std::vector<int64> shape, std::vector<double> values)
{
    auto b = std::make_shared<Blk>(Dtype::Float64, shape);
    EIGEN_TEST_CHECK(static_cast<size_t>(b->numel()) == values.size());
    std::copy(values.begin(), values.end(), b->data<double>());
    return b;
}

BlockPtr
make_c128(std::vector<int64> shape, std::vector<std::complex<double>> values)
{
    auto b = std::make_shared<Blk>(Dtype::Complex128, shape);
    EIGEN_TEST_CHECK(static_cast<size_t>(b->numel()) == values.size());
    std::copy(values.begin(), values.end(), b->data<std::complex<double>>());
    return b;
}

std::vector<double>
values_f64(const BlockCPtr& b)
{
    const Blk& blk = eb(b);
    EIGEN_TEST_CHECK(blk.dtype() == Dtype::Float64);
    const double* p = blk.data<double>();
    return std::vector<double>(p, p + blk.numel());
}

bool
close(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-12)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol)
            return false;
    return true;
}

bool
allclose(EigenBlockBackend& be, const BlockCPtr& a, const BlockCPtr& b, double tol = 1e-10)
{
    return be.allclose(a, b, 0., tol);
}

using Shape = std::vector<int64>;

} // namespace

int
test_eigen_block_backend(int /*argc*/, char** /*args*/)
{
#if LASYTEN_HAS_PYTHON
    py::scoped_interpreter guard{};
    py::module_::import("numpy");
#endif

    // ---------------- factory ----------------
    EigenBlockBackend* be_ptr = EigenBlockBackend::from_factory("eigen");
    EIGEN_TEST_CHECK(be_ptr != nullptr);
    EigenBlockBackend& be = *be_ptr;
    EIGEN_TEST_CHECK(EigenBlockBackend::from_factory("cpu") == be_ptr);
    EIGEN_TEST_CHECK(be.get_backend_name() == "EigenBlockBackend");
    {
        bool threw = false;
        try {
            EigenBlockBackend::from_factory("cuda");
        } catch (const std::exception&) {
            threw = true;
        }
        EIGEN_TEST_CHECK(threw);
    }

    // ---------------- zeros / ones / eye ----------------
    {
        BlockPtr z = be.zeros({ 2, 3 }, Dtype::Float64, std::nullopt);
        EIGEN_TEST_CHECK(z->shape() == (Shape{ 2, 3 }));
        EIGEN_TEST_CHECK(z->dtype() == Dtype::Float64);
        EIGEN_TEST_CHECK(close(values_f64(z), std::vector<double>(6, 0.)));

        BlockPtr o = be.ones_block({ 3, 2 }, Dtype::Float64, std::nullopt);
        EIGEN_TEST_CHECK(o->shape() == (Shape{ 3, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(o), std::vector<double>(6, 1.)));

        BlockPtr oc = be.ones_block({ 2 }, Dtype::Complex128, std::nullopt);
        EIGEN_TEST_CHECK(oc->dtype() == Dtype::Complex128);
        EIGEN_TEST_CHECK(eb(oc).data<std::complex<double>>()[1] == std::complex<double>(1., 0.));

        BlockPtr zi = be.zeros({ 4 }, Dtype::Int64, std::nullopt);
        EIGEN_TEST_CHECK(zi->dtype() == Dtype::Int64);
        BlockPtr zb = be.zeros({ 0, 3 }, Dtype::Float64, std::nullopt);
        EIGEN_TEST_CHECK(zb->shape() == (Shape{ 0, 3 }));

        BlockPtr e = be.eye_matrix(3, Dtype::Float64, std::nullopt);
        EIGEN_TEST_CHECK(e->shape() == (Shape{ 3, 3 }));
        EIGEN_TEST_CHECK(close(values_f64(e), { 1, 0, 0, 0, 1, 0, 0, 0, 1 }));
    }

    // ---------------- reshape / permute_axes ----------------
    {
        BlockPtr a = make_f64({ 2, 3 }, { 0, 1, 2, 3, 4, 5 });
        BlockPtr r = be.reshape(a, { 3, 2 });
        EIGEN_TEST_CHECK(r->shape() == (Shape{ 3, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(r), { 0, 1, 2, 3, 4, 5 }));
        BlockPtr r2 = be.reshape(a, { -1 });
        EIGEN_TEST_CHECK(r2->shape() == (Shape{ 6 }));

        BlockPtr t = be.permute_axes(a, { 1, 0 });
        EIGEN_TEST_CHECK(t->shape() == (Shape{ 3, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(t), { 0, 3, 1, 4, 2, 5 }));

        BlockPtr b = make_f64({ 2, 3, 4 }, std::vector<double>(24));
        for (int i = 0; i < 24; ++i)
            std::dynamic_pointer_cast<Blk>(b)->data<double>()[i] = i;
        BlockPtr p = be.permute_axes(b, { 2, 0, 1 });
        EIGEN_TEST_CHECK(p->shape() == (Shape{ 4, 2, 3 }));
        // p[k, i, j] == b[i, j, k]
        const double* pd = eb(p).data<double>();
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    EIGEN_TEST_CHECK(pd[(k * 2 + i) * 3 + j] == i * 12 + j * 4 + k);
    }

    // ---------------- elementwise: abs / add / sub / mul / broadcast ----------------
    {
        BlockPtr a = make_f64({ 2, 2 }, { -1, 2, -3, 4 });
        BlockPtr ab = be.abs(a);
        EIGEN_TEST_CHECK(close(values_f64(ab), { 1, 2, 3, 4 }));

        BlockPtr s = *a + *ab;
        EIGEN_TEST_CHECK(close(values_f64(s), { 0, 4, 0, 8 }));
        BlockPtr d = *a - *ab;
        EIGEN_TEST_CHECK(close(values_f64(d), { -2, 0, -6, 0 }));
        BlockPtr m = *a * *a;
        EIGEN_TEST_CHECK(close(values_f64(m), { 1, 4, 9, 16 }));

        // broadcasting (2,2) + (2,)
        BlockPtr row = make_f64({ 2 }, { 10, 20 });
        BlockPtr br = *a + *row;
        EIGEN_TEST_CHECK(close(values_f64(br), { 9, 22, 7, 24 }));

        BlockPtr gt = *a > *be.zeros({ 2, 2 }, Dtype::Float64, std::nullopt);
        EIGEN_TEST_CHECK(gt->dtype() == Dtype::Bool);
        EIGEN_TEST_CHECK(!be.all(gt));

        // complex abs
        BlockPtr c = make_c128({ 2 }, { { 3, 4 }, { 0, -2 } });
        BlockPtr cabs = be.abs(c);
        EIGEN_TEST_CHECK(cabs->dtype() == Dtype::Float64);
        EIGEN_TEST_CHECK(close(values_f64(cabs), { 5, 2 }));
    }

    // ---------------- matrix_dot / tdot ----------------
    {
        BlockPtr a = make_f64({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        BlockPtr b = make_f64({ 3, 2 }, { 7, 8, 9, 10, 11, 12 });
        BlockPtr c = be.matrix_dot(a, b);
        EIGEN_TEST_CHECK(c->shape() == (Shape{ 2, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(c), { 58, 64, 139, 154 }));

        BlockPtr t = be.tdot(a, b, { 1 }, { 0 });
        EIGEN_TEST_CHECK(allclose(be, t, c));

        // contract two legs: a[i, j, k] b[j, k, l] -> [i, l]
        BlockPtr x = make_f64({ 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 });
        BlockPtr y = make_f64({ 2, 2, 2 }, { 1, 0, 0, 1, 1, 1, 0, 2 });
        BlockPtr z = be.tdot(x, y, { 1, 2 }, { 0, 1 });
        EIGEN_TEST_CHECK(z->shape() == (Shape{ 2, 2 }));
        // x[0,:,:] = [[1,2],[3,4]], y[:,:,0] = [[1,0],[1,0]] -> 1+0+3+0 = 4;
        // y[:,:,1]=[[0,1],[1,2]] -> 0+2+3+8 = 13 x[1,:,:] = [[5,6],[7,8]] -> 5+7=12 ; 0+6+7+16 =
        // 29
        EIGEN_TEST_CHECK(close(values_f64(z), { 4, 13, 12, 29 }));

        // outer product (no contraction)
        BlockPtr u = make_f64({ 2 }, { 1, 2 });
        BlockPtr v = make_f64({ 2 }, { 3, 4 });
        BlockPtr uv = be.tdot(u, v, {}, {});
        EIGEN_TEST_CHECK(uv->shape() == (Shape{ 2, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(uv), { 3, 4, 6, 8 }));

        // complex
        BlockPtr ca = make_c128({ 2, 2 }, { { 1, 1 }, { 0, 1 }, { 2, 0 }, { 1, -1 } });
        BlockPtr ci = be.eye_matrix(2, Dtype::Complex128, std::nullopt);
        BlockPtr cm = be.matrix_dot(ca, ci);
        EIGEN_TEST_CHECK(allclose(be, cm, ca));
    }

    // ---------------- matrix_svd ----------------
    {
        const auto& algs = be.possible_svd_algorithms();
        EIGEN_TEST_CHECK(!algs.empty());
        for (const std::string& alg : algs) {
            for (bool cplx : { false, true }) {
                BlockPtr a;
                if (cplx) {
                    a = make_c128(
                      { 3, 2 }, { { 1, 1 }, { 2, 0 }, { 0, 3 }, { 1, -1 }, { 4, 2 }, { -1, 0 } });
                } else {
                    a = make_f64({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
                }
                auto [u, s, vh] = be.matrix_svd(a, alg);
                EIGEN_TEST_CHECK(u->shape() == (Shape{ 3, 2 }));
                EIGEN_TEST_CHECK(s->shape() == (Shape{ 2 }));
                EIGEN_TEST_CHECK(vh->shape() == (Shape{ 2, 2 }));
                EIGEN_TEST_CHECK(s->dtype() == Dtype::Float64);
                // descending, non-negative
                EIGEN_TEST_CHECK(eb(s).data<double>()[0] >= eb(s).data<double>()[1]);
                EIGEN_TEST_CHECK(eb(s).data<double>()[1] >= 0.);
                // u @ diag(s) @ vh == a
                BlockPtr us = be.scale_axis(u, s, 1);
                BlockPtr rec = be.matrix_dot(us, vh);
                EIGEN_TEST_CHECK(allclose(be, rec, a));
                // u^dagger u == 1
                BlockPtr uhu = be.matrix_dot(be.conj(be.permute_axes(u, { 1, 0 })), u);
                EIGEN_TEST_CHECK(allclose(be, uhu, be.eye_matrix(2, uhu->dtype(), std::nullopt)));
            }
        }
        // known singular values of [[3, 0], [0, 4]] -> (4, 3)
        BlockPtr d = make_f64({ 2, 2 }, { 3, 0, 0, 4 });
        auto [u, s, vh] = be.matrix_svd(d, std::nullopt);
        EIGEN_TEST_CHECK(close(values_f64(s), { 4, 3 }));
    }

    // ---------------- matrix_qr ----------------
    {
        for (bool cplx : { false, true }) {
            for (bool full : { false, true }) {
                BlockPtr a;
                if (cplx)
                    a = make_c128(
                      { 3, 2 }, { { 1, 1 }, { 2, 0 }, { 0, 3 }, { 1, -1 }, { 4, 2 }, { -1, 0 } });
                else
                    a = make_f64({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
                auto [q, r] = be.matrix_qr(a, full);
                const int64 k = full ? 3 : 2;
                EIGEN_TEST_CHECK(q->shape() == (Shape{ 3, k }));
                EIGEN_TEST_CHECK(r->shape() == (Shape{ k, 2 }));
                EIGEN_TEST_CHECK(allclose(be, be.matrix_dot(q, r), a));
                BlockPtr qhq = be.matrix_dot(be.conj(be.permute_axes(q, { 1, 0 })), q);
                EIGEN_TEST_CHECK(allclose(be, qhq, be.eye_matrix(k, qhq->dtype(), std::nullopt)));
            }
        }
    }

    // ---------------- eigh ----------------
    {
        BlockPtr h = make_f64({ 2, 2 }, { 2, 1, 1, 2 });
        auto [w, v] = be.eigh(h, std::nullopt);
        EIGEN_TEST_CHECK(w->shape() == (Shape{ 2 }));
        EIGEN_TEST_CHECK(close(values_f64(w), { 1, 3 }));
        BlockPtr recon = be.matrix_dot(be.scale_axis(v, w, 1), be.permute_axes(v, { 1, 0 }));
        EIGEN_TEST_CHECK(allclose(be, recon, h));

        BlockPtr hc = make_c128({ 2, 2 }, { { 1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } });
        auto [wc, vc] = be.eigh(hc, std::nullopt);
        EIGEN_TEST_CHECK(close(values_f64(wc), { 0, 2 }));
        BlockPtr rec_c =
          be.matrix_dot(be.scale_axis(vc, wc, 1), be.conj(be.permute_axes(vc, { 1, 0 })));
        EIGEN_TEST_CHECK(allclose(be, rec_c, hc));
    }

    // ---------------- reductions / misc ----------------
    {
        BlockPtr a = make_f64({ 2, 2 }, { 1, 2, 3, 4 });
        EIGEN_TEST_CHECK(std::abs(be.sum_all(a).as_float64() - 10.) < 1e-12);
        EIGEN_TEST_CHECK(std::abs(be.max(a).as_float64() - 4.) < 1e-12);
        EIGEN_TEST_CHECK(std::abs(be.min(a).as_float64() - 1.) < 1e-12);
        EIGEN_TEST_CHECK(std::abs(be.norm(a, 2., std::nullopt).as_float64() - std::sqrt(30.)) <
                         1e-12);
        EIGEN_TEST_CHECK(std::abs(be.trace_full(a).as_float64() - 5.) < 1e-12);
        BlockPtr diag = be.get_diagonal(a, std::nullopt);
        EIGEN_TEST_CHECK(close(values_f64(diag), { 1, 4 }));
        BlockPtr kr = be.kron(make_f64({ 1, 1 }, { 2 }), a);
        EIGEN_TEST_CHECK(close(values_f64(kr), { 2, 4, 6, 8 }));
    }

    // ---------------- apply_axis_permutations (Python-free) ----------------
    {
        BlockPtr a = make_f64({ 2, 3 }, { 0, 1, 2, 3, 4, 5 });
        BlockPtr p = be.apply_axis_permutations(a, { { 1, 0 }, { 2, 0, 1 } });
        EIGEN_TEST_CHECK(close(values_f64(p), { 5, 3, 4, 2, 0, 1 }));
        bool threw = false;
        try {
            (void)be.apply_axis_permutations(a, { { 1, 0 } });
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        EIGEN_TEST_CHECK(threw);
    }

    // ---------------- HDF5 round trip (Python-free) ----------------
    {
        BlockPtr a = make_f64({ 2, 3 }, { 0, 1, 2, 3, 4, 5 });
        BlockPtr c = make_c128({ 2 }, { { 1, 2 }, { 3, -4 } });
        auto mb = std::make_shared<Blk>(Dtype::Bool, Shape{ 3 });
        mb->data<bool>()[0] = true;
        mb->data<bool>()[2] = true;
        BlockPtr m = mb;
        const auto path = std::filesystem::temp_directory_path() / "lasyten_eigen_block_test.h5";
        {
            HighFive::File file(path.string(), HighFive::File::Truncate);
            lasyten::hdf5::Saver saver(file);
            hdf5_export::save_block(saver, "a", a);
            hdf5_export::save_block(saver, "c", c);
            hdf5_export::save_block(saver, "m", m);
        }
        {
            HighFive::File file(path.string(), HighFive::File::ReadOnly);
            lasyten::hdf5::Loader loader(file);
            BlockPtr a2 = hdf5_export::load_block(loader, "a");
            BlockPtr c2 = hdf5_export::load_block(loader, "c");
            BlockPtr m2 = hdf5_export::load_block(loader, "m");
            EIGEN_TEST_CHECK(a2->shape() == a->shape() && a2->dtype() == Dtype::Float64);
            EIGEN_TEST_CHECK(allclose(be, a, a2));
            EIGEN_TEST_CHECK(c2->dtype() == Dtype::Complex128 && allclose(be, c, c2));
            EIGEN_TEST_CHECK(m2->dtype() == Dtype::Bool);
            EIGEN_TEST_CHECK(eb(m2).data<bool>()[0] && !eb(m2).data<bool>()[1] &&
                             eb(m2).data<bool>()[2]);
        }
        std::filesystem::remove(path);
    }

    // ---------------- stream output (Python-free repr) ----------------
    {
        BlockPtr a = make_f64({ 2, 2 }, { 1, 2, 3, 4 });
        std::ostringstream os;
        os << *a;
        EIGEN_TEST_CHECK(os.str().find("shape=(2, 2)") != std::string::npos);
        EIGEN_TEST_CHECK(os.str().find("[[") != std::string::npos);
    }

#if LASYTEN_HAS_PYTHON
    // ---------------- to_numpy round trip ----------------
    {
        BlockPtr a = make_f64({ 2, 3 }, { 0, 1, 2, 3, 4, 5 });
        py::array arr = eb(a).to_numpy();
        EIGEN_TEST_CHECK(arr.ndim() == 2);
        EIGEN_TEST_CHECK(arr.shape(0) == 2 && arr.shape(1) == 3);
        py::object np = py::module_::import("numpy");
        EIGEN_TEST_CHECK(py::str(arr.attr("dtype")).cast<std::string>() == "float64");
        EIGEN_TEST_CHECK(arr.attr("sum")().cast<double>() == 15.);
        BlockPtr back = be.block_from_numpy(arr, std::nullopt, std::nullopt);
        EIGEN_TEST_CHECK(back->shape() == a->shape());
        EIGEN_TEST_CHECK(allclose(be, a, back));
        // non-contiguous (transposed) input is copied into C order
        py::array tr = py::reinterpret_borrow<py::array>(arr.attr("T"));
        BlockPtr bt = be.block_from_numpy(tr, std::nullopt, std::nullopt);
        EIGEN_TEST_CHECK(bt->shape() == (Shape{ 3, 2 }));
        EIGEN_TEST_CHECK(close(values_f64(bt), { 0, 3, 1, 4, 2, 5 }));
        // the block owns its data: mutating the numpy copy does not change the block
        arr.attr("fill")(-1.);
        EIGEN_TEST_CHECK(close(values_f64(a), { 0, 1, 2, 3, 4, 5 }));
        // complex
        BlockPtr c = make_c128({ 2 }, { { 1, 2 }, { 3, -4 } });
        py::array carr = eb(c).to_numpy();
        EIGEN_TEST_CHECK(py::str(carr.attr("dtype")).cast<std::string>() == "complex128");
        BlockPtr cback = be.block_from_numpy(carr, std::nullopt, std::nullopt);
        EIGEN_TEST_CHECK(allclose(be, c, cback));
    }
#endif // LASYTEN_HAS_PYTHON

    std::cout << "test_eigen_block_backend: OK" << std::endl;
    return 0;
}
