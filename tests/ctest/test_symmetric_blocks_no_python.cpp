// Python-free test (also runs with LASYTEN_WITH_PYTHON=OFF, where liblasyten is linked without
// libpython): a symmetry-aware block contraction + SVD built from the pieces that compile against
// Eigen only: product `Symmetry` (U(1) x Z_3) for charge bookkeeping, `Sector` / `SectorArray`,
// and EigenBlockBackend for the dense blocks (tdot, SVD, scale_axis).
//
// Two block-diagonal "tensors" A[x, y] and B[y, z] with legs graded by charge sectors are
// contracted sector by sector and compared with the dense (assembled) product; the SVD of the
// result is computed block by block and used to reconstruct it.

#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <lasyten/block_backend/eigen.h>
#include <lasyten/symmetries/factors/u1.h>
#include <lasyten/symmetries/factors/zn.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/symmetry.h>

#if LASYTEN_HAS_PYTHON
#include <lasyten/embed.h>
#endif

using namespace lasyten;

namespace {

#define SYMM_TEST_CHECK(cond)                                                                     \
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

BlockPtr
make_f64(std::vector<int64> shape, std::vector<double> values)
{
    auto b = std::make_shared<Blk>(Dtype::Float64, shape);
    SYMM_TEST_CHECK(static_cast<size_t>(b->numel()) == values.size());
    std::copy(values.begin(), values.end(), b->data<double>());
    return b;
}

double
at(BlockCPtr const& b, int64 i, int64 j)
{
    auto const* p = dynamic_cast<Blk const*>(b.get());
    SYMM_TEST_CHECK(p != nullptr);
    return p->data<double>()[i * b->shape()[1] + j];
}

/// Block-diagonal matrix: one dense block per sector of the (shared) charge label.
struct BlockMatrix
{
    std::vector<Sector> sectors;  ///< charge of every block (row leg charge == column leg charge)
    std::vector<BlockPtr> blocks; ///< blocks[i] has shape (rows_i, cols_i)
};

} // namespace

int
test_symmetric_blocks_no_python(int /*argc*/, char** /*args*/)
{
#if LASYTEN_HAS_PYTHON
    // With Python enabled, Symmetry reads factor attributes through Python (to support Python
    // subclasses), so an interpreter is needed. With LASYTEN_WITH_PYTHON=OFF none is.
    lasyten::ScopedInterpreter interpreter;
    py::module_::import_("lasyten._core");
    lasyten::init_embedded_nanobind();
#endif

    // ---------------- symmetry: U(1) x Z_3 ----------------
    auto u1 = std::make_shared<U1>("Sz");
    auto z3 = std::make_shared<ZN>(3, "Z3");
    Symmetry symm({ u1, z3 });

    SYMM_TEST_CHECK(symm.num_factors() == 2);
    SYMM_TEST_CHECK(symm.is_abelian());
    SYMM_TEST_CHECK(symm.is_valid_sector(Sector{ 2, 1 }));
    SYMM_TEST_CHECK(!symm.is_valid_sector(Sector{ 2, 3 })); // Z_3 charge out of range
    SYMM_TEST_CHECK(!symm.is_valid_sector(Sector{ 2 }));    // wrong sector length
    SYMM_TEST_CHECK(symm.sector_dim(Sector{ 2, 1 }) == 1);

    Sector a{ 1, 2 };
    Sector b{ -3, 2 };
    SectorArray fused = symm.fusion_outcomes(a, b);
    SYMM_TEST_CHECK(fused.size() == 1);
    SYMM_TEST_CHECK((fused[0] == Sector{ -2, 1 })); // charges add, Z_3: 2 + 2 = 1
    SYMM_TEST_CHECK((symm.dual_sector(a) == Sector{ -1, 1 }));
    SYMM_TEST_CHECK((symm.fusion_outcomes(a, symm.dual_sector(a))[0] == symm.trivial_sector));

    // ---------------- block-wise contraction ----------------
    auto be_ptr = EigenBlockBackend::from_factory("eigen");
    SYMM_TEST_CHECK(be_ptr != nullptr);
    EigenBlockBackend& be = *be_ptr;

    // A[x, y]: charge sectors c0, c1 with x-dims (2, 1) and y-dims (3, 2); B[y, z]: same y-dims,
    // z-dims (2, 2). A block A_c has shape (x_c, y_c), B_c has shape (y_c, z_c).
    Sector c0{ 0, 0 };
    Sector c1{ 1, 2 };
    BlockMatrix A{ { c0, c1 },
                   { make_f64({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }), make_f64({ 1, 2 }, { 7, 8 }) } };
    BlockMatrix B{ { c0, c1 },
                   { make_f64({ 3, 2 }, { 1, 0, 2, 1, 0, 3 }),
                     make_f64({ 2, 2 }, { 1, 2, 3, 4 }) } };
    SYMM_TEST_CHECK(symm.is_valid_sector(c0) && symm.is_valid_sector(c1));

    BlockMatrix C;
    for (std::size_t i = 0; i < A.sectors.size(); ++i) {
        // A_c and B_c carry the same charge on the contracted leg, so they fuse to the trivial
        // sector: only matching sectors contribute (charge conservation).
        SYMM_TEST_CHECK(A.sectors[i] == B.sectors[i]);
        SYMM_TEST_CHECK((symm.fusion_outcomes(A.sectors[i], symm.dual_sector(B.sectors[i]))[0] ==
                         symm.trivial_sector));
        C.sectors.push_back(A.sectors[i]);
        C.blocks.push_back(be.tdot(A.blocks[i], B.blocks[i], { 1 }, { 0 }));
    }
    // Block 0: [[1,2,3],[4,5,6]] @ [[1,0],[2,1],[0,3]] = [[5, 11], [14, 23]];
    // Block 1: [[7, 8]] @ [[1,2],[3,4]] = [[31, 46]].
    SYMM_TEST_CHECK(C.blocks[0]->shape() == (std::vector<int64>{ 2, 2 }));
    SYMM_TEST_CHECK(C.blocks[1]->shape() == (std::vector<int64>{ 1, 2 }));
    SYMM_TEST_CHECK(at(C.blocks[0], 0, 0) == 5 && at(C.blocks[0], 0, 1) == 11);
    SYMM_TEST_CHECK(at(C.blocks[0], 1, 0) == 14 && at(C.blocks[0], 1, 1) == 23);
    SYMM_TEST_CHECK(at(C.blocks[1], 0, 0) == 31 && at(C.blocks[1], 0, 1) == 46);

    // Compare with the dense product of the assembled block-diagonal matrices.
    // Assembled shapes: A (3 x 5), B (5 x 4), C (3 x 4).
    BlockPtr A_dense = make_f64({ 3, 5 },
                                { 1,
                                  2,
                                  3,
                                  0,
                                  0, //
                                  4,
                                  5,
                                  6,
                                  0,
                                  0, //
                                  0,
                                  0,
                                  0,
                                  7,
                                  8 });
    BlockPtr B_dense = make_f64({ 5, 4 }, { 1, 0, 0, 0, //
                                            2, 1, 0, 0, //
                                            0, 3, 0, 0, //
                                            0, 0, 1, 2, //
                                            0, 0, 3, 4 });
    BlockPtr C_dense = be.matrix_dot(A_dense, B_dense);
    SYMM_TEST_CHECK(at(C_dense, 0, 0) == 5 && at(C_dense, 0, 1) == 11);
    SYMM_TEST_CHECK(at(C_dense, 1, 0) == 14 && at(C_dense, 1, 1) == 23);
    SYMM_TEST_CHECK(at(C_dense, 2, 2) == 31 && at(C_dense, 2, 3) == 46);
    SYMM_TEST_CHECK(at(C_dense, 0, 2) == 0 && at(C_dense, 2, 0) == 0); // charge conservation

    // ---------------- block-wise SVD ----------------
    for (std::size_t i = 0; i < C.blocks.size(); ++i) {
        auto [u, s, vh] = be.matrix_svd(C.blocks[i], std::nullopt);
        BlockPtr rec = be.matrix_dot(be.scale_axis(u, s, 1), vh);
        SYMM_TEST_CHECK(be.allclose(rec, C.blocks[i], 0., 1e-10));
        // singular values are non-negative and sorted
        auto const& sv = dynamic_cast<Blk const&>(*s);
        for (int64 k = 0; k + 1 < s->shape()[0]; ++k) {
            SYMM_TEST_CHECK(sv.data<double>()[k] >= sv.data<double>()[k + 1]);
        }
        SYMM_TEST_CHECK(sv.data<double>()[s->shape()[0] - 1] >= 0.);
    }

    // ---------------- sector utilities ----------------
    SectorArray all = SectorArray::from_sector(c0).concat(SectorArray::from_sector(c1));
    SYMM_TEST_CHECK(symm.are_valid_sectors(all));
    SectorArray duals = symm.dual_sectors(all);
    SYMM_TEST_CHECK((duals[1] == Sector{ -1, 1 }));
    SYMM_TEST_CHECK(symm.batch_sector_dim(all) == (std::vector<int64>{ 1, 1 }));

    std::cout << "symmetric block contraction OK (" << symm.repr() << ")\n";
    return 0;
}

#ifdef LASYTEN_TEST_STANDALONE_MAIN
// LASYTEN_WITH_PYTHON=OFF: no create_test_sourcelist driver, build a standalone executable.
int
main(int argc, char** argv)
{
    return test_symmetric_blocks_no_python(argc, argv);
}
#endif
