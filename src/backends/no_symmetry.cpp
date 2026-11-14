#include <lasyten/backends/no_symmetry.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/tools.h>

#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <typeinfo>

namespace lasyten {

namespace {

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(ElementarySpace::Ptr const& leg, int64 idx)
{
    if (!leg) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return leg->parse_index(idx);
}

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(Space::Ptr const& leg, int64 idx)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(leg);
    if (!es) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return es->parse_index(idx);
}

[[nodiscard]] std::pair<int64, int64>
parse_leg_index(Leg const& leg, int64 idx)
{
    auto es = leg.as_elementary_space();
    if (!es) {
        throw std::invalid_argument("parse_index requires an ElementarySpace leg");
    }
    return es->parse_index(idx);
}

int64
leg_dim_i64(py::object leg)
{
    return static_cast<int64>(leg.attr("dim").cast<float64>());
}

int64
space_dim_i64(Space const& space)
{
    return static_cast<int64>(space.dim);
}

std::vector<int64>
shape_from_tensor(TensorCPtr a)
{
    // Tensor.shape is float64 in C++ (non-integer dims for some symmetries); NoSymmetry
    // still expects integer block shapes.
    std::vector<int64> out;
    out.reserve(a->shape.size());
    for (auto item : a->shape) {
        out.push_back(static_cast<int64>(item));
    }
    return out;
}

std::vector<int64>
dims_from_legs(std::vector<Leg::Ptr> const& legs)
{
    std::vector<int64> dims;
    dims.reserve(legs.size());
    for (auto const& leg : legs)
        dims.push_back(static_cast<int64>(leg.dim()));
    return dims;
}

int64
prod_i64(std::vector<int64> const& vals)
{
    return std::accumulate(vals.begin(), vals.end(), int64{ 1 }, std::multiplies<int64>{});
}

std::optional<std::vector<int64>>
rank_basis_perm_masked(py::object basis_perm, BlockBackend& bb, BlockBackend::BlockPtr const& mask)
{
    if (basis_perm.is_none())
        return std::nullopt;
    py::object indexed = basis_perm[bb.to_numpy(mask)];
    auto arr = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(indexed);
    auto buf = arr.unchecked<1>();
    std::vector<int64> vals(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        vals[static_cast<std::size_t>(i)] = buf(i);
    return rank_data(vals);
}

bool
space_is_dual(Space const& space)
{
    if (auto const* es = dynamic_cast<ElementarySpace const*>(&space))
        return es->is_dual;
    return false;
}

ElementarySpace::Ptr
elementary_from_trivial(Space const& large_leg,
                        int64 dim,
                        std::optional<std::vector<int64>> basis_perm)
{
    return ElementarySpace::from_trivial_sector(
      dim, large_leg.symmetry, space_is_dual(large_leg), std::move(basis_perm));
}

} // namespace

NoSymmetryBackend::BlockData::BlockData(BlockBackend::BlockPtr b)
  : block(std::move(b))
{
    if (!block)
        throw std::invalid_argument("NoSymmetryBackend::BlockData: null block");
}

TensorBackend::DataPtr
NoSymmetryBackend::wrap(BlockBackend::BlockPtr b)
{
    return std::make_shared<BlockData>(std::move(b));
}

BlockBackend::BlockPtr
NoSymmetryBackend::unwrap(DataPtr d)
{
    if (!d)
        throw std::invalid_argument("NoSymmetryBackend::unwrap: null DataPtr");
    auto* bd = dynamic_cast<BlockData*>(d.get());
    if (!bd) {
        const auto& obj = *d;
        throw std::invalid_argument(std::format(
          "NoSymmetryBackend::unwrap: expected BlockData, got {}", typeid(obj).name()));
    }
    return bd->block;
}

BlockBackend::BlockPtr
NoSymmetryBackend::block_from_tensor(TensorCPtr tensor)
{
    if (auto st = std::dynamic_pointer_cast<const SymmetricTensor>(tensor))
        return unwrap(st->data);
    if (auto m = std::dynamic_pointer_cast<const Mask>(tensor))
        return unwrap(m->data);
    throw std::invalid_argument(
      "NoSymmetryBackend::block_from_tensor: expected SymmetricTensor or Mask");
}

NoSymmetryBackend::NoSymmetryBackend(std::shared_ptr<BlockBackend> block_backend_)
  : TensorBackend(std::move(block_backend_))
{
}

void
NoSymmetryBackend::test_tensor_sanity(TensorCPtr a, bool is_diagonal)
{
    TensorBackend::test_tensor_sanity(a, is_diagonal);
    std::vector<int64> expect_shape;
    if (is_diagonal) {
        expect_shape = { leg_dim_i64(py::cast(a->legs()).attr("__getitem__")(0)) };
    } else {
        expect_shape = shape_from_tensor(a);
    }
    block_backend->test_block_sanity(block_from_tensor(a), expect_shape, a->dtype, a->device);
}

void
NoSymmetryBackend::test_mask_sanity(MaskCPtr a)
{
    TensorBackend::test_mask_sanity(a);
    auto data = block_from_tensor(a);
    block_backend->test_block_sanity(
      data, std::vector<int64>{ leg_dim_i64(py::cast(a->large_leg())) }, Dtype::Bool, a->device);
    auto small_dim = static_cast<int64>(py::cast(a->small_leg()).attr("dim").cast<float64>());
    check(block_backend->sum_all(data).as_int64() == small_dim,
          "test_mask_sanity: mask block sum mismatch");
}

TensorBackend::DataPtr
NoSymmetryBackend::act_block_diagonal_square_matrix(SymmetricTensorCPtr a,
                                                    BlockUnaryFn block_method,
                                                    std::optional<DtypeMapFn> /*dtype_map*/)
{
    return wrap(block_method(block_from_tensor(a)));
}

TensorBackend::DataPtr
NoSymmetryBackend::add_trivial_leg(TensorCPtr a,
                                   int64 legs_pos,
                                   bool add_to_domain,
                                   int64 co_domain_pos,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    return wrap(block_backend->add_axis(block_from_tensor(a), legs_pos));
}

bool
NoSymmetryBackend::almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol)
{
    return block_backend->allclose(block_from_tensor(a), block_from_tensor(b), rtol, atol);
}

TensorBackend::DataPtr
NoSymmetryBackend::apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask)
{
    return wrap(block_backend->apply_mask(block_from_tensor(tensor), block_from_tensor(mask), 0));
}

TensorBackend::DataPtr
NoSymmetryBackend::combine_legs(TensorCPtr tensor,
                                std::vector<std::vector<int64>> leg_idcs_combine,
                                std::vector<Leg::Ptr> pipes,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain)
{
    int64 num_codomain_legs = tensor->num_codomain_legs();
    std::vector<bool> cstyles;
    cstyles.reserve(pipes.size());
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        bool in_domain = leg_idcs_combine[i][0] >= num_codomain_legs;
        auto pipe = pipes[i].as_leg_pipe();
        if (!pipe)
            throw std::invalid_argument("NoSymmetryBackend::combine_legs expects LegPipe");
        cstyles.push_back(pipe->combine_cstyle != in_domain);
    }
    return wrap(block_backend->combine_legs(block_from_tensor(tensor), leg_idcs_combine, cstyles));
}

TensorBackend::DataPtr
NoSymmetryBackend::compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    int64 a_num_codomain = a->num_codomain_legs();
    int64 a_num_legs = a->num_legs;
    int64 b_num_codomain = b->num_codomain_legs();
    std::vector<int64> a_domain;
    for (int64 i = a_num_legs - 1; i >= a_num_codomain; --i)
        a_domain.push_back(i);
    std::vector<int64> b_codomain(static_cast<std::size_t>(b_num_codomain));
    std::iota(b_codomain.begin(), b_codomain.end(), int64{ 0 });
    return wrap(
      block_backend->tdot(block_from_tensor(a), block_from_tensor(b), a_domain, b_codomain));
}

TensorBackend::DataPtr
NoSymmetryBackend::copy_data(TensorCPtr a, std::optional<std::string> device)
{
    return wrap(block_backend->copy_block(block_from_tensor(a), device));
}

TensorBackend::DataPtr
NoSymmetryBackend::dagger(TensorCPtr a)
{
    return wrap(block_backend->dagger(block_from_tensor(a)));
}

BlockBackend::Scalar
NoSymmetryBackend::data_item(DataPtr a)
{
    return block_backend->item(unwrap(a));
}

bool
NoSymmetryBackend::diagonal_all(DiagonalTensorCPtr a)
{
    return block_backend->all(block_from_tensor(a));
}

bool
NoSymmetryBackend::diagonal_any(DiagonalTensorCPtr a)
{
    return block_backend->any(block_from_tensor(a));
}

TensorBackend::DataPtr
NoSymmetryBackend::diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                               DiagonalTensorCPtr b,
                                               BlockBinaryFn func,
                                               bool /*partial_zero_is_zero*/)
{
    return wrap(func(block_from_tensor(a), block_from_tensor(b)));
}

TensorBackend::DataPtr
NoSymmetryBackend::diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                              BlockUnaryFn func,
                                              bool /*maps_zero_to_zero*/)
{
    return wrap(func(block_from_tensor(a)));
}

TensorBackend::DataPtr
NoSymmetryBackend::diagonal_from_block(BlockBackend::BlockPtr a,
                                       TensorProduct::Ptr /*co_domain*/,
                                       float64 /*tol*/)
{
    return wrap(std::move(a));
}

TensorBackend::DataPtr
NoSymmetryBackend::diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                                   TensorProduct::Ptr co_domain)
{
    Sector coupled = co_domain->symmetry->trivial_sector;
    return wrap(func(std::vector<int64>{ space_dim_i64(*co_domain) }, coupled));
}

TensorBackend::DataPtr
NoSymmetryBackend::diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                                    std::optional<float64> tol)
{
    return wrap(block_backend->get_diagonal(block_from_tensor(a), tol));
}

BlockBackend::Scalar
NoSymmetryBackend::diagonal_tensor_trace_full(DiagonalTensorCPtr a)
{
    return block_backend->sum_all(block_from_tensor(a));
}

BlockBackend::BlockPtr
NoSymmetryBackend::diagonal_tensor_to_block(DiagonalTensorCPtr a)
{
    return block_from_tensor(a);
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::diagonal_to_mask(DiagonalTensorCPtr tens)
{
    auto large_leg = py::cast(tens->leg()).cast<Space::Ptr>();
    auto data = block_from_tensor(tens);
    py::object basis_perm_obj = py::cast(tens->leg()).attr("_basis_perm");
    auto basis_perm = rank_basis_perm_masked(basis_perm_obj, *block_backend, data);
    int64 dim = block_backend->sum_all(data).as_int64();
    auto small_leg = elementary_from_trivial(*large_leg, dim, std::move(basis_perm));
    return { wrap(data), std::move(small_leg) };
}

std::tuple<Space::Ptr, TensorBackend::DataPtr>
NoSymmetryBackend::diagonal_transpose(DiagonalTensorCPtr tens)
{
    auto leg = py::cast(tens->leg()).cast<Space::Ptr>();
    return { leg->dual_space(), wrap(block_from_tensor(tens)) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::eigh(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    auto new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    int64 J = a->num_codomain_legs();
    int64 N = 2 * J;
    std::vector<int64> perm;
    perm.reserve(static_cast<std::size_t>(N));
    for (int64 i = 0; i < J; ++i)
        perm.push_back(i);
    for (int64 i = N - 1; i >= J; --i)
        perm.push_back(i);
    auto mat = block_backend->permute_axes(block_from_tensor(a), perm);
    int64 k = space_dim_i64(*py::cast(a->domain).cast<TensorProduct::Ptr>());
    mat = block_backend->reshape(mat, { k, k });
    auto [w, v] = block_backend->eigh(mat, sort);
    auto shape_codom = shape_from_tensor(a);
    shape_codom.resize(static_cast<std::size_t>(J));
    shape_codom.push_back(k);
    v = block_backend->reshape(v, shape_codom);
    return { wrap(std::move(w)), wrap(std::move(v)), std::move(new_leg) };
}

namespace {

struct NoSymSquareMat
{
    BlockBackend::BlockPtr mat;
    int64 k = 0;
    int64 J = 0;
    ElementarySpace::Ptr new_leg;
};

NoSymSquareMat
no_sym_as_square_matrix(NoSymmetryBackend& self, SymmetricTensorCPtr a, bool new_leg_dual)
{
    NoSymSquareMat out;
    out.new_leg = a->domain->as_ElementarySpace(new_leg_dual);
    out.J = a->num_codomain_legs();
    int64 N = 2 * out.J;
    std::vector<int64> perm;
    perm.reserve(static_cast<std::size_t>(N));
    for (int64 i = 0; i < out.J; ++i)
        perm.push_back(i);
    for (int64 i = N - 1; i >= out.J; --i)
        perm.push_back(i);
    out.mat = self.block_backend->permute_axes(NoSymmetryBackend::block_from_tensor(a), perm);
    out.k = space_dim_i64(*py::cast(a->domain).cast<TensorProduct::Ptr>());
    out.mat = self.block_backend->reshape(out.mat, { out.k, out.k });
    return out;
}

BlockBackend::BlockPtr
no_sym_reshape_eigvecs(NoSymmetryBackend& self,
                       SymmetricTensorCPtr a,
                       BlockBackend::BlockPtr v,
                       int64 J,
                       int64 k)
{
    auto shape_codom = shape_from_tensor(a);
    shape_codom.resize(static_cast<std::size_t>(J));
    shape_codom.push_back(k);
    return self.block_backend->reshape(std::move(v), shape_codom);
}

} // namespace

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::eig(SymmetricTensorCPtr a, bool new_leg_dual, std::optional<std::string> sort)
{
    auto sq = no_sym_as_square_matrix(*this, a, new_leg_dual);
    auto [w, v] = block_backend->eig(sq.mat, sort);
    v = no_sym_reshape_eigvecs(*this, a, std::move(v), sq.J, sq.k);
    return { wrap(std::move(w)), wrap(std::move(v)), std::move(sq.new_leg) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::eigvalsh(SymmetricTensorCPtr a,
                            bool new_leg_dual,
                            std::optional<std::string> sort)
{
    auto sq = no_sym_as_square_matrix(*this, a, new_leg_dual);
    auto w = block_backend->eigvalsh(sq.mat, sort);
    return { wrap(std::move(w)), std::move(sq.new_leg) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::eigvals(SymmetricTensorCPtr a,
                           bool new_leg_dual,
                           std::optional<std::string> sort)
{
    auto sq = no_sym_as_square_matrix(*this, a, new_leg_dual);
    auto w = block_backend->eigvals(sq.mat, sort);
    return { wrap(std::move(w)), std::move(sq.new_leg) };
}

TensorBackend::DataPtr
NoSymmetryBackend::eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device)
{
    // --- hints from Python NoSymmetryBackend.eye_data ---
    // Note: the identity has the same matrix elements in all ONB, so ne need to consider
    // ---
    // Note: the identity has the same matrix elements in all ONB, so no need to consider
    //       the basis perms.
    std::vector<int64> legs;
    legs.reserve(co_domain->factors.size());
    for (auto const& f : co_domain->factors)
        legs.push_back(static_cast<int64>(f.dim()));
    return wrap(block_backend->eye_block(legs, dtype, device));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_dense_block(BlockBackend::BlockPtr a,
                                    TensorProduct::Ptr /*codomain*/,
                                    TensorProduct::Ptr /*domain*/,
                                    float64 /*tol*/)
{
    return wrap(std::move(a));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_dense_block_trivial_sector(BlockBackend::BlockPtr block, Space::Ptr leg)
{
    // --- hints from Python NoSymmetryBackend.from_dense_block_trivial_sector ---
    // there are no other sectors, so this is just the unmodified block.
    // ---
    // there are no other sectors, so this is just the unmodified block.
    if (block_backend->get_shape(block) != std::vector<int64>{ space_dim_i64(*leg) }) {
        throw std::invalid_argument(
          "from_dense_block_trivial_sector: block shape does not match leg dim");
    }
    return wrap(std::move(block));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_grid(std::vector<std::vector<py::object>> grid,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain,
                             std::vector<std::vector<int64>> left_mult_slices,
                             std::vector<std::vector<int64>> right_mult_slices,
                             Dtype dtype,
                             std::string device)
{
    auto const& heights = left_mult_slices[0];
    auto const& widths = right_mult_slices[0];
    auto data = unwrap(zero_data(new_codomain, new_domain, dtype, device, false));
    std::size_t n_codom = new_codomain->factors.size();
    std::size_t n_dom = new_domain->factors.size();
    auto one = block_backend->as_scalar(1.0);
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid[i].size(); ++j) {
            py::object op = grid[i][j];
            if (op.is_none())
                continue;
            py::list slcs;
            slcs.append(py::slice(heights[i], heights[i + 1], 1));
            for (std::size_t k = 0; k + 1 < n_codom; ++k)
                slcs.append(py::slice(py::none(), py::none(), py::none()));
            slcs.append(py::slice(widths[j], widths[j + 1], 1));
            for (std::size_t k = 0; k + 1 < n_dom; ++k)
                slcs.append(py::slice(py::none(), py::none(), py::none()));
            py::tuple key = py::tuple(slcs);
            auto view = data->get_item(key);
            auto updated = block_backend->linear_combination(
              one, view, one, block_from_tensor(op.cast<TensorCPtr>()));
            data->set_item(key, py::cast(updated));
        }
    }
    return wrap(std::move(data));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_random_normal(TensorProduct::Ptr codomain,
                                      TensorProduct::Ptr domain,
                                      float64 sigma,
                                      Dtype dtype,
                                      std::string device)
{
    auto shape = dims_from_legs(conventional_leg_order(codomain, domain));
    return wrap(block_backend->random_normal(shape, dtype, sigma, device));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_sector_block_func(SectorBlockFactoryFn func,
                                          TensorProduct::Ptr codomain,
                                          TensorProduct::Ptr domain)
{
    Sector coupled = codomain->symmetry->trivial_sector;
    auto dims = dims_from_legs(conventional_leg_order(codomain, domain));
    return wrap(func(dims, coupled));
}

TensorBackend::DataPtr
NoSymmetryBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype /*dtype*/,
  std::string /*device*/)
{
    if (trees.size() != 1) {
        throw std::invalid_argument("from_tree_map: expected exactly one tree for NoSymmetry");
    }
    auto block = trees.begin()->second;
    auto expect_shape = dims_from_legs(conventional_leg_order(codomain, domain));
    if (block_backend->get_shape(block) != expect_shape) {
        throw std::invalid_argument("from_tree_map: block shape does not match legs");
    }
    return wrap(std::move(block));
}

TensorBackend::DataPtr
NoSymmetryBackend::full_data_from_diagonal_tensor(DiagonalTensorCPtr a)
{
    return wrap(block_backend->block_from_diagonal(block_from_tensor(a)));
}

TensorBackend::DataPtr
NoSymmetryBackend::full_data_from_mask(MaskCPtr a, Dtype dtype)
{
    return wrap(block_backend->block_from_mask(block_from_tensor(a), dtype));
}

std::string
NoSymmetryBackend::get_device_from_data(DataPtr a)
{
    return block_backend->get_device(unwrap(a));
}

Dtype
NoSymmetryBackend::get_dtype_from_data(DataPtr a)
{
    return block_backend->get_dtype(unwrap(a));
}

BlockBackend::Scalar
NoSymmetryBackend::get_element(SymmetricTensorCPtr a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    std::vector<int64> internal;
    internal.reserve(idcs.size());
    for (std::size_t i = 0; i < idcs.size(); ++i) {
        internal.push_back(py::cast(legs[i])
                             .attr("apply_basis_perm")(
                               idcs[i], py::arg("inverse") = true, py::arg("pre_compose") = true)
                             .cast<int64>());
    }
    return block_backend->get_block_element(block_from_tensor(a), internal);
}

BlockBackend::Scalar
NoSymmetryBackend::get_element_diagonal(DiagonalTensorCPtr a, int64 idx)
{
    // --- hints from Python NoSymmetryBackend.get_element_diagonal ---
    // a.data is a single 1D block
    // ---
    // a.data is a single 1D block
    idx = parse_leg_index(a->leg(), idx).second;
    return block_backend->get_block_element(block_from_tensor(a), { idx });
}

BlockBackend::Scalar
NoSymmetryBackend::get_element_mask(MaskCPtr a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    std::vector<int64> parsed;
    parsed.reserve(idcs.size());
    for (std::size_t i = 0; i < idcs.size(); ++i) {
        parsed.push_back(parse_leg_index(legs[i], idcs[i]).second);
    }
    int64 large, small;
    if (a->is_projection) {
        small = parsed[0];
        large = parsed[1];
    } else {
        large = parsed[0];
        small = parsed[1];
    }
    return block_backend->get_block_mask_element(block_from_tensor(a), large, small);
}

BlockBackend::Scalar
NoSymmetryBackend::inner(SymmetricTensorCPtr a, SymmetricTensorCPtr b, bool do_dagger)
{
    return block_backend->inner(block_from_tensor(a), block_from_tensor(b), do_dagger);
}

TensorBackend::DataPtr
NoSymmetryBackend::inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                           Space::Ptr /*space*/,
                                                           ElementarySpace::Ptr /*charge_leg*/)
{
    return wrap(block_backend->add_axis(vector, 1));
}

BlockBackend::BlockPtr
NoSymmetryBackend::inv_part_to_dense_block_single_sector(SymmetricTensorCPtr tensor)
{
    return block_from_tensor(tensor)->get_item(
      py::make_tuple(py::slice(py::none(), py::none(), py::none()), 0));
}

TensorBackend::DataPtr
NoSymmetryBackend::linear_combination(BlockBackend::Scalar a,
                                      TensorCPtr v,
                                      BlockBackend::Scalar b,
                                      TensorCPtr w)
{
    return wrap(
      block_backend->linear_combination(a, block_from_tensor(v), b, block_from_tensor(w)));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
NoSymmetryBackend::lq(SymmetricTensorCPtr tensor, TensorProduct::Ptr new_co_domain)
{
    auto shape = shape_from_tensor(tensor);
    int64 n_codom = tensor->num_codomain_legs();
    std::vector<int64> l_dims(shape.begin(), shape.begin() + n_codom);
    std::vector<int64> q_dims(shape.begin() + n_codom, shape.end());
    auto mat =
      block_backend->reshape(block_from_tensor(tensor), { prod_i64(l_dims), prod_i64(q_dims) });
    auto [l, q] = block_backend->matrix_lq(mat, false);
    int64 k = block_backend->get_shape(q)[0];
    l_dims.push_back(k);
    std::vector<int64> q_shape;
    q_shape.push_back(k);
    q_shape.insert(q_shape.end(), q_dims.begin(), q_dims.end());
    l = block_backend->reshape(l, l_dims);
    q = block_backend->reshape(q, q_shape);
    return { wrap(std::move(l)), wrap(std::move(q)) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::mask_binary_operand(MaskCPtr mask1, MaskCPtr mask2, BlockBinaryFn func)
{
    auto large_leg = py::cast(mask1->large_leg()).cast<Space::Ptr>();
    auto data = func(block_from_tensor(mask1), block_from_tensor(mask2));
    auto basis_perm = rank_basis_perm_masked(
      py::cast(mask1->large_leg()).attr("_basis_perm"), *block_backend, data);
    auto small_leg = elementary_from_trivial(
      *large_leg, block_backend->sum_all(data).as_int64(), std::move(basis_perm));
    return { wrap(std::move(data)), std::move(small_leg) };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
NoSymmetryBackend::mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    auto parsed = tensor->_parse_leg_idx(leg_idx);
    bool in_domain = std::get<0>(parsed);
    int64 co_domain_idx = std::get<1>(parsed);
    leg_idx = std::get<2>(parsed);
    auto data =
      block_backend->apply_mask(block_from_tensor(tensor), block_from_tensor(mask), leg_idx);
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    if (in_domain) {
        codomain = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
        auto spaces = tensor->domain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] = mask->small_leg();
        domain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    } else {
        domain = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
        auto spaces = tensor->codomain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] = mask->small_leg();
        codomain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    }
    return { wrap(std::move(data)), std::move(codomain), std::move(domain) };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
NoSymmetryBackend::mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx)
{
    auto parsed = tensor->_parse_leg_idx(leg_idx);
    bool in_domain = std::get<0>(parsed);
    int64 co_domain_idx = std::get<1>(parsed);
    leg_idx = std::get<2>(parsed);
    auto data =
      block_backend->enlarge_leg(block_from_tensor(tensor), block_from_tensor(mask), leg_idx);
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    if (in_domain) {
        codomain = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
        auto spaces = tensor->domain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] = mask->large_leg();
        domain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    } else {
        domain = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
        auto spaces = tensor->codomain->factors;
        spaces[static_cast<std::size_t>(co_domain_idx)] = mask->large_leg();
        codomain = std::make_shared<TensorProduct>(std::move(spaces), tensor->symmetry);
    }
    return { wrap(std::move(data)), std::move(codomain), std::move(domain) };
}

TensorBackend::DataPtr
NoSymmetryBackend::mask_dagger(MaskCPtr mask)
{
    return wrap(block_from_tensor(mask));
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    py::object basis_perm_obj = py::cast(large_leg).attr("_basis_perm");
    auto basis_perm = rank_basis_perm_masked(basis_perm_obj, *block_backend, a);
    auto small_leg = elementary_from_trivial(
      *large_leg, block_backend->sum_all(a).as_int64(), std::move(basis_perm));
    return { wrap(std::move(a)), std::move(small_leg) };
}

BlockBackend::BlockPtr
NoSymmetryBackend::mask_to_block(MaskCPtr a)
{
    return block_from_tensor(a);
}

TensorBackend::DataPtr
NoSymmetryBackend::mask_to_diagonal(MaskCPtr a, Dtype dtype)
{
    return wrap(block_backend->to_dtype(block_from_tensor(a), dtype));
}

std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
NoSymmetryBackend::mask_transpose(MaskCPtr tens)
{
    auto space_in =
      py::cast(tens->codomain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    auto space_out = py::cast(tens->domain).attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    return { std::move(space_in), std::move(space_out), wrap(block_from_tensor(tens)) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
NoSymmetryBackend::mask_unary_operand(MaskCPtr mask, BlockUnaryFn func)
{
    auto large_leg = py::cast(mask->large_leg()).cast<Space::Ptr>();
    auto data = func(block_from_tensor(mask));
    auto basis_perm = rank_basis_perm_masked(
      py::cast(mask->large_leg()).attr("_basis_perm"), *block_backend, data);
    auto small_leg = elementary_from_trivial(
      *large_leg, block_backend->sum_all(data).as_int64(), std::move(basis_perm));
    return { wrap(std::move(data)), std::move(small_leg) };
}

TensorBackend::DataPtr
NoSymmetryBackend::move_to_device(TensorCPtr a, std::string device)
{
    return wrap(block_backend->as_block(py::cast(block_from_tensor(a)), std::nullopt, device));
}

TensorBackend::DataPtr
NoSymmetryBackend::mul(BlockBackend::Scalar a, TensorCPtr b)
{
    return wrap(block_backend->mul(a, block_from_tensor(b)));
}

BlockBackend::Scalar
NoSymmetryBackend::norm(TensorCPtr a)
{
    return block_backend->norm(block_from_tensor(a));
}

TensorBackend::DataPtr
NoSymmetryBackend::outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b)
{
    return wrap(block_backend->tensor_outer(
      block_from_tensor(a), block_from_tensor(b), a->num_codomain_legs()));
}

TensorBackend::DataPtr
NoSymmetryBackend::partial_compose(SymmetricTensorCPtr a,
                                   SymmetricTensorCPtr b,
                                   int64 a_first_leg,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    int64 a_num_codomain = a->num_codomain_legs();
    int64 a_num_legs = a->num_legs;
    int64 b_num_codomain = b->num_codomain_legs();
    int64 b_num_domain = b->num_domain_legs();
    int64 b_num_legs = b->num_legs;
    int64 num_contr_legs;
    int64 num_add_legs;
    std::vector<int64> idcs_b;
    if (a_first_leg < a_num_codomain) {
        num_contr_legs = b_num_domain;
        num_add_legs = b_num_codomain;
        for (int64 i = b_num_legs - 1; i >= b_num_codomain; --i)
            idcs_b.push_back(i);
    } else {
        num_contr_legs = b_num_codomain;
        num_add_legs = b_num_domain;
        for (int64 i = b_num_codomain - 1; i >= 0; --i)
            idcs_b.push_back(i);
    }
    std::vector<int64> idcs_a;
    for (int64 i = a_first_leg; i < a_first_leg + num_contr_legs; ++i)
        idcs_a.push_back(i);
    auto res = block_backend->tdot(block_from_tensor(a), block_from_tensor(b), idcs_a, idcs_b);
    std::vector<int64> perm;
    for (int64 i = 0; i < a_first_leg; ++i)
        perm.push_back(i);
    for (int64 i = 0; i < num_add_legs; ++i)
        perm.push_back(a_num_legs - num_contr_legs + i);
    for (int64 i = a_first_leg; i < a_num_legs - num_contr_legs; ++i)
        perm.push_back(i);
    return wrap(block_backend->permute_axes(res, perm));
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
NoSymmetryBackend::partial_trace(SymmetricTensorCPtr tensor,
                                 std::vector<std::pair<int64, int64>> pairs,
                                 std::vector<std::optional<int64>> levels)
{
    int64 N = tensor->num_legs;
    std::vector<int64> idcs1;
    std::vector<int64> idcs2;
    idcs1.reserve(pairs.size());
    idcs2.reserve(pairs.size());
    for (auto const& [i1, i2] : pairs) {
        idcs1.push_back(i1);
        idcs2.push_back(i2);
    }
    std::vector<bool> used(static_cast<std::size_t>(N), false);
    for (auto i : idcs1)
        used[static_cast<std::size_t>(i)] = true;
    for (auto i : idcs2)
        used[static_cast<std::size_t>(i)] = true;
    std::vector<int64> remaining;
    for (int64 n = 0; n < N; ++n) {
        if (!used[static_cast<std::size_t>(n)])
            remaining.push_back(n);
    }
    auto data = block_backend->trace_partial(block_from_tensor(tensor), idcs1, idcs2, remaining);
    if (remaining.empty()) {
        // Python returns item(data), None, None — keep 0-d block as Data for abstract API.
        return { wrap(std::move(data)), nullptr, nullptr };
    }
    auto codomain_legs = py::cast(tensor->codomain).cast<TensorProduct::Ptr>();
    auto domain_legs = py::cast(tensor->domain).cast<TensorProduct::Ptr>();
    std::vector<Leg::Ptr> codom_factors;
    for (std::size_t n = 0; n < tensor->codomain->factors.size(); ++n) {
        if (std::ranges::find(remaining, static_cast<int64>(n)) != remaining.end())
            codom_factors.push_back(tensor->codomain->factors[n]);
    }
    std::vector<Leg::Ptr> dom_factors;
    for (std::size_t n = 0; n < tensor->domain->factors.size(); ++n) {
        if (std::ranges::find(remaining, N - 1 - static_cast<int64>(n)) != remaining.end())
            dom_factors.push_back(tensor->domain->factors[n]);
    }
    auto sym = tensor->symmetry;
    auto codomain = std::make_shared<TensorProduct>(std::move(codom_factors), sym);
    auto domain = std::make_shared<TensorProduct>(std::move(dom_factors), sym);
    return { wrap(std::move(data)), std::move(codomain), std::move(domain) };
}

TensorBackend::DataPtr
NoSymmetryBackend::permute_legs(TensorCPtr a,
                                std::vector<int64> codomain_idcs,
                                std::vector<int64> domain_idcs,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain,
                                bool mixes_codomain_domain,
                                std::vector<std::optional<int64>> levels,
                                std::vector<std::optional<bool>> bend_right)
{
    std::vector<int64> perm = std::move(codomain_idcs);
    for (auto it = domain_idcs.rbegin(); it != domain_idcs.rend(); ++it)
        perm.push_back(*it);
    return wrap(block_backend->permute_axes(block_from_tensor(a), perm));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
NoSymmetryBackend::qr(SymmetricTensorCPtr a, TensorProduct::Ptr new_co_domain)
{
    auto shape = shape_from_tensor(a);
    int64 n_codom = a->num_codomain_legs();
    std::vector<int64> q_dims(shape.begin(), shape.begin() + n_codom);
    std::vector<int64> r_dims(shape.begin() + n_codom, shape.end());
    auto mat =
      block_backend->reshape(block_from_tensor(a), { prod_i64(q_dims), prod_i64(r_dims) });
    auto [q, r] = block_backend->matrix_qr(mat, false);
    int64 k = block_backend->get_shape(r)[0];
    q_dims.push_back(k);
    std::vector<int64> r_shape;
    r_shape.push_back(k);
    r_shape.insert(r_shape.end(), r_dims.begin(), r_dims.end());
    q = block_backend->reshape(q, q_dims);
    r = block_backend->reshape(r, r_shape);
    return { wrap(std::move(q)), wrap(std::move(r)) };
}

BlockBackend::Scalar
NoSymmetryBackend::reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                         BlockToScalarFn block_func,
                                         ScalarReduceFn /*func*/)
{
    return block_func(block_from_tensor(tensor));
}

TensorBackend::DataPtr
NoSymmetryBackend::scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg)
{
    return wrap(block_backend->scale_axis(block_from_tensor(a), block_from_tensor(b), leg));
}

TensorBackend::DataPtr
NoSymmetryBackend::split_legs(TensorCPtr a,
                              std::vector<int64> leg_idcs,
                              TensorProduct::Ptr new_codomain,
                              TensorProduct::Ptr new_domain)
{
    std::vector<std::vector<int64>> dims;
    std::vector<bool> cstyles;
    dims.reserve(leg_idcs.size());
    cstyles.reserve(leg_idcs.size());
    py::object legs = py::cast(a->legs());
    for (int64 n : leg_idcs) {
        py::object pipe = legs.attr("__getitem__")(n);
        bool cstyle = pipe.attr("combine_cstyle").cast<bool>();
        cstyles.push_back(cstyle);
        auto pipe_legs = pipe.attr("legs").cast<std::vector<py::object>>();
        std::vector<int64> d;
        if (cstyle) {
            for (auto const& s : pipe_legs)
                d.push_back(leg_dim_i64(s));
        } else {
            for (auto it = pipe_legs.rbegin(); it != pipe_legs.rend(); ++it)
                d.push_back(leg_dim_i64(*it));
        }
        dims.push_back(std::move(d));
    }
    return wrap(block_backend->split_legs(block_from_tensor(a), leg_idcs, dims, cstyles));
}

TensorBackend::DataPtr
NoSymmetryBackend::squeeze_legs(TensorCPtr a, std::vector<int64> idcs)
{
    return wrap(block_backend->squeeze_axes(block_from_tensor(a), idcs));
}

bool
NoSymmetryBackend::supports_symmetry(Symmetry::Ptr symmetry)
{
    if (!symmetry)
        return false;
    Symmetry no_sym{ std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() } };
    return symmetry->is_equivalent_to(no_sym);
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, TensorBackend::DataPtr>
NoSymmetryBackend::svd(SymmetricTensorCPtr a,
                       TensorProduct::Ptr new_co_domain,
                       std::optional<std::string> algorithm)
{
    auto shape = shape_from_tensor(a);
    int64 n_codom = a->num_codomain_legs();
    std::vector<int64> u_dims(shape.begin(), shape.begin() + n_codom);
    std::vector<int64> vh_dims(shape.begin() + n_codom, shape.end());
    auto mat =
      block_backend->reshape(block_from_tensor(a), { prod_i64(u_dims), prod_i64(vh_dims) });
    auto [u, s, vh] = block_backend->matrix_svd(mat, algorithm);
    int64 k = block_backend->get_shape(s)[0];
    u_dims.push_back(k);
    std::vector<int64> vh_shape;
    vh_shape.push_back(k);
    vh_shape.insert(vh_shape.end(), vh_dims.begin(), vh_dims.end());
    u = block_backend->reshape(u, u_dims);
    vh = block_backend->reshape(vh, vh_shape);
    return { wrap(std::move(u)), wrap(std::move(s)), wrap(std::move(vh)) };
}

py::object
NoSymmetryBackend::state_tensor_product(BlockBackend::BlockPtr /*state1*/,
                                        BlockBackend::BlockPtr /*state2*/,
                                        Leg::Ptr /*pipe*/)
{
    // --- hints from Python NoSymmetryBackend.state_tensor_product ---
    // TODO clearly define what this should do in tensors.py first!
    // ---
    // TODO clearly define what this should do in tensors.py first!
    throw NotImplemented("state_tensor_product not implemented");
}

TensorBackend::DataPtr
NoSymmetryBackend::to_block_backend(DataPtr data,
                                    std::shared_ptr<BlockBackend> bb,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    return wrap(bb->as_block(py::cast(unwrap(data)), dtype, device));
}

BlockBackend::BlockPtr
NoSymmetryBackend::to_dense_block(TensorCPtr a)
{
    return block_from_tensor(a);
}

BlockBackend::BlockPtr
NoSymmetryBackend::to_dense_block_trivial_sector(TensorCPtr tensor)
{
    // --- hints from Python NoSymmetryBackend.to_dense_block_trivial_sector ---
    // there are no other sectors, so this is essentially the same as to_dense_block.
    // ---
    // there are no other sectors, so this is essentially the same as to_dense_block.
    return block_from_tensor(tensor);
}

TensorBackend::DataPtr
NoSymmetryBackend::to_dtype(TensorCPtr a, Dtype dtype)
{
    return wrap(block_backend->to_dtype(block_from_tensor(a), dtype));
}

BlockBackend::Scalar
NoSymmetryBackend::trace_full(SymmetricTensorCPtr a,
                              std::vector<int64> idcs1,
                              std::vector<int64> idcs2)
{
    // Python NoSymmetryBackend ignores idcs (signature mismatch with abstract base).
    return block_backend->trace_full(block_from_tensor(a));
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr, float64, float64>
NoSymmetryBackend::truncate_singular_values(DiagonalTensorCPtr S,
                                            std::optional<int64> chi_max,
                                            int64 chi_min,
                                            float64 degeneracy_tol,
                                            float64 trunc_cut,
                                            std::optional<float64> svd_min,
                                            bool minimize_error)
{
    py::array S_np = block_backend->to_numpy(block_from_tensor(S)).cast<py::array>();
    auto [keep, err, new_norm] = _truncate_singular_values_selection(
      S_np, py::none(), chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);
    auto mask_data = block_backend->block_from_numpy(keep, Dtype::Bool);
    bool is_dual = true;
    py::object leg = py::cast(S->leg());
    if (py::isinstance<ElementarySpace>(leg))
        is_dual = leg.cast<ElementarySpace::Ptr>()->is_dual;
    // keep.sum() via numpy
    int64 dim = keep.attr("sum")().cast<int64>();
    auto new_leg = ElementarySpace::from_trivial_sector(
      dim, py::cast(S->symmetry).cast<Symmetry::Ptr>(), is_dual);
    return { wrap(std::move(mask_data)), std::move(new_leg), err, new_norm };
}

TensorBackend::DataPtr
NoSymmetryBackend::zero_data(TensorProduct::Ptr codomain,
                             TensorProduct::Ptr domain,
                             Dtype dtype,
                             std::string device,
                             bool /*all_blocks*/)
{
    auto shape = dims_from_legs(conventional_leg_order(codomain, domain));
    return wrap(block_backend->zeros(shape, dtype, device));
}

TensorBackend::DataPtr
NoSymmetryBackend::zero_diagonal_data(TensorProduct::Ptr co_domain,
                                      Dtype dtype,
                                      std::string device)
{
    return wrap(block_backend->zeros({ space_dim_i64(*co_domain) }, dtype, device));
}

TensorBackend::DataPtr
NoSymmetryBackend::zero_mask_data(Space::Ptr large_leg, std::string device)
{
    return wrap(block_backend->zeros({ space_dim_i64(*large_leg) }, Dtype::Bool, device));
}

} // namespace lasyten
