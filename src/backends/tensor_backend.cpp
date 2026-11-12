#include <lasyten/backends/tensor_backend.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <lasyten/tools/misc.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lasyten {

namespace {

std::vector<float64>
f64_array_to_vector(py::array arr)
{
    py::array_t<float64, py::array::c_style | py::array::forcecast> flat =
      py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(arr).reshape({ -1 });
    auto buf = flat.unchecked<1>();
    std::vector<float64> out(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = buf(i);
    }
    return out;
}

py::array
bool_vector_to_array(std::vector<bool> const& v)
{
    py::array_t<bool> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = v[i];
    }
    return arr;
}

std::string
backend_type_name(TensorBackend const& self)
{
    try {
        py::object py_self = py::cast(const_cast<TensorBackend*>(&self));
        return py::str(py::type::of(py_self).attr("__name__")).cast<std::string>();
    } catch (py::cast_error const&) {
        return "TensorBackend";
    } catch (py::error_already_set const&) {
        return "TensorBackend";
    }
}

bool
legs_equal(std::vector<Leg::Ptr> const& a, std::vector<Leg::Ptr> const& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i])
            continue;
        if (!a[i] || !b[i] || !(a[i] == b[i]))
            return false;
    }
    return true;
}

} // namespace

TensorBackend::TensorBackend(std::shared_ptr<BlockBackend> block_backend_)
  : block_backend(std::move(block_backend_))
{
}

std::string
TensorBackend::__repr__() const
{
    std::ostringstream oss;
    oss << backend_type_name(*this) << '(';
    try {
        oss << py::repr(py::cast(block_backend)).cast<std::string>();
    } catch (...) {
        oss << "BlockBackend";
    }
    oss << ')';
    return oss.str();
}

std::string
TensorBackend::__str__() const
{
    return __repr__();
}

bool
TensorBackend::operator==(TensorBackend const& other) const
{
    if (this == &other)
        return true;
    if (backend_type_name(*this) != backend_type_name(other))
        return false;
    auto const& a = block_backend;
    auto const& b = other.block_backend;
    if (a.get() == b.get())
        return true;
    if (!a || !b)
        return false;
    return *a == *b;
}

BlockBackend::Scalar
TensorBackend::item(TensorCPtr a)
{
    DataPtr data_ptr;
    if (auto st = std::dynamic_pointer_cast<const SymmetricTensor>(a))
        data_ptr = st->data;
    else if (auto m = std::dynamic_pointer_cast<const Mask>(a))
        data_ptr = m->data;
    else
        throw std::invalid_argument("TensorBackend::item: expected SymmetricTensor or Mask");
    return data_item(data_ptr);
}

void
TensorBackend::test_tensor_sanity(TensorCPtr a, bool /*is_diagonal*/)
{
    DataPtr data_ptr;
    if (auto st = std::dynamic_pointer_cast<const SymmetricTensor>(a))
        data_ptr = st->data;
    else if (auto m = std::dynamic_pointer_cast<const Mask>(a))
        data_ptr = m->data;
    else
        throw std::invalid_argument(
          "TensorBackend::test_tensor_sanity: expected SymmetricTensor or Mask");
    if (!data_ptr || !is_correct_data_type(data_ptr))
        throw std::runtime_error("wrong tensor data type");
}

void
TensorBackend::test_mask_sanity(MaskCPtr a)
{
    if (!a->data || !is_correct_data_type(a->data))
        throw std::runtime_error("wrong tensor data type");
}

Leg::Ptr
TensorBackend::make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, Leg::Ptr pipe)
{
    if (pipe) {
        auto lp = pipe.as_leg_pipe();
        assert(lp);
        assert(lp->combine_cstyle == !is_dual);
        assert(lp->is_dual == is_dual);
        assert(legs_equal(lp->legs, legs));
        return lp;
    }
    return std::make_shared<LegPipe>(std::move(legs), is_dual, /*combine_cstyle=*/!is_dual);
}

std::tuple<py::array, float64, float64>
TensorBackend::_truncate_singular_values_selection(py::array S,
                                                   py::object qdims,
                                                   std::optional<int64> chi_max,
                                                   int64 chi_min,
                                                   float64 degeneracy_tol,
                                                   float64 trunc_cut,
                                                   std::optional<float64> svd_min,
                                                   bool minimize_error)
{
    // --- hints from Python TensorBackend._truncate_singular_values_selection ---
    // contributions ``err[i] = d[i] * S[i] ** 2`` to the error, if S[i] would be truncated.
    // ---
    auto S_v = f64_array_to_vector(S);
    std::vector<float64> marginal_errs(S_v.size());
    if (qdims.is_none()) {
        for (std::size_t i = 0; i < S_v.size(); ++i) {
            marginal_errs[i] = S_v[i] * S_v[i];
        }
    } else {
        auto q = f64_array_to_vector(qdims.cast<py::array>());
        if (q.size() != S_v.size()) {
            throw std::invalid_argument(
              "_truncate_singular_values_selection: qdims size mismatch");
        }
        for (std::size_t i = 0; i < S_v.size(); ++i) {
            marginal_errs[i] = q[i] * S_v[i] * S_v[i];
        }
    }

    std::size_t const n = S_v.size();
    std::vector<std::size_t> piv(n);
    std::iota(piv.begin(), piv.end(), std::size_t{ 0 });
    std::stable_sort(piv.begin(), piv.end(), [&](std::size_t a, std::size_t b) {
        return marginal_errs[a] < marginal_errs[b];
    });

    std::vector<float64> S_sorted(n);
    std::vector<float64> err_sorted(n);
    for (std::size_t i = 0; i < n; ++i) {
        S_sorted[i] = S_v[piv[i]];
        err_sorted[i] = marginal_errs[piv[i]];
    }

    std::vector<float64> logS(n);
    for (std::size_t i = 0; i < n; ++i) {
        float64 v = S_sorted[i] <= 1.0e-100 ? 1.0e-100 : S_sorted[i];
        logS[i] = std::log(v);
    }

    std::vector<bool> good(n, true);

    if (chi_max.has_value() && static_cast<std::size_t>(*chi_max) < n) {
        std::vector<bool> good2(n, false);
        for (std::size_t i = n - static_cast<std::size_t>(*chi_max); i < n; ++i) {
            good2[i] = true;
        }
        good = combine_constraints(good, good2, "chi_max");
    }

    if (chi_min > 1) {
        std::vector<bool> good2(n, true);
        std::size_t start = n >= static_cast<std::size_t>(chi_min - 1)
                              ? n - static_cast<std::size_t>(chi_min - 1)
                              : 0;
        for (std::size_t i = start; i < n; ++i) {
            good2[i] = false;
        }
        good = combine_constraints(good, good2, "chi_min");
    }

    if (degeneracy_tol > 0) {
        std::vector<bool> good2(n, true);
        for (std::size_t i = 1; i < n; ++i) {
            good2[i] = (logS[i] - logS[i - 1]) >= degeneracy_tol;
        }
        good = combine_constraints(good, good2, "degeneracy_tol");
    }

    if (svd_min.has_value()) {
        std::vector<bool> good2(n);
        for (std::size_t i = 0; i < n; ++i) {
            good2[i] = S_sorted[i] >= *svd_min;
        }
        good = combine_constraints(good, good2, "svd_min");
    }

    {
        std::vector<bool> good2(n);
        float64 csum = 0.0;
        float64 const cut2 = trunc_cut * trunc_cut;
        for (std::size_t i = 0; i < n; ++i) {
            csum += err_sorted[i];
            good2[i] = csum > cut2;
        }
        good = combine_constraints(good, good2, "trunc_cut");
    }

    std::vector<std::size_t> nonzero;
    for (std::size_t i = 0; i < n; ++i) {
        if (good[i]) {
            nonzero.push_back(i);
        }
    }
    if (nonzero.empty()) {
        throw std::runtime_error("_truncate_singular_values_selection: no valid cut");
    }
    std::size_t cut = minimize_error ? nonzero.front() : nonzero.back();

    float64 err_sum = 0.0;
    for (std::size_t i = 0; i < cut; ++i) {
        err_sum += err_sorted[i];
    }
    float64 norm_sum = 0.0;
    for (std::size_t i = cut; i < n; ++i) {
        norm_sum += err_sorted[i];
    }
    float64 err = std::sqrt(err_sum);
    float64 new_norm = std::sqrt(norm_sum);

    std::vector<bool> mask(n, false);
    for (std::size_t i = cut; i < n; ++i) {
        mask[piv[i]] = true;
    }
    return { bool_vector_to_array(mask), err, new_norm };
}

bool
TensorBackend::is_real(TensorCPtr a)
{
    // --- hints from Python TensorBackend.is_real ---
    // FusionTree backend might implement this differently.
    // ---
    // FusionTree backend might implement this differently.
    return dtype::is_real(a->dtype);
}

void
TensorBackend::save_hdf5(lasyten::hdf5::Saver& saver,
                         HighFive::Group& /*h5gr*/,
                         std::string subpath)
{
    hdf5_export::save_block_backend(saver, subpath + "block_backend", block_backend);
}

TensorBackend::Ptr
TensorBackend::from_hdf5(py::object cls,
                         lasyten::hdf5::Loader& loader,
                         HighFive::Group& h5gr,
                         std::string subpath)
{
    auto block_backend = hdf5_export::load_block_backend(loader, subpath + "block_backend");
    py::object obj = cls(block_backend);
    auto ptr = obj.cast<Ptr>();
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(ptr));
    return ptr;
}

std::vector<Leg::Ptr>
conventional_leg_order(TensorProduct::Ptr codomain, TensorProduct::Ptr domain)
{
    std::vector<Leg::Ptr> out;
    out.reserve(codomain->factors.size() + domain->factors.size());
    for (auto const& f : codomain->factors)
        out.push_back(f);
    for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it)
        out.push_back(*it);
    return out;
}

std::vector<Leg::Ptr>
conventional_leg_order(py::object tensor_or_codomain, py::object domain)
{
    TensorProduct::Ptr codomain_ptr;
    TensorProduct::Ptr domain_ptr;
    if (domain.is_none()) {
        codomain_ptr = tensor_or_codomain.attr("codomain").cast<TensorProduct::Ptr>();
        domain_ptr = tensor_or_codomain.attr("domain").cast<TensorProduct::Ptr>();
    } else {
        codomain_ptr = tensor_or_codomain.cast<TensorProduct::Ptr>();
        domain_ptr = domain.cast<TensorProduct::Ptr>();
    }
    return conventional_leg_order(codomain_ptr, domain_ptr);
}

std::vector<Leg::Ptr>
conventional_leg_order(TensorCPtr tensor)
{
    return conventional_leg_order(tensor->codomain, tensor->domain);
}

TensorBackend::Ptr
get_same_backend(const std::vector<py::object>& objs, std::string error_msg)
{
    if (objs.empty())
        throw std::invalid_argument("Need at least one tensor");
    for (std::size_t i = 0; i < objs.size(); ++i) {
        if (objs[i].is_none()) {
            throw std::invalid_argument(std::format(
              "get_same_backend() requires non-null Tensor arguments (null at index {})", i));
        }
    }
    TensorBackend::Ptr backend = objs[0].attr("backend").cast<TensorBackend::Ptr>();
    for (std::size_t i = 1; i < objs.size(); ++i) {
        TensorBackend::Ptr other = objs[i].attr("backend").cast<TensorBackend::Ptr>();
        if (!backend || !other || !(backend == other))
            throw std::invalid_argument(std::move(error_msg));
    }
    return backend;
}

TensorBackend::Ptr
get_same_backend(const std::vector<TensorCPtr>& objs, std::string error_msg)
{
    if (objs.empty())
        throw std::invalid_argument("Need at least one tensor");
    for (std::size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i]) {
            throw std::invalid_argument(std::format(
              "get_same_backend() requires non-null Tensor arguments (null at index {})", i));
        }
    }
    TensorBackend::Ptr backend = objs[0]->backend;
    for (std::size_t i = 1; i < objs.size(); ++i) {
        TensorBackend::Ptr const& other = objs[i]->backend;
        if (!backend || !other || !(backend == other))
            throw std::invalid_argument(std::move(error_msg));
    }
    return backend;
}

} // namespace lasyten
