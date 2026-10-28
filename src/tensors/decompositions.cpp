#include <lasyten/tensors/decompositions.h>

#include <lasyten/backends/no_symmetry.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/charged_tensor.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/helpers.h>
#include <lasyten/tensors/hidden_leg_tensor.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_elementwise.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace lasyten {

namespace {

bool
is_Mask(py::object obj)
{
    return py::isinstance<Mask>(obj);
}

bool
is_DiagonalTensor(py::object obj)
{
    return py::isinstance<DiagonalTensor>(obj);
}

bool
is_Identity(py::object obj)
{
    return py::isinstance<Identity>(obj);
}

bool
is_SymmetricTensor(py::object obj)
{
    return py::isinstance<SymmetricTensor>(obj);
}

bool
is_ChargedTensor(py::object obj)
{
    return py::isinstance<ChargedTensor>(obj);
}

bool
py_eq(py::object a, py::object b)
{
    py::object eq = a.attr("__eq__")(b);
    if (eq.is(py::reinterpret_borrow<py::object>(Py_NotImplemented))) {
        return false;
    }
    return eq.cast<bool>();
}

[[nodiscard]] Space::Ptr
as_space_leg(py::object leg)
{
    if (py::isinstance<LegPipe>(leg)) {
        throw std::invalid_argument("DiagonalTensor / Mask is not defined on LegPipes.");
    }
    return leg.cast<Space::Ptr>();
}

py::object
make_python_symmetric_tensor(TensorBackend::DataPtr data,
                             py::object codomain,
                             py::object domain,
                             TensorBackend::Ptr backend,
                             py::object labels)
{
    auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
    return py::cast(std::make_shared<SymmetricTensor>(
      std::move(data), init.codomain, init.domain, init.backend, init.symmetry, init.labels));
}

py::object
make_python_diagonal_tensor(TensorBackend::DataPtr data,
                            py::object leg,
                            TensorBackend::Ptr backend,
                            py::object labels)
{
    auto init = parse_tensor_init(
      py::make_tuple(leg), py::make_tuple(leg), std::move(backend), labels, true);
    return py::cast(std::make_shared<DiagonalTensor>(
      std::move(data), as_space_leg(leg), init.backend, init.symmetry, init.labels));
}

py::object
make_python_mask(TensorBackend::DataPtr data,
                 py::object space_in,
                 py::object space_out,
                 bool is_projection,
                 TensorBackend::Ptr backend,
                 py::object labels)
{
    auto init = parse_tensor_init(
      py::make_tuple(space_out), py::make_tuple(space_in), std::move(backend), labels);
    auto device_s = init.backend->get_device_from_data(data);
    return py::cast(std::make_shared<Mask>(std::move(data),
                                           as_space_leg(space_in),
                                           as_space_leg(space_out),
                                           is_projection,
                                           init.backend,
                                           init.symmetry,
                                           init.labels,
                                           std::move(device_s)));
}

py::object
make_python_charged_tensor(py::object invariant_part, py::object charged_state)
{
    return py::cast(std::make_shared<ChargedTensor>(invariant_part.cast<SymmetricTensor::Ptr>(),
                                                    charged_state.cast<BlockBackend::BlockPtr>()));
}

py::object
labels_to_py(LegLabelList const& labels)
{
    py::list out;
    for (auto const& lab : labels) {
        out.append(lab); // empty string = unset
    }
    return out;
}

LegLabelList
leg_labels_from_py(py::object seq)
{
    LegLabelList out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        if (item.is_none()) {
            out.push_back("");
        } else {
            out.push_back(item.cast<std::string>());
        }
    }
    return out;
}

py::object
leg_label_to_py(std::string const& lab)
{
    return py::cast(lab);
}

[[nodiscard]] std::string
py_label_as_string(py::handle obj)
{
    if (obj.is_none()) {
        return "";
    }
    return obj.cast<std::string>();
}

template<typename... Args>
py::list
py_list(Args&&... args)
{
    py::list out;
    (out.append(std::forward<Args>(args)), ...);
    return out;
}

py::object
nested_labels_codomain_domain(py::object codomain_labels, py::object domain_one_label)
{
    py::list out;
    out.append(codomain_labels);
    out.append(py_list(domain_one_label));
    return out;
}

py::object
nested_labels_one_and_domain(py::object one_label, py::object domain_labels)
{
    py::list out;
    out.append(py_list(one_label));
    out.append(domain_labels);
    return out;
}

char const*
charge_leg_label()
{
    return ChargedTensor::_CHARGE_LEG_LABEL;
}

bool
is_inf_number(py::object n)
{
    try {
        if (py::isinstance<py::float_>(n) || py::isinstance<py::int_>(n)) {
            return std::isinf(n.cast<float64>());
        }
        if (py::hasattr(n, "__float__")) {
            return std::isinf(n.cast<float64>());
        }
    } catch (...) {
    }
    return false;
}

std::string
same_device2(py::object t1, py::object t2, std::string const& error_msg = "Incompatible devices.")
{
    std::string device = t1.attr("device").cast<std::string>();
    if (t2.attr("device").cast<std::string>() != device) {
        throw std::invalid_argument(error_msg);
    }
    return device;
}

std::pair<std::string, std::string>
parse_eigenvalue_labels(py::object new_labels)
{
    py::object labels_iter = to_iterable(new_labels);
    py::ssize_t nlab = py::len(labels_iter);
    std::string b;
    std::string c;
    if (nlab == 0) {
        b = "";
        c = "";
    } else if (nlab == 1) {
        auto lab0 = labels_iter.attr("__getitem__")(0);
        b = py_label_as_string(lab0);
        c = _dual_leg_label(b);
    } else if (nlab == 2) {
        auto lab0 = labels_iter.attr("__getitem__")(0);
        auto lab1 = labels_iter.attr("__getitem__")(1);
        b = py_label_as_string(lab0);
        c = py_label_as_string(lab1);
    } else {
        throw std::invalid_argument(std::format(
          "Expected 0, 1 or 2 new_labels for eigenvalues. Got {}.", static_cast<int>(nlab)));
    }
    return { std::move(b), std::move(c) };
}

struct EigenPrepared
{
    py::object tensor;
    TensorBackend::Ptr backend;
    bool combined = false;
};

EigenPrepared
prepare_square_eigen(py::object tensor, bool new_leg_dual, char const* name)
{
    if (!py_eq(tensor.attr("domain"), tensor.attr("codomain"))) {
        throw std::invalid_argument(std::string(name) + " requires matching domain and codomain");
    }
    if (is_ChargedTensor(tensor)) {
        throw NotImplemented(std::string(name) + std::string(" for ChargedTensor"));
    }
    tensor = tensor.attr("as_SymmetricTensor")();
    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    bool combined = false;
    if (!backend->can_decompose_tensors()) {
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        int64 n_legs = tensor.attr("num_legs").cast<int64>();
        std::vector<LegRef> cod_idcs;
        std::vector<LegRef> dom_idcs;
        for (int64 i = 0; i < n_cod; ++i) {
            cod_idcs.emplace_back(i);
        }
        for (int64 i = n_cod; i < n_legs; ++i) {
            dom_idcs.emplace_back(i);
        }
        tensor = py::cast(
          combine_legs(tensor.cast<TensorCPtr>(),
                       { std::move(cod_idcs), std::move(dom_idcs) },
                       PipeDualities{ std::vector<bool>{ new_leg_dual, !new_leg_dual } }));
        backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        combined = true;
    }
    return { std::move(tensor), std::move(backend), combined };
}

std::optional<std::string>
sort_opt_from_py(py::object sort)
{
    if (sort.is_none()) {
        return std::nullopt;
    }
    return sort.cast<std::string>();
}

} // namespace

py::object apply_mask_DiagonalTensor_py(py::object tensor, py::object mask);
std::tuple<py::object, py::object> eigh_py(py::object tensor,
                                           py::object new_labels,
                                           bool new_leg_dual,
                                           py::object sort);
py::object entropy_py(py::object p, py::object n);
std::tuple<py::object, py::object> qr_py(py::object tensor,
                                         py::object new_labels,
                                         bool new_leg_dual,
                                         bool charge_leg_top);
std::tuple<py::object, py::object> lq_py(py::object tensor,
                                         py::object new_labels,
                                         bool new_leg_dual,
                                         bool charge_leg_top);
std::tuple<py::object, py::object, py::object> svd_py(py::object tensor,
                                                      py::object new_labels,
                                                      bool new_leg_dual,
                                                      bool charge_leg_top,
                                                      py::object algorithm);
std::tuple<py::object, py::object, py::object> svd_apply_mask_py(py::object U,
                                                                 py::object S,
                                                                 py::object Vh,
                                                                 py::object mask);
std::tuple<py::object, float64, float64> truncate_singular_values_py(
  py::object S,
  std::optional<int64> chi_max,
  int64 chi_min,
  float64 degeneracy_tol,
  float64 trunc_cut,
  float64 svd_min,
  bool minimize_error = true,
  py::object mask_labels = py::none());
std::tuple<py::object, py::object, py::object, float64, float64> truncated_svd_py(
  py::object tensor,
  py::object new_labels,
  bool new_leg_dual,
  bool charge_leg_top,
  py::object algorithm,
  std::optional<float64> normalize_to,
  std::optional<int64> chi_max,
  int64 chi_min,
  float64 degeneracy_tol,
  float64 trunc_cut,
  float64 svd_min);

py::object
move_charge_leg(py::object tensor_part,
                std::string const& which,
                std::optional<int64> cpos,
                std::optional<int64> dpos)
{
    return py::cast(move_leg(
      tensor_part.cast<TensorCPtr>(), which, cpos, dpos, std::nullopt, BendRight{ false }));
}

py::object
split_one_leg(py::object tensor, int64 idx)
{
    return py::cast(split_legs(tensor.cast<TensorCPtr>(), std::vector<LegRef>{ idx }));
}

/// Labels of the hidden legs of `tensor`, without the ``!`` prefix.
/// Empty unless `tensor` is a `HiddenLegTensor`.
std::vector<std::string>
stripped_hidden_labels(py::object tensor)
{
    std::vector<std::string> res;
    for (auto const& lab : leg_labels_from_py(tensor.attr("labels").attr("as_list")())) {
        if (HiddenLegTensor::is_hidden_leg_label(lab)) {
            res.push_back(HiddenLegTensor::strip_hidden_prefix(lab));
        }
    }
    return res;
}

/// Hide the legs of a decomposition `factor` that were hidden on the decomposed tensor.
///
/// Decompositions act on the unhidden tensor (see `_decomposition_prepare`), such that the
/// hidden legs end up as ordinary legs on one of the factors. The new leg at `new_leg_idx` is
/// never hidden, even if its label coincides with one of the `hidden` labels.
py::object
rehide_factor(py::object factor, std::vector<std::string> const& hidden, int64 new_leg_idx)
{
    if (hidden.empty()) {
        return factor;
    }
    auto labs = leg_labels_from_py(factor.attr("labels").attr("as_list")());
    auto num_legs = static_cast<int64>(labs.size());
    if (new_leg_idx < 0) {
        new_leg_idx += num_legs;
    }
    std::vector<std::variant<int64, std::string>> which;
    for (int64 i = 0; i < num_legs; ++i) {
        auto const& lab = labs[static_cast<std::size_t>(i)];
        if (i == new_leg_idx || lab.empty()) {
            continue;
        }
        if (std::find(hidden.begin(), hidden.end(), lab) != hidden.end()) {
            which.emplace_back(i);
        }
    }
    if (which.empty()) {
        return factor;
    }
    return py::cast(HiddenLegTensor::from_tensor(factor.cast<TensorPtr>(), std::move(which)));
}

py::object
apply_mask_DiagonalTensor_py(py::object tensor, py::object mask)
{
    same_device2(tensor, mask);
    if (!mask.attr("is_projection").cast<bool>()) {
        throw std::invalid_argument("mask must be a projection");
    }
    if (!py_eq(mask.attr("large_leg"), tensor.attr("leg"))) {
        throw std::invalid_argument("mask.large_leg must match the tensor leg");
    }
    auto backend = get_same_backend({ tensor, mask });
    if (is_Identity(tensor)) {
        auto init = parse_tensor_init(py::make_tuple(mask.attr("small_leg")),
                                      py::make_tuple(mask.attr("small_leg")),
                                      backend,
                                      tensor.attr("labels").attr("as_list")(),
                                      true);
        auto dt = SymmetricTensor::_parse_default_dtype(std::nullopt, init.symmetry);
        if (!dt.has_value()) {
            dt = Dtype::Float64;
        }
        std::string device_s = init.backend->block_backend->default_device;
        return py::cast(std::make_shared<Identity>(as_space_leg(mask.attr("small_leg")),
                                                   init.backend,
                                                   init.symmetry,
                                                   init.labels,
                                                   *dt,
                                                   std::move(device_s)));
    }
    auto data = backend->apply_mask_to_DiagonalTensor(tensor.cast<DiagonalTensorCPtr>(),
                                                      mask.cast<MaskCPtr>());
    return make_python_diagonal_tensor(
      std::move(data), mask.attr("small_leg"), backend, tensor.attr("labels").attr("as_list")());
}

std::tuple<py::object, py::object>
eigh_py(py::object tensor, py::object new_labels, bool new_leg_dual, py::object sort)
{
    // --- hints from Python eigh_py ---
    // do not define decompositions for ChargedTensors.
    // If the backend requires it, combine legs first
    // first, compute a decomposition where the new leg is a ket space
    // undo the combine
    // if required, flip the leg duality
    // ---
    py::object labels_iter = to_iterable(new_labels);
    py::ssize_t nlab = py::len(labels_iter);
    std::string a;
    std::string b;
    std::string c;
    if (nlab == 1) {
        a = c = py_label_as_string(labels_iter.attr("__getitem__")(0));
        b = _dual_leg_label(a);
    } else if (nlab == 2) {
        a = c = py_label_as_string(labels_iter.attr("__getitem__")(0));
        b = py_label_as_string(labels_iter.attr("__getitem__")(1));
    } else if (nlab == 3) {
        auto lab0 = labels_iter.attr("__getitem__")(0);
        auto lab1 = labels_iter.attr("__getitem__")(1);
        auto lab2 = labels_iter.attr("__getitem__")(2);
        a = py_label_as_string(lab0);
        b = py_label_as_string(lab1);
        c = py_label_as_string(lab2);
    } else {
        throw std::invalid_argument(
          std::format("Expected 1, 2 or 3 new_labels. Got {}.", static_cast<int>(nlab)));
    }

    if (!py_eq(tensor.attr("domain"), tensor.attr("codomain"))) {
        throw std::invalid_argument("eigh requires matching domain and codomain");
    }
    if (is_ChargedTensor(tensor)) {
        // do not define decompositions for ChargedTensors.
        throw NotImplemented("eigh for ChargedTensor");
    }
    if (is_DiagonalTensor(tensor)) {
        std::string cod_lab =
          tensor.attr("codomain_labels").attr("__getitem__")(0).cast<std::string>();
        auto leg_sp = as_space_leg(tensor.attr("leg"));
        auto V =
          py::cast(SymmetricTensor::from_eye(std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{
                                               std::dynamic_pointer_cast<Leg>(leg_sp) }),
                                             tensor.attr("backend").cast<TensorBackend::Ptr>(),
                                             LegLabelList{ cod_lab, a },
                                             tensor.attr("dtype").cast<Dtype>(),
                                             tensor.attr("device").cast<std::string>()));
        py::object W = tensor.attr("as_DiagonalTensor")(py::arg("guarantee_copy") = true);
        W.attr("labels").attr("set")(py_list(leg_label_to_py(b), leg_label_to_py(c)));
        return { W, V };
    }
    tensor = tensor.attr("as_SymmetricTensor")();

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    // If the backend requires it, combine legs first
    if (!backend->can_decompose_tensors()) {
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        int64 n_legs = tensor.attr("num_legs").cast<int64>();
        std::vector<LegRef> cod_idcs;
        std::vector<LegRef> dom_idcs;
        for (int64 i = 0; i < n_cod; ++i) {
            cod_idcs.emplace_back(i);
        }
        for (int64 i = n_cod; i < n_legs; ++i) {
            dom_idcs.emplace_back(i);
        }
        tensor = py::cast(
          combine_legs(tensor.cast<TensorCPtr>(),
                       { std::move(cod_idcs), std::move(dom_idcs) },
                       PipeDualities{ std::vector<bool>{ new_leg_dual, !new_leg_dual } }));
        backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    }

    std::optional<std::string> sort_opt;
    if (!sort.is_none()) {
        sort_opt = sort.cast<std::string>();
    }
    // first, compute a decomposition where the new leg is a ket space
    auto [w_data, v_data, new_leg] =
      backend->eigh(tensor.cast<SymmetricTensorCPtr>(), new_leg_dual, sort_opt);
    py::object W = make_python_diagonal_tensor(std::move(w_data),
                                               py::cast(new_leg),
                                               backend,
                                               py_list(leg_label_to_py(b), leg_label_to_py(c)));
    py::object V = make_python_symmetric_tensor(
      std::move(v_data),
      tensor.attr("codomain"),
      py_list(py::cast(new_leg)),
      backend,
      nested_labels_codomain_domain(tensor.attr("codomain_labels"), leg_label_to_py(a)));

    // undo the combine
    if (!backend->can_decompose_tensors()) {
        V = py::cast(split_legs(V.cast<TensorCPtr>(), std::vector<LegRef>{ int64{ 0 } }));
    }

    // if required, flip the leg duality
    if (new_leg_dual != new_leg->is_dual) {
        throw NotImplemented("eigh flip new_leg duality");
    }

    return { W, V };
}

std::tuple<py::object, py::object>
eig_py(py::object tensor, py::object new_labels, bool new_leg_dual, py::object sort)
{
    py::object labels_iter = to_iterable(new_labels);
    py::ssize_t nlab = py::len(labels_iter);
    std::string a;
    std::string b;
    std::string c;
    if (nlab == 1) {
        a = c = py_label_as_string(labels_iter.attr("__getitem__")(0));
        b = _dual_leg_label(a);
    } else if (nlab == 2) {
        a = c = py_label_as_string(labels_iter.attr("__getitem__")(0));
        b = py_label_as_string(labels_iter.attr("__getitem__")(1));
    } else if (nlab == 3) {
        auto lab0 = labels_iter.attr("__getitem__")(0);
        auto lab1 = labels_iter.attr("__getitem__")(1);
        auto lab2 = labels_iter.attr("__getitem__")(2);
        a = py_label_as_string(lab0);
        b = py_label_as_string(lab1);
        c = py_label_as_string(lab2);
    } else {
        throw std::invalid_argument(
          std::format("Expected 1, 2 or 3 new_labels. Got {}.", static_cast<int>(nlab)));
    }

    if (is_ChargedTensor(tensor)) {
        throw NotImplemented("eig for ChargedTensor");
    }
    if (!py_eq(tensor.attr("domain"), tensor.attr("codomain"))) {
        throw std::invalid_argument("eig requires matching domain and codomain");
    }
    if (is_DiagonalTensor(tensor)) {
        std::string cod_lab =
          tensor.attr("codomain_labels").attr("__getitem__")(0).cast<std::string>();
        auto leg_sp = as_space_leg(tensor.attr("leg"));
        auto V =
          py::cast(SymmetricTensor::from_eye(std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{
                                               std::dynamic_pointer_cast<Leg>(leg_sp) }),
                                             tensor.attr("backend").cast<TensorBackend::Ptr>(),
                                             LegLabelList{ cod_lab, a },
                                             tensor.attr("dtype").cast<Dtype>(),
                                             tensor.attr("device").cast<std::string>()));
        py::object W = tensor.attr("as_DiagonalTensor")(py::arg("guarantee_copy") = true);
        W.attr("labels").attr("set")(py_list(leg_label_to_py(b), leg_label_to_py(c)));
        return { W, V };
    }

    auto prep = prepare_square_eigen(tensor, new_leg_dual, "eig");
    auto [w_data, v_data, new_leg] = prep.backend->eig(
      prep.tensor.cast<SymmetricTensorCPtr>(), new_leg_dual, sort_opt_from_py(sort));
    py::object W = make_python_diagonal_tensor(std::move(w_data),
                                               py::cast(new_leg),
                                               prep.backend,
                                               py_list(leg_label_to_py(b), leg_label_to_py(c)));
    py::object V = make_python_symmetric_tensor(
      std::move(v_data),
      prep.tensor.attr("codomain"),
      py_list(py::cast(new_leg)),
      prep.backend,
      nested_labels_codomain_domain(prep.tensor.attr("codomain_labels"), leg_label_to_py(a)));
    if (prep.combined) {
        V = py::cast(split_legs(V.cast<TensorCPtr>(), std::vector<LegRef>{ int64{ 0 } }));
    }
    if (new_leg_dual != new_leg->is_dual) {
        throw NotImplemented("eig flip new_leg duality");
    }
    return { W, V };
}

py::object
eigvalsh_py(py::object tensor, py::object new_labels, bool new_leg_dual, py::object sort)
{
    auto [b, c] = parse_eigenvalue_labels(new_labels);
    if (is_ChargedTensor(tensor)) {
        throw NotImplemented("eigvalsh for ChargedTensor");
    }
    if (!py_eq(tensor.attr("domain"), tensor.attr("codomain"))) {
        throw std::invalid_argument("eigvalsh requires matching domain and codomain");
    }
    if (is_DiagonalTensor(tensor)) {
        py::object W = tensor.attr("as_DiagonalTensor")(py::arg("guarantee_copy") = true);
        W.attr("labels").attr("set")(py_list(leg_label_to_py(b), leg_label_to_py(c)));
        return W;
    }
    auto prep = prepare_square_eigen(tensor, new_leg_dual, "eigvalsh");
    auto [w_data, new_leg] = prep.backend->eigvalsh(
      prep.tensor.cast<SymmetricTensorCPtr>(), new_leg_dual, sort_opt_from_py(sort));
    if (new_leg_dual != new_leg->is_dual) {
        throw NotImplemented("eigvalsh flip new_leg duality");
    }
    return make_python_diagonal_tensor(std::move(w_data),
                                       py::cast(new_leg),
                                       prep.backend,
                                       py_list(leg_label_to_py(b), leg_label_to_py(c)));
}

py::object
eigvals_py(py::object tensor, py::object new_labels, bool new_leg_dual, py::object sort)
{
    auto [b, c] = parse_eigenvalue_labels(new_labels);
    if (is_ChargedTensor(tensor)) {
        throw NotImplemented("eigvals for ChargedTensor");
    }
    if (!py_eq(tensor.attr("domain"), tensor.attr("codomain"))) {
        throw std::invalid_argument("eigvals requires matching domain and codomain");
    }
    if (is_DiagonalTensor(tensor)) {
        py::object W = tensor.attr("as_DiagonalTensor")(py::arg("guarantee_copy") = true);
        W.attr("labels").attr("set")(py_list(leg_label_to_py(b), leg_label_to_py(c)));
        return W;
    }
    auto prep = prepare_square_eigen(tensor, new_leg_dual, "eigvals");
    auto [w_data, new_leg] = prep.backend->eigvals(
      prep.tensor.cast<SymmetricTensorCPtr>(), new_leg_dual, sort_opt_from_py(sort));
    if (new_leg_dual != new_leg->is_dual) {
        throw NotImplemented("eigvals flip new_leg duality");
    }
    return make_python_diagonal_tensor(std::move(w_data),
                                       py::cast(new_leg),
                                       prep.backend,
                                       py_list(leg_label_to_py(b), leg_label_to_py(c)));
}

py::object
entropy_py(py::object p, py::object n)
{
    // --- hints from Python entropy_py ---
    // for stability of log
    // ---
    if (is_Identity(p)) {
        throw py::type_error(
          "entropy_py does not support Identity. It is never a normalized distribution.");
    }
    if (is_DiagonalTensor(p)) {
        float64 n_f = 1.;
        if (is_inf_number(n)) {
            n_f = std::numeric_limits<float64>::infinity();
        } else {
            n_f = n.cast<float64>();
        }
        return py::cast(entropy(p.cast<DiagonalTensorCPtr>(), n_f));
    }
    // else: sequence of floats (kept for internal callers; pybind has its own path)
    auto arr = py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(p);
    if (!arr) {
        throw std::invalid_argument("entropy_py: expected array-like of floats");
    }
    auto info = arr.request();
    auto const* ptr = static_cast<float64 const*>(info.ptr);
    std::vector<float64> vals;
    vals.reserve(static_cast<std::size_t>(info.size));
    for (py::ssize_t i = 0; i < info.size; ++i) {
        if (ptr[i] > 1e-30) { // for stability of log
            vals.push_back(ptr[i]);
        }
    }
    if (vals.empty()) {
        return py::float_(0.);
    }
    if (py_eq(n, py::int_(1))) {
        float64 s = 0.;
        for (float64 x : vals) {
            s += std::log(x) * x;
        }
        return py::float_(-s);
    }
    if (is_inf_number(n)) {
        float64 mx = *std::max_element(vals.begin(), vals.end());
        return py::float_(-std::log(mx));
    }
    float64 n_f = n.cast<float64>();
    float64 sum = 0.;
    for (float64 x : vals) {
        sum += std::pow(x, n_f);
    }
    return py::float_(std::log(sum) / (1.0 - n_f));
}

std::tuple<py::object, py::object>
qr_py(py::object tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top)
{
    if (is_ChargedTensor(tensor)) {
        auto charged = tensor.cast<std::shared_ptr<ChargedTensor const>>();
        py::object inv_part = py::cast(charged->invariant_part);
        if (!charge_leg_top) {
            inv_part = move_charge_leg(inv_part, charge_leg_label(), 0, std::nullopt);
        }
        auto [Q, R] = qr_py(inv_part, new_labels, new_leg_dual, true);
        if (charge_leg_top) {
            R = make_python_charged_tensor(R, py::cast(charged->charged_state));
        } else {
            Q = move_charge_leg(Q, charge_leg_label(), std::nullopt, 0);
            Q = make_python_charged_tensor(Q, py::cast(charged->charged_state));
        }
        return { Q, R };
    }

    auto [a, b] = _decomposition_labels(leg_labels_from_py(to_iterable(new_labels)));
    auto hidden = stripped_hidden_labels(tensor);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor.cast<TensorCPtr>(), new_leg_dual);
    auto backend = tens->backend;
    auto [q_data, r_data] = backend->qr(tens, new_co_domain);
    py::object Q = make_python_symmetric_tensor(
      std::move(q_data),
      py::cast(tens->codomain),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(labels_to_py(tens->codomain_labels()), leg_label_to_py(a)));
    py::object R = make_python_symmetric_tensor(
      std::move(r_data),
      py::cast(new_co_domain),
      py::cast(tens->domain),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(b), labels_to_py(tens->domain_labels())));
    if (combine_codomain) {
        Q = split_one_leg(Q, 0);
    }
    if (combine_domain) {
        R = split_one_leg(R, -1);
    }
    Q = rehide_factor(Q, hidden, -1);
    R = rehide_factor(R, hidden, 0);
    return { Q, R };
}

std::tuple<py::object, py::object>
lq_py(py::object tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top)
{
    if (is_ChargedTensor(tensor)) {
        auto charged = tensor.cast<std::shared_ptr<ChargedTensor const>>();
        py::object inv_part = py::cast(charged->invariant_part);
        if (!charge_leg_top) {
            inv_part = move_charge_leg(inv_part, charge_leg_label(), 0, std::nullopt);
        }
        auto [L, Q] = lq_py(inv_part, new_labels, new_leg_dual, true);
        if (charge_leg_top) {
            Q = make_python_charged_tensor(Q, py::cast(charged->charged_state));
        } else {
            L = move_charge_leg(L, charge_leg_label(), std::nullopt, 0);
            L = make_python_charged_tensor(L, py::cast(charged->charged_state));
        }
        return { L, Q };
    }

    auto [a, b] = _decomposition_labels(leg_labels_from_py(to_iterable(new_labels)));
    auto hidden = stripped_hidden_labels(tensor);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor.cast<TensorCPtr>(), new_leg_dual);
    auto backend = tens->backend;
    auto [l_data, q_data] = backend->lq(tens, new_co_domain);
    py::object L = make_python_symmetric_tensor(
      std::move(l_data),
      py::cast(tens->codomain),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(labels_to_py(tens->codomain_labels()), leg_label_to_py(a)));
    py::object Q = make_python_symmetric_tensor(
      std::move(q_data),
      py::cast(new_co_domain),
      py::cast(tens->domain),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(b), labels_to_py(tens->domain_labels())));
    if (combine_codomain) {
        L = split_one_leg(L, 0);
    }
    if (combine_domain) {
        Q = split_one_leg(Q, -1);
    }
    L = rehide_factor(L, hidden, -1);
    Q = rehide_factor(Q, hidden, 0);
    return { L, Q };
}

std::tuple<py::object, py::object, py::object>
svd_py(py::object tensor,
       py::object new_labels,
       bool new_leg_dual,
       bool charge_leg_top,
       py::object algorithm)
{
    // --- hints from Python svd_py ---
    // split legs, if they were previously combined
    // ---
    if (is_ChargedTensor(tensor)) {
        auto charged = tensor.cast<std::shared_ptr<ChargedTensor const>>();
        py::object inv_part = py::cast(charged->invariant_part);
        if (!charge_leg_top) {
            inv_part = move_charge_leg(inv_part, charge_leg_label(), 0, std::nullopt);
        }
        // Intentional: pass algorithm by keyword (Python positional call was ambiguous).
        auto [U, S, Vh] = svd_py(inv_part, new_labels, new_leg_dual, true, algorithm);
        if (charge_leg_top) {
            Vh = make_python_charged_tensor(Vh, py::cast(charged->charged_state));
        } else {
            U = move_charge_leg(U, charge_leg_label(), std::nullopt, 0);
            U = make_python_charged_tensor(U, py::cast(charged->charged_state));
        }
        return { U, S, Vh };
    }

    std::optional<LegLabelList> svd_labs;
    if (!new_labels.is_none()) {
        svd_labs = leg_labels_from_py(to_iterable(new_labels));
    }
    auto [a, b, c, d] = _svd_new_labels(svd_labs);
    auto hidden = stripped_hidden_labels(tensor);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor.cast<TensorCPtr>(), new_leg_dual);
    auto backend = tens->backend;
    std::optional<std::string> algo;
    if (!algorithm.is_none()) {
        algo = algorithm.cast<std::string>();
    }
    auto [u_data, s_data, vh_data] = backend->svd(tens, new_co_domain, algo);
    py::object U = make_python_symmetric_tensor(
      std::move(u_data),
      py::cast(tens->codomain),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(labels_to_py(tens->codomain_labels()), leg_label_to_py(a)));
    py::object S = make_python_diagonal_tensor(std::move(s_data),
                                               py::cast(new_co_domain).attr("__getitem__")(0),
                                               backend,
                                               py_list(leg_label_to_py(b), leg_label_to_py(c)));
    py::object Vh = make_python_symmetric_tensor(
      std::move(vh_data),
      py::cast(new_co_domain),
      py::cast(tens->domain),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(d), labels_to_py(tens->domain_labels())));
    // split legs, if they were previously combined
    if (combine_codomain) {
        U = split_one_leg(U, 0);
    }
    if (combine_domain) {
        Vh = split_one_leg(Vh, -1);
    }
    U = rehide_factor(U, hidden, -1);
    Vh = rehide_factor(Vh, hidden, 0);
    return { U, S, Vh };
}

std::tuple<py::object, py::object, py::object>
svd_apply_mask_py(py::object U, py::object S, py::object Vh, py::object mask)
{
    if (!mask.attr("is_projection").cast<bool>()) {
        throw std::invalid_argument("mask must be a projection");
    }
    if (!py_eq(mask.attr("domain").attr("__getitem__")(0),
               S.attr("domain").attr("__getitem__")(0))) {
        throw std::invalid_argument("mask domain leg must match the singular-value tensor");
    }

    U = py::cast(_compose_with_Mask(
      U.cast<TensorCPtr>(), std::dynamic_pointer_cast<Mask>(dagger(mask.cast<TensorCPtr>())), -1));
    S = apply_mask_DiagonalTensor_py(S, mask);
    Vh = py::cast(_compose_with_Mask(Vh.cast<TensorCPtr>(), mask.cast<MaskCPtr>(), 0));
    return { U, S, Vh };
}

std::tuple<py::object, float64, float64>
truncate_singular_values_py(py::object S,
                            std::optional<int64> chi_max,
                            int64 chi_min,
                            float64 degeneracy_tol,
                            float64 trunc_cut,
                            float64 svd_min,
                            bool minimize_error,
                            py::object mask_labels)
{
    if (!S.attr("dtype").attr("is_real").cast<bool>()) {
        throw std::invalid_argument("truncate_singular_values requires a real dtype");
    }
    auto backend = S.attr("backend").cast<TensorBackend::Ptr>();
    auto [mask_data, new_leg, err, new_norm] =
      backend->truncate_singular_values(S.cast<DiagonalTensorCPtr>(),
                                        chi_max,
                                        chi_min,
                                        degeneracy_tol,
                                        trunc_cut,
                                        std::optional<float64>{ svd_min },
                                        minimize_error);
    if (mask_labels.is_none()) {
        std::string lab0 = py_label_as_string(S.attr("labels").attr("__getitem__")(0));
        std::string dual = _dual_leg_label(lab0);
        mask_labels = py_list(leg_label_to_py(lab0), leg_label_to_py(dual));
    }
    py::object mask = make_python_mask(std::move(mask_data),
                                       S.attr("leg"),
                                       py::cast(new_leg),
                                       /*is_projection=*/true,
                                       backend,
                                       mask_labels);
    return { mask, err, new_norm };
}

std::tuple<py::object, py::object, py::object, float64, float64>
truncated_svd_py(py::object tensor,
                 py::object new_labels,
                 bool new_leg_dual,
                 bool charge_leg_top,
                 py::object algorithm,
                 std::optional<float64> normalize_to,
                 std::optional<int64> chi_max,
                 int64 chi_min,
                 float64 degeneracy_tol,
                 float64 trunc_cut,
                 float64 svd_min)
{
    // --- hints from Python truncated_svd_py ---
    // norm(S[mask]) == S_norm * new_norm
    // ---
    auto [U, S, Vh] = svd_py(tensor, new_labels, new_leg_dual, charge_leg_top, algorithm);
    py::object S_norm_obj = py::cast(norm(S.cast<TensorCPtr>()));
    // norm() returns a BlockBackend.Scalar (or number); normalize to float64.
    float64 S_norm;
    if (py::hasattr(S_norm_obj, "to_numpy")) {
        S_norm = S_norm_obj.attr("to_numpy")().cast<float64>();
    } else {
        S_norm = S_norm_obj.cast<float64>();
    }
    // S / S_norm via Python
    py::object S_normed = S.attr("__truediv__")(S_norm_obj);
    auto [mask, err, new_norm] =
      truncate_singular_values_py(S_normed, chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min);
    std::tie(U, S, Vh) = svd_apply_mask_py(U, S, Vh, mask);
    float64 renormalize = 1.;
    if (normalize_to.has_value()) {
        // Scale the returned S to normalize_to; renormalize restores the original scale.
        renormalize = S_norm * new_norm / *normalize_to;
        S = S.attr("__mul__")(*normalize_to / S_norm / new_norm);
    }
    return { U, S, Vh, err, renormalize };
}

namespace {

py::object
labels_to_py_opt(std::optional<LegLabelList> const& labels)
{
    if (!labels.has_value()) {
        return py::none();
    }
    py::list out;
    for (auto const& lab : *labels) {
        out.append(lab);
    }
    return out;
}

py::object
labels_to_py_opt(LegLabelList const& labels)
{
    return labels_to_py_opt(std::optional<LegLabelList>{ labels });
}

py::object
algo_to_py(std::optional<std::string> const& algorithm)
{
    if (!algorithm.has_value()) {
        return py::none();
    }
    return py::cast(*algorithm);
}

BlockBackend::Scalar
coerce_scalar_decomp(py::object o, TensorCPtr hint)
{
    try {
        return o.cast<BlockBackend::Scalar>();
    } catch (py::cast_error const&) {
    }
    return hint->backend->block_backend->as_scalar(o, hint->dtype);
}

} // namespace

DiagonalTensorPtr
apply_mask_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask)
{
    return apply_mask_DiagonalTensor_py(py::cast(tensor), py::cast(mask))
      .cast<DiagonalTensorPtr>();
}

std::tuple<DiagonalTensorPtr, TensorPtr>
eigh(TensorCPtr tensor,
     LegLabelList new_labels,
     bool new_leg_dual,
     std::optional<std::string> sort)
{
    py::object sort_py = sort.has_value() ? py::cast(*sort) : py::none();
    auto [W, V] = eigh_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, sort_py);
    return { W.cast<DiagonalTensorPtr>(), V.cast<TensorPtr>() };
}

std::tuple<DiagonalTensorPtr, TensorPtr>
eig(TensorCPtr tensor, LegLabelList new_labels, bool new_leg_dual, std::optional<std::string> sort)
{
    py::object sort_py = sort.has_value() ? py::cast(*sort) : py::none();
    auto [W, V] = eig_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, sort_py);
    return { W.cast<DiagonalTensorPtr>(), V.cast<TensorPtr>() };
}

DiagonalTensorPtr
eigvalsh(TensorCPtr tensor,
         LegLabelList new_labels,
         bool new_leg_dual,
         std::optional<std::string> sort)
{
    py::object sort_py = sort.has_value() ? py::cast(*sort) : py::none();
    return eigvalsh_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, sort_py)
      .cast<DiagonalTensorPtr>();
}

DiagonalTensorPtr
eigvals(TensorCPtr tensor,
        LegLabelList new_labels,
        bool new_leg_dual,
        std::optional<std::string> sort)
{
    py::object sort_py = sort.has_value() ? py::cast(*sort) : py::none();
    return eigvals_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, sort_py)
      .cast<DiagonalTensorPtr>();
}

BlockBackend::Scalar
entropy(DiagonalTensorCPtr p, float64 n)
{
    if (std::dynamic_pointer_cast<Identity const>(p)) {
        throw py::type_error(
          "entropy does not support Identity. It is never a normalized distribution.");
    }
    if (!dtype::is_real(p->dtype)) {
        throw std::invalid_argument("entropy requires a real dtype");
    }
    auto mut = std::const_pointer_cast<DiagonalTensor>(p);
    auto mul = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) * (*b);
    };
    auto pow = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return a->pow(*b);
    };
    if (n == 1.0) {
        auto logged = stable_log(p, 1e-30);
        auto prod = mut->_binary_operand(DiagonalTensorCPtr(logged), mul, "mul");
        return -trace(prod);
    }
    if (std::isinf(n)) {
        return -p->max().log();
    }
    auto powered = mut->_binary_operand(p->backend->block_backend->as_scalar(n), pow, "pow");
    return trace(powered).log() / (1.0 - n);
}

std::tuple<TensorPtr, TensorPtr>
lq(TensorCPtr tensor,
   std::optional<LegLabelList> new_labels,
   bool new_leg_dual,
   bool charge_leg_top)
{
    auto [L, Q] =
      lq_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, charge_leg_top);
    return { L.cast<TensorPtr>(), Q.cast<TensorPtr>() };
}

std::tuple<TensorPtr, TensorPtr>
qr(TensorCPtr tensor,
   std::optional<LegLabelList> new_labels,
   bool new_leg_dual,
   bool charge_leg_top)
{
    auto [Q, R] =
      qr_py(py::cast(tensor), labels_to_py_opt(new_labels), new_leg_dual, charge_leg_top);
    return { Q.cast<TensorPtr>(), R.cast<TensorPtr>() };
}

std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr>
svd(TensorCPtr tensor,
    std::optional<LegLabelList> new_labels,
    bool new_leg_dual,
    bool charge_leg_top,
    std::optional<std::string> algorithm)
{
    auto [U, S, Vh] = svd_py(py::cast(tensor),
                             labels_to_py_opt(new_labels),
                             new_leg_dual,
                             charge_leg_top,
                             algo_to_py(algorithm));
    return { U.cast<TensorPtr>(), S.cast<DiagonalTensorPtr>(), Vh.cast<TensorPtr>() };
}

std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr>
svd_apply_mask(TensorCPtr U, DiagonalTensorCPtr S, TensorCPtr Vh, MaskCPtr mask)
{
    auto [u, s, vh] = svd_apply_mask_py(py::cast(U), py::cast(S), py::cast(Vh), py::cast(mask));
    return { u.cast<TensorPtr>(), s.cast<DiagonalTensorPtr>(), vh.cast<TensorPtr>() };
}

std::tuple<MaskPtr, float64, float64>
truncate_singular_values(DiagonalTensorCPtr S,
                         std::optional<int64> chi_max,
                         int64 chi_min,
                         float64 degeneracy_tol,
                         float64 trunc_cut,
                         float64 svd_min,
                         bool minimize_error,
                         std::optional<LegLabelList> mask_labels)
{
    auto [mask, err, new_norm] = truncate_singular_values_py(py::cast(S),
                                                             chi_max,
                                                             chi_min,
                                                             degeneracy_tol,
                                                             trunc_cut,
                                                             svd_min,
                                                             minimize_error,
                                                             labels_to_py_opt(mask_labels));
    return { mask.cast<MaskPtr>(), err, new_norm };
}

std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64>
truncated_svd(TensorCPtr tensor,
              std::optional<LegLabelList> new_labels,
              bool new_leg_dual,
              bool charge_leg_top,
              std::optional<std::string> algorithm,
              std::optional<float64> normalize_to,
              std::optional<int64> chi_max,
              int64 chi_min,
              float64 degeneracy_tol,
              float64 trunc_cut,
              float64 svd_min)
{
    auto [U, S, Vh, err, renormalize] = truncated_svd_py(py::cast(tensor),
                                                         labels_to_py_opt(new_labels),
                                                         new_leg_dual,
                                                         charge_leg_top,
                                                         algo_to_py(algorithm),
                                                         normalize_to,
                                                         chi_max,
                                                         chi_min,
                                                         degeneracy_tol,
                                                         trunc_cut,
                                                         svd_min);
    return {
        U.cast<TensorPtr>(), S.cast<DiagonalTensorPtr>(), Vh.cast<TensorPtr>(), err, renormalize
    };
}

} // namespace lasyten
