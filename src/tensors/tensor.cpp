#include <lasyten/tensors/tensor.h>

#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tools.h>
#include <lasyten/tools/misc.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <ranges>
#include <sstream>
#include <stdexcept>

namespace lasyten {

namespace {

Space::Ptr
as_space(py::handle obj)
{
    return obj.cast<Space::Ptr>();
}

/// Symmetry of a :class:`TensorProduct` factor (:class:`Space` or :class:`Leg` /
/// :class:`LegPipe`).
Symmetry::Ptr
factor_symmetry(py::handle factor)
{
    if (py::isinstance<Space>(factor)) {
        return factor.cast<Space*>()->symmetry;
    }
    if (py::isinstance<Leg>(factor)) {
        return factor.cast<Leg*>()->symmetry;
    }
    return factor.attr("symmetry").cast<Symmetry::Ptr>();
}

/// Quantum dimension of a :class:`TensorProduct` factor.
float64
factor_dim(py::handle factor)
{
    if (py::isinstance<Space>(factor)) {
        return factor.cast<Space*>()->dim;
    }
    if (py::isinstance<Leg>(factor)) {
        return factor.cast<Leg*>()->dim;
    }
    return factor.attr("dim").cast<float64>();
}

std::string
as_leg_label(py::handle obj)
{
    if (obj.is_none()) {
        return "";
    }
    return obj.cast<std::string>();
}

LegLabelList
sequence_as_leg_labels(py::handle seq)
{
    LegLabelList out;
    for (auto item : seq) {
        out.push_back(as_leg_label(item));
    }
    return out;
}

std::string
format_dim(float64 dim, int distance, std::string const& huge_dim, float64 huge_dim_value)
{
    auto s0 = std::format("{}", dim);
    // Prefer integer-looking formatting when close to int
    if (std::floor(dim) == dim && dim >= 0) {
        s0 = std::format("{}", static_cast<int64>(dim));
    }
    if (static_cast<int>(s0.size()) <= distance) {
        return std::string(distance - s0.size(), ' ') + s0;
    }
    if (dim >= huge_dim_value) {
        return huge_dim;
    }
    auto s = std::format("{:.1f}", dim);
    if (static_cast<int>(s.size()) <= distance) {
        return std::string(distance - s.size(), ' ') + s;
    }
    s = std::format("{}", static_cast<int64>(std::llround(dim)));
    if (static_cast<int>(s.size()) <= distance) {
        return std::string(distance - s.size(), ' ') + s;
    }
    throw std::runtime_error("format_dim: unexpected dim formatting failure");
}

/// Match Space.dim / Tensor.shape Python exposure: whole-number dims as int.
py::list
dims_to_python(std::vector<float64> const& dims)
{
    py::list out;
    for (float64 d : dims) {
        if (std::isfinite(d) && std::floor(d) == d) {
            out.append(py::int_(static_cast<long long>(d)));
        } else {
            out.append(py::float_(d));
        }
    }
    return out;
}

/// Number of Unicode code points in a UTF-8 string (matches Python ``len`` on str).
int
utf8_len(std::string const& s)
{
    int n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

std::string
rjust(std::string const& s, int width)
{
    int const len = utf8_len(s);
    if (len >= width) {
        return s;
    }
    return std::string(width - len, ' ') + s;
}

std::string
join_spaced(std::vector<std::string> const& parts)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out << ' ';
        }
        out << parts[i];
    }
    return out.str();
}

} // namespace

std::vector<Dtype> Tensor::_forbidden_dtypes = { Dtype::Bool };

std::vector<Dtype> const&
Tensor::forbidden_dtypes() const
{
    return _forbidden_dtypes;
}

Tensor::Tensor(TensorProduct::Ptr codomain_,
               TensorProduct::Ptr domain_,
               TensorBackend::Ptr backend_,
               Symmetry::Ptr symmetry_,
               LegLabelList labels_in,
               Dtype dtype_,
               std::string device_)
  : labels(std::move(labels_in))
  , num_legs(labels.num_legs())
  , codomain(std::move(codomain_))
  , domain(std::move(domain_))
  , backend(std::move(backend_))
  , symmetry(std::move(symmetry_))
  , dtype(dtype_)
  , device(std::move(device_))
{
    if (!codomain || !domain) {
        throw std::invalid_argument("codomain and domain must be non-null TensorProduct");
    }
    if (!symmetry) {
        symmetry = codomain->symmetry;
    }
    if (!backend) {
        backend = get_backend(symmetry);
    }
    if (!backend->supports_symmetry(symmetry)) {
        throw std::invalid_argument("backend does not support this symmetry");
    }
    if (!codomain->symmetry || !symmetry || !codomain->symmetry->equals(*symmetry)) {
        throw std::invalid_argument("codomain symmetry must match tensor symmetry");
    }
    if (!domain->symmetry || !symmetry || !domain->symmetry->equals(*symmetry)) {
        throw std::invalid_argument("domain symmetry must match tensor symmetry");
    }

    shape.clear();
    shape.reserve(static_cast<std::size_t>(codomain->num_factors + domain->num_factors));
    for (auto const& f : codomain->factors) {
        shape.push_back(f->dim);
    }
    for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it) {
        shape.push_back((*it)->dim);
    }

    if (num_legs != codomain->num_factors + domain->num_factors) {
        throw std::invalid_argument(std::format(
          "expected {} labels, got {}", codomain->num_factors + domain->num_factors, num_legs));
    }

    // Keep num_legs in sync if labels are replaced (size is fixed for a live tensor).
    labels.set_hooks(
      [this](LegLabels& /*self*/, LegLabelList const& proposed) {
          if (static_cast<int64>(proposed.size()) != num_legs) {
              throw std::invalid_argument(
                std::format("expected {} labels, got {}", num_legs, proposed.size()));
          }
      },
      [this](LegLabels& self) { num_legs = self.num_legs(); });
}

Tensor::Tensor(InitParsed init, Dtype dtype, std::string device)
  : Tensor(std::move(init.codomain),
           std::move(init.domain),
           std::move(init.backend),
           std::move(init.symmetry),
           std::move(init.labels),
           dtype,
           std::move(device))
{
}

std::tuple<TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, Symmetry::Ptr>
Tensor::_init_parse_args(TensorProduct::Ptr codomain,
                         TensorProduct::Ptr domain,
                         TensorBackend::Ptr backend)
{
    // --- hints from Python Tensor._init_parse_args ---
    // Extract the symmetry from codomain or domain. Note that either may be empty, but not both.
    // Make sure backend is compatible with symmetry
    // Bring (co-)domain to TensorProduct form
    // ---
    if (!codomain) {
        throw std::invalid_argument("codomain must be a TensorProduct");
    }
    if (!domain) {
        domain = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{}, codomain->symmetry);
    }
    Symmetry::Ptr symmetry;
    if (codomain->num_factors > 0 || codomain->symmetry) {
        symmetry = codomain->symmetry;
    }
    if (!symmetry && domain->symmetry) {
        symmetry = domain->symmetry;
    }
    if (!symmetry) {
        throw std::invalid_argument("domain and codomain can not both be empty");
    }
    if (codomain->num_factors == 0 && domain->num_factors == 0 && !codomain->symmetry &&
        !domain->symmetry) {
        throw std::invalid_argument("domain and codomain can not both be empty");
    }

    if (!backend) {
        backend = get_backend(symmetry);
    }
    if (!backend->supports_symmetry(symmetry)) {
        throw std::invalid_argument("backend does not support this symmetry");
    }
    if (!codomain->symmetry || !symmetry || !codomain->symmetry->equals(*symmetry)) {
        throw std::invalid_argument("codomain symmetry must match tensor symmetry");
    }
    if (!domain->symmetry || !symmetry || !domain->symmetry->equals(*symmetry)) {
        throw std::invalid_argument("domain symmetry must match tensor symmetry");
    }
    return { std::move(codomain), std::move(domain), std::move(backend), std::move(symmetry) };
}

LegLabelList
Tensor::_init_parse_labels(std::optional<LegLabelList> labels,
                           TensorProduct::Ptr const& codomain,
                           TensorProduct::Ptr const& domain,
                           bool is_endomorphism)
{
    int64 const num_legs = codomain->num_factors + domain->num_factors;
    if (is_endomorphism && codomain->num_factors != domain->num_factors) {
        throw std::invalid_argument(
          "endomorphism requires the same number of domain and codomain legs");
    }
    if (!labels.has_value()) {
        return LegLabelList(static_cast<std::size_t>(num_legs), "");
    }
    if (labels->empty()) {
        if (num_legs != 0) {
            throw std::invalid_argument("empty labels are only allowed for tensors with no legs");
        }
        return {};
    }
    if (is_endomorphism && static_cast<int64>(labels->size()) == codomain->num_factors) {
        LegLabelList out = *labels;
        for (auto it = labels->rbegin(); it != labels->rend(); ++it) {
            out.push_back(_dual_leg_label(*it));
        }
        return out;
    }
    if (static_cast<int64>(labels->size()) != num_legs) {
        throw std::invalid_argument(
          std::format("expected {} labels, got {}", num_legs, labels->size()));
    }
    return *labels;
}

TensorProduct::Ptr
tensor_product_from_python(py::object obj, Symmetry::Ptr symmetry)
{
    if (obj.is_none()) {
        if (!symmetry) {
            throw std::invalid_argument("Need a symmetry to build an empty TensorProduct");
        }
        return std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{}, std::move(symmetry));
    }
    if (py::isinstance<TensorProduct>(obj)) {
        return obj.cast<TensorProduct::Ptr>();
    }
    std::vector<Leg::Ptr> factors;
    for (auto item : obj) {
        factors.push_back(item.cast<Leg::Ptr>());
    }
    return std::make_shared<TensorProduct>(std::move(factors), std::move(symmetry));
}

std::tuple<TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, Symmetry::Ptr>
parse_tensor_init_args(py::object codomain, py::object domain, TensorBackend::Ptr backend)
{
    Symmetry::Ptr symmetry;
    if (py::isinstance<TensorProduct>(codomain)) {
        symmetry = codomain.cast<TensorProduct::Ptr>()->symmetry;
    } else if (py::len(codomain) > 0) {
        symmetry = factor_symmetry(codomain[py::int_(0)]);
    } else if (!domain.is_none() && py::isinstance<TensorProduct>(domain)) {
        symmetry = domain.cast<TensorProduct::Ptr>()->symmetry;
    } else if (!domain.is_none() && py::len(domain) > 0) {
        symmetry = factor_symmetry(domain[py::int_(0)]);
    } else {
        throw std::invalid_argument("domain and codomain can not both be empty");
    }
    auto codomain_tp = tensor_product_from_python(codomain, symmetry);
    if (domain.is_none()) {
        domain = py::list();
    }
    auto domain_tp = tensor_product_from_python(domain, symmetry);
    return Tensor::_init_parse_args(
      std::move(codomain_tp), std::move(domain_tp), std::move(backend));
}

LegLabelList
parse_tensor_init_labels(py::object labels,
                         TensorProduct::Ptr const& codomain,
                         TensorProduct::Ptr const& domain,
                         bool is_endomorphism)
{
    // --- hints from Python Tensor._init_parse_labels ---
    // case 1: None
    // case 2: two lists, one each for codomain and domain
    // expect nested lists
    // case 3a: (only if is_endomorphism) a flat list for the codomain
    // case 3: a flat list for the legs
    // ---
    int64 const num_legs = codomain->num_factors + domain->num_factors;
    if (is_endomorphism && codomain->num_factors != domain->num_factors) {
        throw std::invalid_argument(
          "endomorphism requires the same number of domain and codomain legs");
    }

    if (labels.is_none()) {
        return Tensor::_init_parse_labels(std::nullopt, codomain, domain, is_endomorphism);
    }

    py::sequence seq = labels.cast<py::sequence>();
    if (py::len(seq) == 0) {
        return Tensor::_init_parse_labels(LegLabelList{}, codomain, domain, is_endomorphism);
    }

    py::object first = seq[py::int_(0)];
    if (!(py::isinstance<py::str>(first) || first.is_none())) {
        if (py::len(seq) != 2) {
            throw std::invalid_argument("Expected [codomain_labels, domain_labels]");
        }
        py::object codomain_labels_obj = seq[py::int_(0)];
        py::object domain_labels_obj = seq[py::int_(1)];
        LegLabelList codomain_labels;
        LegLabelList domain_labels;
        if (codomain_labels_obj.is_none()) {
            if (is_endomorphism && !domain_labels_obj.is_none()) {
                for (auto item : domain_labels_obj) {
                    codomain_labels.push_back(_dual_leg_label(as_leg_label(item)));
                }
            } else {
                codomain_labels =
                  LegLabelList(static_cast<std::size_t>(codomain->num_factors), "");
            }
        } else {
            codomain_labels = sequence_as_leg_labels(codomain_labels_obj);
        }
        if (static_cast<int64>(codomain_labels.size()) != codomain->num_factors) {
            throw std::invalid_argument(std::format("expected {} codomain labels, got {}",
                                                    codomain->num_factors,
                                                    codomain_labels.size()));
        }

        if (domain_labels_obj.is_none()) {
            if (is_endomorphism) {
                for (auto const& l : codomain_labels) {
                    domain_labels.push_back(_dual_leg_label(l));
                }
            } else {
                domain_labels = LegLabelList(static_cast<std::size_t>(domain->num_factors), "");
            }
        } else {
            domain_labels = sequence_as_leg_labels(domain_labels_obj);
        }
        if (static_cast<int64>(domain_labels.size()) != domain->num_factors) {
            throw std::invalid_argument(std::format(
              "expected {} domain labels, got {}", domain->num_factors, domain_labels.size()));
        }

        LegLabelList out = codomain_labels;
        for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
            out.push_back(*it);
        }
        return out;
    }

    LegLabelList flat = sequence_as_leg_labels(seq);
    return Tensor::_init_parse_labels(std::move(flat), codomain, domain, is_endomorphism);
}

Tensor::InitParsed
parse_tensor_init(py::object codomain,
                  py::object domain,
                  TensorBackend::Ptr backend,
                  py::object labels,
                  bool is_endomorphism)
{
    auto [c, d, b, s] = parse_tensor_init_args(codomain, domain, std::move(backend));
    auto labs = parse_tensor_init_labels(labels, c, d, is_endomorphism);
    return { std::move(c), std::move(d), std::move(b), std::move(s), std::move(labs) };
}

void
Tensor::test_sanity() const
{
    // --- hints from Python Tensor.test_sanity ---
    // this checks all legs, and recursively through pipes
    // ---
    domain->test_sanity();   // this checks all legs, and recursively through pipes
    codomain->test_sanity(); // this checks all legs, and recursively through pipes
    assert(std::find(forbidden_dtypes().begin(), forbidden_dtypes().end(), dtype) ==
           forbidden_dtypes().end());
    for (auto const& leg : domain->factors) {
        (void)leg;
    }
    for (auto const& leg : codomain->factors) {
        (void)leg;
    }
    labels.test_sanity();
}

std::string
Tensor::ascii_diagram_type_name() const
{
    return "???";
}

std::string
Tensor::class_name() const
{
    return "Tensor";
}

std::string
Tensor::ascii_diagram() const
{
    // --- hints from Python Tensor.ascii_diagram ---
    // distance between legs in chars, i.e. number of '━' between the '┯'
    // for numbers that can not fit in DISTANCE digits
    // this should not happen
    // such that f'{start}┗┯' has length DISTANCE
    // make room for the text
    // top border:
    // bottom border:
    // stitch together
    // ---
    std::string text = ascii_diagram_type_name();

    int const DISTANCE = 5; // distance between legs in chars, i.e. number of '━' between the '┯'

    std::string huge_dim = std::format(">1e{}", DISTANCE + 1); // for numbers that can not fit
    assert(static_cast<int>(huge_dim.size()) <= DISTANCE);
    huge_dim = rjust(huge_dim, DISTANCE);
    float64 const huge_dim_value = std::pow(10.0, DISTANCE);
    assert(static_cast<int>(std::format("{}", static_cast<int64>(huge_dim_value)).size()) >
           DISTANCE);
    assert(static_cast<int>(std::format("{}", static_cast<int64>(huge_dim_value) - 1).size()) <=
           DISTANCE);

    auto const leg_spaces = legs();
    std::vector<std::string> dims;
    dims.reserve(leg_spaces.size());
    for (auto const& l : leg_spaces) {
        dims.push_back(format_dim(l->dim, DISTANCE, huge_dim, huge_dim_value));
    }
    auto const n_cod = num_codomain_legs();
    std::vector<std::string> codomain_dims(dims.begin(), dims.begin() + n_cod);
    std::vector<std::string> domain_dims(dims.begin() + n_cod, dims.end());
    std::reverse(domain_dims.begin(), domain_dims.end());

    std::vector<std::string> codomain_arrows;
    std::vector<std::string> domain_arrows;
    for (auto const& f : codomain->factors) {
        codomain_arrows.push_back(rjust(f->ascii_arrow(), DISTANCE));
    }
    for (auto const& f : domain->factors) {
        domain_arrows.push_back(rjust(f->ascii_arrow(), DISTANCE));
    }

    auto const c_labs = codomain_labels();
    auto const d_labs = domain_labels();
    std::vector<std::string> codomain_labels_s;
    std::vector<std::string> domain_labels_s;
    for (auto const& l : c_labs) {
        std::string s = l.empty() ? "None" : l;
        if (static_cast<int>(s.size()) <= DISTANCE) {
            codomain_labels_s.push_back(rjust(s, DISTANCE));
        } else {
            codomain_labels_s.push_back(rjust("...", DISTANCE));
        }
    }
    for (auto const& l : d_labs) {
        std::string s = l.empty() ? "None" : l;
        if (static_cast<int>(s.size()) <= DISTANCE) {
            domain_labels_s.push_back(rjust(s, DISTANCE));
        } else {
            domain_labels_s.push_back(rjust("...", DISTANCE));
        }
    }

    std::string const start(DISTANCE - 2, ' '); // such that f'{start}┗┯' has length DISTANCE
    //
    assert(DISTANCE % 2 == 1);
    int codomain_extra = 0;
    int domain_extra = 0;
    if (num_codomain_legs() > num_domain_legs()) {
        domain_extra =
          ((DISTANCE + 1) / 2) * static_cast<int>(num_codomain_legs() - num_domain_legs());
    } else {
        codomain_extra =
          ((DISTANCE + 1) / 2) * static_cast<int>(num_domain_legs() - num_codomain_legs());
    }
    //
    if (num_codomain_legs() < 2 && num_domain_legs() < 2) {
        // make room for the text
        codomain_extra += 3;
        domain_extra += 3;
    }

    auto repeat = [](std::string const& unit, int n) {
        std::string out;
        for (int i = 0; i < n; ++i) {
            out += unit;
        }
        return out;
    };

    // top border:
    std::string top_border;
    if (num_domain_legs() > 0) {
        std::string mid;
        for (int64 i = 0; i < num_domain_legs(); ++i) {
            if (i > 0) {
                mid += repeat("━", DISTANCE);
            }
            mid += "┷";
        }
        top_border =
          start + "┏" + repeat("━", domain_extra) + mid + repeat("━", domain_extra) + "┓";
    } else {
        top_border = start + "┏" +
                     repeat("━", (DISTANCE + 1) * static_cast<int>(num_codomain_legs() - 1) + 1) +
                     "┓";
    }
    // body:
    // top_border uses UTF-8 box-drawing chars; match Python ``len`` (codepoints, not bytes).
    int const chars_in_box = utf8_len(top_border) - utf8_len(start) - 2;
    std::string front_pad((chars_in_box - static_cast<int>(text.size())) / 2, ' ');
    std::string back_pad(
      chars_in_box - static_cast<int>(text.size()) - static_cast<int>(front_pad.size()), ' ');
    std::string body = start + "┃" + front_pad + text + back_pad + "┃";
    // bottom border:
    std::string bottom_border;
    if (num_codomain_legs() > 0) {
        std::string mid;
        for (int64 i = 0; i < num_codomain_legs(); ++i) {
            if (i > 0) {
                mid += repeat("━", DISTANCE);
            }
            mid += "┯";
        }
        bottom_border =
          start + "┗" + repeat("━", codomain_extra) + mid + repeat("━", codomain_extra) + "┛";
    } else {
        bottom_border = start + "┗" +
                        repeat("━", (DISTANCE + 1) * static_cast<int>(num_domain_legs() - 1) + 1) +
                        "┛";
    }

    // stitch together
    std::ostringstream out;
    out << std::string(domain_extra, ' ') << join_spaced(domain_dims) << '\n';
    out << std::string(domain_extra, ' ') << join_spaced(domain_arrows) << '\n';
    out << std::string(domain_extra, ' ') << join_spaced(domain_labels_s) << '\n';
    out << top_border << '\n';
    out << body << '\n';
    out << bottom_border << '\n';
    out << std::string(codomain_extra, ' ') << join_spaced(codomain_labels_s) << '\n';
    out << std::string(codomain_extra, ' ') << join_spaced(codomain_arrows) << '\n';
    out << std::string(codomain_extra, ' ') << join_spaced(codomain_dims);
    return out.str();
}

LegLabelList
Tensor::codomain_labels() const
{
    auto const labs = labels.as_list();
    return LegLabelList(labs.begin(), labs.begin() + num_codomain_legs());
}

Tensor::Ptr
Tensor::dagger() const
{
    throw NotImplemented("Tensor::dagger (free function dagger not yet converted)");
}

LegLabelList
Tensor::domain_labels() const
{
    auto const labs = labels.as_list();
    LegLabelList out(labs.begin() + num_codomain_legs(), labs.end());
    std::reverse(out.begin(), out.end());
    return out;
}

bool
Tensor::has_pipes() const
{
    return codomain->has_pipes() || domain->has_pipes();
}

std::vector<Leg::Ptr>
Tensor::legs() const
{
    std::vector<Leg::Ptr> out;
    out.reserve(static_cast<std::size_t>(num_legs));
    for (auto const& f : codomain->factors) {
        out.push_back(f);
    }
    for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it) {
        out.push_back((*it)->dual());
    }
    return out;
}

int64
Tensor::num_codomain_legs() const
{
    return codomain->num_factors;
}

int64
Tensor::num_domain_legs() const
{
    return domain->num_factors;
}

int64
Tensor::num_codomain_flat_legs() const
{
    return codomain->num_flat_legs();
}

int64
Tensor::num_domain_flat_legs() const
{
    return domain->num_flat_legs();
}

int64
Tensor::num_flat_legs() const
{
    return num_domain_flat_legs() + num_codomain_flat_legs();
}

int64
Tensor::num_parameters() const
{
    assert(codomain->sector_order == std::string("sorted") &&
           domain->sector_order == std::string("sorted"));
    int64 res = 0;
    SectorArray::iter_common_sorted(codomain->sector_decomposition,
                                    domain->sector_decomposition,
                                    true,
                                    true,
                                    [&](std::ptrdiff_t i, std::ptrdiff_t j) {
                                        res +=
                                          codomain->multiplicities[static_cast<std::size_t>(i)] *
                                          domain->multiplicities[static_cast<std::size_t>(j)];
                                    });
    return res;
}

int64
Tensor::size() const
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Tensor.size is not defined for symmetry {}", symmetry->repr()));
    }
    return static_cast<int64>(domain->dim * codomain->dim);
}

Tensor::Ptr
Tensor::T() const
{
    throw NotImplemented("Tensor::T (free function transpose not yet converted)");
}

Leg::Ptr
Tensor::_as_codomain_leg(std::variant<int64, std::string> idx) const
{
    auto [in_domain, co_domain_idx, _] = _parse_leg_idx(idx);
    if (in_domain) {
        return domain->factors[static_cast<std::size_t>(co_domain_idx)]->dual();
    }
    return codomain->factors[static_cast<std::size_t>(co_domain_idx)];
}

Leg::Ptr
Tensor::_as_domain_leg(std::variant<int64, std::string> idx) const
{
    auto [in_domain, co_domain_idx, _] = _parse_leg_idx(idx);
    if (in_domain) {
        return domain->factors[static_cast<std::size_t>(co_domain_idx)];
    }
    return codomain->factors[static_cast<std::size_t>(co_domain_idx)]->dual();
}

void
Tensor::dbg() const
{
    std::cout << ascii_diagram() << '\n';
}

std::tuple<bool, int64, int64>
Tensor::_parse_leg_idx(std::variant<int64, std::string> which_leg) const
{
    int64 idx;
    if (std::holds_alternative<std::string>(which_leg)) {
        idx = labels.index(std::get<std::string>(which_leg));
    } else {
        idx = labels.index(std::get<int64>(which_leg));
    }
    bool const in_domain = idx >= static_cast<int64>(codomain->factors.size());
    int64 co_domain_idx;
    if (in_domain) {
        co_domain_idx = num_legs - 1 - idx;
    } else {
        co_domain_idx = idx;
    }
    return { in_domain, co_domain_idx, idx };
}

std::vector<std::string>
Tensor::_repr_header_lines(std::string const& indent, bool use_symm_str) const
{
    // --- hints from Python Tensor._repr_header_lines ---
    // TODO should we put some info still ...?
    // ---
    std::string labels_str;
    auto const labs = labels.as_list();
    if (std::ranges::all_of(labs, [](std::string const& l) { return l.empty(); })) {
        labels_str = "None";
    } else {
        // Match Python f'{self.labels}   ;   {self.codomain_labels} <- {self.domain_labels}'
        labels_str = std::format("{}   ;   {} <- {}",
                                 format_like_list(py::cast(labs)),
                                 format_like_list(py::cast(codomain_labels())),
                                 format_like_list(py::cast(domain_labels())));
    }
    std::vector<std::string> lines = {
        std::format("{}* Device: {}", indent, device),
        std::format("{}* Backend: {}", indent, backend->__str__()),
        std::format("{}* Symmetry: {}", indent, use_symm_str ? symmetry->str() : symmetry->repr()),
        std::format("{}* Labels: {}", indent, labels_str),
    };
    if (symmetry->can_be_dropped()) {
        std::vector<float64> codomain_dims(shape.begin(), shape.begin() + num_codomain_legs());
        std::vector<float64> domain_dims(shape.begin() + num_codomain_legs(), shape.end());
        std::reverse(domain_dims.begin(), domain_dims.end());
        lines.push_back(std::format("{}* Shape: {}   ;   {} <- {}",
                                    indent,
                                    format_like_list(dims_to_python(shape)),
                                    format_like_list(dims_to_python(codomain_dims)),
                                    format_like_list(dims_to_python(domain_dims))));
    }
    if ((!symmetry->can_be_dropped()) || (!symmetry->is_abelian())) {
        if (has_pipes()) {
            // TODO should we put some info still ...?
        } else {
            std::vector<int64> codomain_nums;
            for (auto const& leg : codomain->factors) {
                int64 s = 0;
                for (auto m : as_space(leg)->multiplicities) {
                    s += m;
                }
                codomain_nums.push_back(s);
            }
            std::vector<int64> domain_nums;
            for (auto const& leg : domain->factors) {
                int64 s = 0;
                for (auto m : as_space(leg)->multiplicities) {
                    s += m;
                }
                domain_nums.push_back(s);
            }
            std::vector<int64> all_nums = codomain_nums;
            for (auto it = domain_nums.rbegin(); it != domain_nums.rend(); ++it) {
                all_nums.push_back(*it);
            }
            lines.push_back(std::format("{}* Num Sectors: {}   ;   {} <- {}",
                                        indent,
                                        format_like_list(py::cast(all_nums)),
                                        format_like_list(py::cast(codomain_nums)),
                                        format_like_list(py::cast(domain_nums))));
        }
    }
    return lines;
}

Leg::Ptr
Tensor::get_leg(std::variant<int64, std::string> which_leg) const
{
    // --- hints from Python Tensor.get_leg ---
    // which_leg is a list
    // ---
    auto [in_domain, co_domain_idx, _] = _parse_leg_idx(which_leg);
    if (in_domain) {
        return domain->factors[static_cast<std::size_t>(co_domain_idx)]->dual();
    }
    return codomain->factors[static_cast<std::size_t>(co_domain_idx)];
}

std::vector<Leg::Ptr>
Tensor::get_leg(std::vector<std::variant<int64, std::string>> const& which_legs) const
{
    std::vector<Leg::Ptr> out;
    out.reserve(which_legs.size());
    for (auto const& w : which_legs) {
        out.push_back(get_leg(w));
    }
    return out;
}

Leg::Ptr
Tensor::get_leg_co_domain(std::variant<int64, std::string> which_leg) const
{
    // --- hints from Python Tensor.get_leg_co_domain ---
    // which_leg is a list
    // ---
    auto [in_domain, co_domain_idx, _] = _parse_leg_idx(which_leg);
    if (in_domain) {
        return domain->factors[static_cast<std::size_t>(co_domain_idx)];
    }
    return codomain->factors[static_cast<std::size_t>(co_domain_idx)];
}

std::vector<Leg::Ptr>
Tensor::get_leg_co_domain(std::vector<std::variant<int64, std::string>> const& which_legs) const
{
    std::vector<Leg::Ptr> out;
    out.reserve(which_legs.size());
    for (auto const& w : which_legs) {
        out.push_back(get_leg_co_domain(w));
    }
    return out;
}

py::array
Tensor::to_numpy(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                 py::object numpy_dtype,
                 bool understood_braiding)
{
    auto block = to_dense_block(std::move(leg_order), std::nullopt, understood_braiding);
    std::optional<py::object> np_dtype;
    if (!numpy_dtype.is_none()) {
        np_dtype = numpy_dtype;
    }
    return backend->block_backend->to_numpy(block, np_dtype).cast<py::array>();
}

std::string
Tensor::__repr__() const
{
    // --- hints from Python Tensor.__repr__ ---
    // skipped showing data. see commit 4bdaa5c for an old implementation of showing data.
    // ---
    std::string indent(static_cast<std::size_t>(get_config().print_indent), ' ');
    std::ostringstream lines;
    lines << '<' << class_name() << '\n';
    for (auto const& line : _repr_header_lines(indent)) {
        lines << line << '\n';
    }
    // skipped showing data. see commit 4bdaa5c for an old implementation of showing data.
    lines << '>';
    return lines.str();
}

std::string
Tensor::__str__() const
{
    auto lines = _repr_header_lines("", true);
    std::string right = class_name();
    for (auto const& line : lines) {
        right += '\n';
        right += line;
    }
    // Call C++ vert_join (ported from lasyten.tools.string.vert_join).
    return vert_join({ ascii_diagram(), right }, 'c', 'l', "   |  ");
}

VectorLike::Ptr
Tensor::clone() const
{
    return const_cast<Tensor*>(this)->copy();
}

Dtype
Tensor::vector_dtype() const
{
    return dtype;
}

std::string
Tensor::vector_device() const
{
    return device;
}

TensorBackend::Ptr
Tensor::vector_backend() const
{
    return backend;
}

BlockBackend::Scalar
Tensor::vector_norm() const
{
    return norm(shared_from_this());
}

BlockBackend::Scalar
Tensor::vector_inner(VectorLike::CPtr other, bool do_dagger) const
{
    auto const t = std::dynamic_pointer_cast<Tensor const>(other);
    if (!t) {
        throw std::invalid_argument("inner: expected a Tensor, not a different VectorLike");
    }
    return inner(shared_from_this(), t, do_dagger);
}

VectorLike::Ptr
Tensor::scaled(BlockBackend::Scalar const& a) const
{
    return scalar_multiply(a, shared_from_this());
}

VectorLike::Ptr
Tensor::axpy(BlockBackend::Scalar const& a, VectorLike::CPtr other) const
{
    auto const t = std::dynamic_pointer_cast<Tensor const>(other);
    if (!t) {
        throw std::invalid_argument("add: expected a Tensor, not a different VectorLike");
    }
    auto one = backend->block_backend->as_scalar(1.0);
    return linear_combination(a, shared_from_this(), one, t);
}

bool
Tensor::compatible_with(VectorLike::CPtr other) const
{
    auto const t = std::dynamic_pointer_cast<Tensor const>(other);
    if (!t) {
        return false;
    }
    auto const legs1 = legs();
    auto const legs2 = t->legs();
    if (legs1.size() != legs2.size()) {
        return false;
    }
    for (std::size_t i = 0; i < legs1.size(); ++i) {
        if (!(*legs1[i] == *legs2[i])) {
            return false;
        }
    }
    return true;
}

} // namespace lasyten
