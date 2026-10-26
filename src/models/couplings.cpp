#include <lasyten/models/couplings.h>

#include <lasyten/config.h>
#include <lasyten/symmetries/exceptions.h>
#include <lasyten/symmetries/factors/fibonacci_anyon_category.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/constructors.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/planar.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <format>
#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace lasyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

TensorBackend::Ptr
same_backend(std::vector<Site::Ptr> const& sites)
{
    if (sites.empty()) {
        throw std::invalid_argument("Need at least one site");
    }
    TensorBackend::Ptr backend = sites.front()->backend;
    for (std::size_t i = 1; i < sites.size(); ++i) {
        if (!backend || !sites[i]->backend || !(*backend == *sites[i]->backend)) {
            throw std::invalid_argument("Incompatible backends among sites");
        }
    }
    return backend;
}

std::optional<std::string>
parse_name(py::object name, std::string const& default_name)
{
    if (name.is_none()) {
        return default_name;
    }
    return name.cast<std::string>();
}

int64
site_dim(Site::Ptr const& site)
{
    return static_cast<int64>(site->dim());
}

void
check_site_count(std::vector<Site::Ptr> const& sites, int64 expected)
{
    if (static_cast<int64>(sites.size()) != expected) {
        throw std::invalid_argument(
          std::format("Invalid number of sites. Expected {}, got {}", expected, sites.size()));
    }
}

bool
legs_equal(Leg::Ptr const& a, Leg::Ptr const& b)
{
    return a && b && a->operator==(*b);
}

Leg::Ptr
as_leg(ElementarySpace::Ptr space)
{
    return std::static_pointer_cast<Leg>(space);
}

LevelsSpec
levels_from_label_dict(SymmetricTensorPtr const& tensor,
                       std::map<std::string, int64> const& level_dict)
{
    LevelsSpec spec(static_cast<std::size_t>(tensor->num_legs), std::nullopt);
    auto labels = tensor->labels();
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (!labels[i].has_value()) {
            continue;
        }
        auto it = level_dict.find(*labels[i]);
        if (it != level_dict.end()) {
            spec[i] = it->second;
        }
    }
    return spec;
}

BendRight
make_bend_right_dict(int64 num_legs, std::map<int64, bool> const& values)
{
    std::vector<std::optional<bool>> br(static_cast<std::size_t>(num_legs), std::nullopt);
    for (auto const& [idx, val] : values) {
        int64 pos = idx;
        if (pos < 0) {
            pos += num_legs;
        }
        if (pos < 0 || pos >= num_legs) {
            throw std::out_of_range("bend_right index out of range");
        }
        br[static_cast<std::size_t>(pos)] = val;
    }
    return br;
}

SymmetricTensorPtr
as_symmetric(TensorPtr tensor)
{
    if (!tensor) {
        throw std::invalid_argument("Expected non-null tensor");
    }
    return std::dynamic_pointer_cast<SymmetricTensor>(tensor);
}

SymmetricTensorPtr
compose_tensors(SymmetricTensorPtr tensor1,
                SymmetricTensorPtr tensor2,
                std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt)
{
    auto res = compose(tensor1, tensor2, std::nullopt, std::move(relabel2));
    if (std::holds_alternative<BlockBackend::Scalar>(res)) {
        throw std::runtime_error("compose unexpectedly returned a scalar");
    }
    return as_symmetric(std::get<TensorPtr>(std::move(res)));
}

/// Swap two adjacent `Coupling::factorization` tensors, braiding their physical ``(p, p*)``
/// legs past each other as a single unit. `Wa`'s `wL` and `Wb`'s `wR` (the coupling's own
/// boundary legs, possibly non-trivial) are left untouched; a fresh bond is created between the
/// two returned tensors. `over` selects the same chirality convention as the `p{pos}`/`p{pos}*`
/// vs. `p{pos+1}`/`p{pos+1}*` levels used to braid a fully-contracted coupling tensor: ``true``
/// means `Wa`'s site braids over `Wb`'s.
std::pair<SymmetricTensorPtr, SymmetricTensorPtr>
swap_adjacent_factors(SymmetricTensorPtr const& Wa, SymmetricTensorPtr const& Wb, bool over)
{
    // Bring each tensor into a shape where the wR/wL bond is the sole domain/codomain leg, so
    // `compose_tensors` can contract it without an ambiguous implicit permutation (mirrors the
    // per-tensor prep in `Coupling::to_tensor`).
    SymmetricTensorPtr Wa_prepped =
      as_symmetric(permute_legs(Wa,
                                std::vector<LegRef>{ "wL", "p*", "p" },
                                std::vector<LegRef>{ "wR" },
                                std::nullopt,
                                false));
    Wa_prepped->relabel({ { "p", "p0" }, { "p*", "p0*" } });
    SymmetricTensorPtr Wb_prepped =
      as_symmetric(permute_legs(Wb,
                                std::vector<LegRef>{ "wL" },
                                std::vector<LegRef>{ "p*", "wR", "p" },
                                std::nullopt,
                                true));
    SymmetricTensorPtr T =
      compose_tensors(Wa_prepped,
                      Wb_prepped,
                      std::map<std::string, std::string>{ { "p", "p1" }, { "p*", "p1*" } });

    std::map<std::string, int64> level_dict{
        { "p0", over ? 1 : 0 },
        { "p0*", over ? 1 : 0 },
        { "p1", over ? 0 : 1 },
        { "p1*", over ? 0 : 1 },
    };
    T = as_symmetric(permute_legs(T,
                                  std::vector<LegRef>{ "wL", "p1", "p0" },
                                  std::vector<LegRef>{ "p1*", "p0*", "wR" },
                                  levels_from_label_dict(T, level_dict),
                                  true));

    auto [left, right] =
      horizontal_factorization(T, 2, 1, OptionalLabels{ "wR", "wL" }, std::nullopt);
    SymmetricTensorPtr Wleft = as_symmetric(std::move(left));
    SymmetricTensorPtr Wright = as_symmetric(std::move(right));
    Wleft->relabel({ { "p1", "p" }, { "p1*", "p*" } });
    Wright->relabel({ { "p0", "p" }, { "p0*", "p*" } });
    return { Wleft, Wright };
}

py::object
np_multiply(py::object a, py::object b)
{
    return numpy().attr("multiply")(std::move(a), std::move(b));
}

py::object
np_add(py::object a, py::object b)
{
    return numpy().attr("add")(std::move(a), std::move(b));
}

py::object
occupation_jw(OccupationDOF::Ptr const& occ)
{
    if (auto ferm = std::dynamic_pointer_cast<FermionicDOF>(occ)) {
        return ferm->JW;
    }
    return numpy().attr("diag")(numpy().attr("ones")(site_dim(occ)));
}

Symmetry::Ptr
fibonacci_symmetry()
{
    static Symmetry::Ptr sym = [] {
        auto cat = std::make_shared<FibonacciAnyonCategory>();
        return cat->as_Symmetry();
    }();
    return sym;
}

Coupling
coupling_from_dense_block(py::array h,
                          std::vector<Site::Ptr> sites,
                          std::optional<std::string> name)
{
    return Coupling::from_dense_block(
      py::cast<py::object>(h), sites, name, std::nullopt, true, std::nullopt);
}

py::tuple
default_species_pair()
{
    return py::make_tuple(all_species_sentinel(), all_species_sentinel());
}

py::array
quadratic_coupling_numpy(std::vector<Site::Ptr> const& sites, bool is_pairing, py::object species)
{
    check_site_count(sites, 2);
    auto occ_i = std::dynamic_pointer_cast<OccupationDOF>(sites[0]);
    auto occ_j = std::dynamic_pointer_cast<OccupationDOF>(sites[1]);
    if (!occ_i || !occ_j) {
        throw std::invalid_argument("Expected occupation sites");
    }
    bool boson_i = static_cast<bool>(std::dynamic_pointer_cast<BosonicDOF>(sites[0]));
    bool boson_j = static_cast<bool>(std::dynamic_pointer_cast<BosonicDOF>(sites[1]));
    if (boson_i != boson_j) {
        throw SymmetryError("Bosonic and fermionic sites are incompatible and cannot be combined "
                            "for constructing couplings.");
    }

    py::tuple species_tuple =
      species.is_none() ? default_species_pair() : species.cast<py::tuple>();
    py::list species_i;
    py::list species_j;
    if (is_all_species(species_tuple[0].cast<py::object>())) {
        for (int64 k = 0; k < occ_i->num_species; ++k) {
            species_i.append(k);
        }
    } else {
        species_i = py::list(species_tuple[0]);
    }
    if (is_all_species(species_tuple[1].cast<py::object>())) {
        for (int64 k = 0; k < occ_j->num_species; ++k) {
            species_j.append(k);
        }
    } else {
        species_j = py::list(species_tuple[1]);
    }
    if (species_i.empty() || species_j.empty()) {
        auto np = numpy();
        return np
          .attr("zeros")(
            py::make_tuple(site_dim(occ_i), site_dim(occ_j), site_dim(occ_j), site_dim(occ_i)))
          .cast<py::array>();
    }

    auto np = numpy();
    py::object h = np.attr("zeros")(
      py::make_tuple(site_dim(occ_i), site_dim(occ_j), site_dim(occ_j), site_dim(occ_i)));
    py::object jw = occupation_jw(occ_i);
    for (py::ssize_t idx = 0; idx < py::len(species_i); ++idx) {
        py::object k_i = species_i[py::int_(idx)];
        py::object k_j = species_j[py::int_(idx)];
        py::object op_i = occ_i->get_creator_numpy(k_i, true);
        py::object op_j = is_pairing ? occ_i->get_creator_numpy(k_j, true)
                                     : occ_i->get_annihilator_numpy(k_j, true);
        h = np_add(h,
                   np_multiply(np.attr("matmul")(op_i, jw).attr("__getitem__")(
                                 py::make_tuple(py::slice(), py::none(), py::none())),
                               op_j.attr("__getitem__")(py::make_tuple(
                                 py::none(), py::slice(), py::slice(), py::none()))));
    }
    py::array h_conj = np.attr("transpose")(h.attr("conj")(), py::make_tuple(3, 2, 1, 0));
    return (h + h_conj).cast<py::array>();
}

py::object
structural_key(Coupling const& coupling)
{
    py::list factor_keys;
    for (auto const& t : coupling.factorization) {
        py::list shape_list;
        for (auto d : t->shape) {
            shape_list.append(d);
        }
        py::list labels_list;
        for (auto const& lab : t->labels()) {
            labels_list.append(lab.has_value() ? py::object(py::str(*lab)) : py::none());
        }
        py::list codomain_spaces;
        for (auto const& f : t->codomain->factors) {
            codomain_spaces.append(
              freeze(py::cast(space_to_dict(std::dynamic_pointer_cast<ElementarySpace>(f)))));
        }
        py::list domain_spaces;
        for (auto const& f : t->domain->factors) {
            domain_spaces.append(
              freeze(py::cast(space_to_dict(std::dynamic_pointer_cast<ElementarySpace>(f)))));
        }
        factor_keys.append(py::make_tuple(py::tuple(shape_list),
                                          py::tuple(labels_list),
                                          py::str(dtype::repr(t->dtype)),
                                          py::tuple(codomain_spaces),
                                          py::tuple(domain_spaces)));
    }

    py::list site_reprs;
    for (auto const& s : coupling.sites) {
        site_reprs.append(s->repr());
    }

    py::object name_obj =
      coupling.name.has_value() ? py::object(py::str(*coupling.name)) : py::none();
    return py::make_tuple(name_obj, py::tuple(site_reprs), py::tuple(factor_keys));
}

} // namespace

std::map<std::string, py::object>
space_to_dict(ElementarySpace::Ptr space)
{
    if (!space) {
        throw std::invalid_argument("space_to_dict requires a non-null ElementarySpace");
    }

    py::list sectors;
    for (auto const& sector : space->defining_sectors) {
        py::list row;
        for (std::size_t j = 0; j < sector.len(); ++j) {
            row.append(sector.q[j]);
        }
        sectors.append(row);
    }

    py::list multiplicities;
    for (auto m : space->multiplicities) {
        multiplicities.append(m);
    }

    py::object basis_perm = py::none();
    if (space->has_custom_basis_perm()) {
        py::list bp;
        for (auto p : space->basis_perm()) {
            bp.append(p);
        }
        basis_perm = bp;
    }

    return {
        { "symmetry", py::str(space->Space::symmetry->repr()) },
        { "sectors", sectors },
        { "multiplicities", multiplicities },
        { "is_dual", py::bool_(space->Leg::is_dual) },
        { "basis_perm", basis_perm },
    };
}

py::object
freeze(py::object obj)
{
    if (py::isinstance<py::dict>(obj)) {
        py::dict d = obj.cast<py::dict>();
        std::vector<std::pair<std::string, py::object>> items;
        items.reserve(d.size());
        for (auto item : d) {
            items.emplace_back(py::str(item.first).cast<std::string>(),
                               freeze(py::reinterpret_borrow<py::object>(item.second)));
        }
        std::sort(items.begin(), items.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
        });
        py::list out;
        for (auto const& [k, v] : items) {
            out.append(py::make_tuple(k, v));
        }
        return py::tuple(out);
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::list out;
        for (auto item : obj) {
            out.append(freeze(py::reinterpret_borrow<py::object>(item)));
        }
        return py::tuple(out);
    }
    return obj;
}

Coupling::Coupling(std::vector<Site::Ptr> sites,
                   std::vector<SymmetricTensorPtr> factorization,
                   std::optional<std::string> name)
  : sites(std::move(sites))
  , factorization(std::move(factorization))
  , name(std::move(name))
{
    if (this->sites.size() != this->factorization.size()) {
        throw std::invalid_argument("factorization must have one tensor per site");
    }
    _levels.resize(this->sites.size());
    std::iota(_levels.begin(), _levels.end(), int64(1));
    test_sanity();
}

void
Coupling::test_sanity() const
{
    auto backend = same_backend(sites);
    for (std::size_t i = 0; i < sites.size(); ++i) {
        auto const& s = sites[i];
        auto const& W = factorization[i];
        s->test_sanity();
        W->test_sanity();
        if (!(*W->backend == *backend)) {
            throw std::runtime_error("Coupling factorization backend mismatch");
        }
        if (W->num_codomain_legs() != 2 || W->num_domain_legs() != 2) {
            throw std::runtime_error("Coupling tensor must have 2 codomain and 2 domain legs");
        }
        static std::vector<std::string> const expected_labels = { "wL", "p", "wR", "p*" };
        auto labels = W->labels();
        for (std::size_t j = 0; j < expected_labels.size(); ++j) {
            if (!labels[j].has_value() || *labels[j] != expected_labels[j]) {
                throw std::runtime_error("Coupling tensor labels must be ['wL', 'p', 'wR', 'p*']");
            }
        }
        if (!legs_equal(W->get_leg_co_domain("p"), as_leg(s->leg))) {
            throw std::runtime_error("Coupling physical leg mismatch");
        }
        if (!legs_equal(W->get_leg_co_domain("p*"), as_leg(s->leg))) {
            throw std::runtime_error("Coupling dual physical leg mismatch");
        }
    }
    if (!factorization.front()->get_leg("wL")->is_trivial()) {
        throw std::runtime_error("Leftmost wL leg must be trivial");
    }
    for (std::size_t i = 0; i + 1 < factorization.size(); ++i) {
        if (!legs_equal(factorization[i]->get_leg_co_domain("wR"),
                        factorization[i + 1]->get_leg_co_domain("wL"))) {
            throw std::runtime_error("Coupling virtual legs must match between neighbors");
        }
    }
    if (!factorization.back()->get_leg("wR")->is_trivial()) {
        throw std::runtime_error("Rightmost wR leg must be trivial");
    }
}

Coupling
Coupling::from_dense_block(py::object operator_,
                           std::vector<Site::Ptr> sites,
                           std::optional<std::string> name,
                           std::optional<Dtype> dtype,
                           bool understood_braiding,
                           std::optional<float64> cutoff_singular_values)
{
    auto backend = same_backend(sites);
    std::string device = sites.front()->default_device;
    for (auto const& s : sites | std::views::drop(1)) {
        if (s->default_device != device) {
            throw std::invalid_argument("All sites must share the same default_device");
        }
    }

    std::vector<Leg::Ptr> co_domain;
    co_domain.reserve(sites.size());
    for (auto const& s : sites) {
        co_domain.push_back(as_leg(s->leg));
    }
    std::vector<std::string> p_labels;
    p_labels.reserve(sites.size());
    for (std::size_t i = 0; i < sites.size(); ++i) {
        p_labels.push_back(std::format("p{}", i));
    }
    OptionalLabels labels;
    for (auto const& pl : p_labels) {
        labels.emplace_back(pl);
    }
    for (auto it = p_labels.rbegin(); it != p_labels.rend(); ++it) {
        labels.emplace_back(*it + "*");
    }

    auto codomain = std::make_shared<TensorProduct>(co_domain);
    auto op =
      SymmetricTensor::from_dense_block(backend->block_backend->as_block(operator_, dtype, device),
                                        codomain,
                                        codomain,
                                        backend,
                                        labels,
                                        dtype,
                                        device,
                                        1e-6,
                                        understood_braiding);
    return from_tensor(op, sites, name, cutoff_singular_values);
}

Coupling
Coupling::from_tensor(SymmetricTensorPtr operator_,
                      std::vector<Site::Ptr> sites,
                      std::optional<std::string> name,
                      std::optional<float64> cutoff)
{
    if (!operator_) {
        throw std::invalid_argument("operator must be non-null");
    }
    if (!(*operator_->backend == *same_backend(sites))) {
        throw std::invalid_argument("operator backend must match sites");
    }
    for (std::size_t i = 0; i < sites.size(); ++i) {
        if (!legs_equal(operator_->codomain->factors[i], as_leg(sites[i]->leg))) {
            throw std::invalid_argument("operator codomain must match site legs");
        }
        if (!legs_equal(operator_->domain->factors[i], as_leg(sites[i]->leg))) {
            throw std::invalid_argument("operator domain must match site legs");
        }
    }

    std::vector<std::string> p_labels;
    p_labels.reserve(sites.size());
    for (std::size_t i = 0; i < sites.size(); ++i) {
        p_labels.push_back(std::format("p{}", i));
    }
    OptionalLabels expected_labels;
    for (auto const& pl : p_labels) {
        expected_labels.emplace_back(pl);
    }
    for (auto it = p_labels.rbegin(); it != p_labels.rend(); ++it) {
        expected_labels.emplace_back(*it + "*");
    }
    if (operator_->labels() != expected_labels) {
        throw std::invalid_argument("operator labels do not match expected p-label order");
    }

    if (!cutoff) {
        cutoff = get_config().coupling_cutoff;
    }
    // Truncated SVD divides by ||S|| and keeps at least one singular value. A (numerically) zero
    // operator hits divide-by-zero and cannot satisfy svd_min vs chi_min, so fall back to QR.
    if (norm(TensorCPtr{ operator_ }).as_float64() <= *cutoff) {
        cutoff.reset();
    }

    std::vector<SymmetricTensorPtr> factorization;
    if (sites.size() == 1) {
        SymmetricTensorPtr W =
          as_symmetric(add_trivial_leg(operator_, std::nullopt, 0, std::nullopt, "wL"));
        W = as_symmetric(add_trivial_leg(W, std::nullopt, std::nullopt, 1, "wR"));
        W->relabel({ { "p0", "p" }, { "p0*", "p*" } });
        factorization = { W };
    } else {
        SymmetricTensorPtr rest = operator_;
        auto [W0, rest0] =
          horizontal_factorization(rest, 1, 1, OptionalLabels{ "wR", "wL" }, cutoff);
        W0 = as_symmetric(W0);
        rest = as_symmetric(rest0);
        W0->relabel({ { "p0", "p" }, { "p0*", "p*" } });
        factorization.push_back(
          as_symmetric(add_trivial_leg(W0, std::nullopt, 0, std::nullopt, "wL")));

        for (std::size_t i = 1; i + 1 < sites.size(); ++i) {
            auto [Wi, resti] =
              horizontal_factorization(rest, 2, 1, OptionalLabels{ "wR", "wL" }, cutoff);
            Wi = as_symmetric(Wi);
            rest = as_symmetric(resti);
            Wi->relabel({ { std::format("p{}", i), "p" }, { std::format("p{}*", i), "p*" } });
            factorization.push_back(as_symmetric(Wi));
        }

        if (rest->num_codomain_legs() != 2 || rest->num_domain_legs() != 1) {
            throw std::runtime_error("Unexpected remaining legs after horizontal factorization");
        }
        rest->relabel({ { std::format("p{}", sites.size() - 1), "p" },
                        { std::format("p{}*", sites.size() - 1), "p*" } });
        factorization.push_back(
          as_symmetric(add_trivial_leg(rest, std::nullopt, std::nullopt, 1, "wR")));
    }
    return Coupling(sites, factorization, name);
}

SymmetricTensorPtr
Coupling::to_tensor() const
{
    SymmetricTensorPtr res = as_symmetric(
      squeeze_legs(factorization.front(), std::vector<LegRef>{ LegRef{ std::string("wL") } }));
    res = as_symmetric(permute_legs(res,
                                    std::vector<LegRef>{ LegRef{ int64(-1) }, LegRef{ int64(0) } },
                                    std::vector<LegRef>{ LegRef{ int64(1) } },
                                    std::nullopt,
                                    false));
    res->relabel({ { "p", "p0" }, { "p*", "p0*" } });

    int64 const num_sites = static_cast<int64>(sites.size());
    for (int64 i = 1; i < num_sites; ++i) {
        SymmetricTensorPtr W =
          as_symmetric(permute_legs(factorization[static_cast<std::size_t>(i)],
                                    std::vector<LegRef>{ std::string("wL") },
                                    std::vector<LegRef>{ "p*", "wR", "p" },
                                    std::nullopt,
                                    true));
        std::map<std::string, std::string> relabel2{
            { "p", std::format("p{}", i) },
            { "p*", std::format("p{}*", i) },
        };
        res = compose_tensors(res, W, relabel2);

        std::vector<LegRef> codomain;
        codomain.push_back(LegRef{ int64(-1) });
        for (int64 j = 0; j < 2 * i; ++j) {
            codomain.push_back(LegRef{ j });
        }
        codomain.push_back(LegRef{ 2 * i });
        BendRight bend = make_bend_right_dict(res->num_legs, { { -1, false }, { -3, true } });
        res = as_symmetric(permute_legs(
          res, codomain, std::vector<LegRef>{ LegRef{ int64(-2) } }, std::nullopt, bend));
    }

    res = as_symmetric(squeeze_legs(res, std::vector<LegRef>{ LegRef{ std::string("wR") } }));
    std::vector<LegRef> codom_labels;
    std::vector<LegRef> dom_labels;
    for (int64 i = 0; i < num_sites; ++i) {
        codom_labels.emplace_back(std::format("p{}", i));
        dom_labels.emplace_back(std::format("p{}*", i));
    }
    return as_symmetric(permute_legs(res, codom_labels, dom_labels, std::nullopt, false));
}

py::array
Coupling::to_numpy(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                   py::object dtype,
                   bool understood_braiding) const
{
    return to_tensor()->to_numpy(std::move(leg_order), dtype, understood_braiding);
}

Coupling
Coupling::stretch_with_identities(std::vector<Site::Ptr> const& all_sites,
                                  std::vector<int64> const& coupling_positions) const
{
    int64 const n = static_cast<int64>(factorization.size());
    if (static_cast<int64>(coupling_positions.size()) != n) {
        throw std::invalid_argument(std::format(
          "need {} positions (one per coupling tensor), got {}", n, coupling_positions.size()));
    }
    for (std::size_t i = 1; i < coupling_positions.size(); ++i) {
        if (coupling_positions[i] <= coupling_positions[i - 1]) {
            throw std::invalid_argument("`coupling_positions` must be strictly ascending");
        }
    }
    int64 const num_all_sites = static_cast<int64>(all_sites.size());
    for (std::size_t i = 0; i < coupling_positions.size(); ++i) {
        if (coupling_positions[i] < 0 || coupling_positions[i] >= num_all_sites) {
            throw std::invalid_argument(
              std::format("`coupling_positions[{}]` = {} is out of range for `all_sites` of "
                          "length {}",
                          i,
                          coupling_positions[i],
                          num_all_sites));
        }
    }
    for (std::size_t i = 0; i < sites.size(); ++i) {
        if (!legs_equal(as_leg(sites[i]->leg),
                        as_leg(all_sites[static_cast<std::size_t>(coupling_positions[i])]->leg))) {
            throw std::invalid_argument(
              std::format("physical leg mismatch at position {}", coupling_positions[i]));
        }
    }

    if (coupling_positions.empty()) {
        throw std::invalid_argument(
          "`coupling_positions` is empty; a `Coupling` with no factorization tensors cannot be "
          "stretched");
    }
    int64 const start = coupling_positions.front();
    int64 const stop = coupling_positions.back() + 1;
    std::map<int64, SymmetricTensorPtr> by_position;
    for (std::size_t i = 0; i < factorization.size(); ++i) {
        by_position[coupling_positions[i]] = factorization[i];
    }

    std::vector<SymmetricTensorPtr> new_factorization;
    ElementarySpace::Ptr wR_space = nullptr;
    for (int64 pos = start; pos < stop; ++pos) {
        SymmetricTensorPtr tensor;
        auto it = by_position.find(pos);
        if (it == by_position.end()) {
            tensor = all_sites[static_cast<std::size_t>(pos)]->identity_tensor(wR_space);
        } else {
            tensor = it->second;
        }
        new_factorization.push_back(tensor);
        wR_space = std::dynamic_pointer_cast<ElementarySpace>(tensor->get_leg_co_domain("wR"));
    }

    std::vector<Site::Ptr> new_sites;
    new_sites.reserve(static_cast<std::size_t>(stop - start));
    for (int64 pos = start; pos < stop; ++pos) {
        new_sites.push_back(all_sites[static_cast<std::size_t>(pos)]);
    }
    return Coupling(new_sites, new_factorization, name);
}

std::vector<Coupling>
Coupling::elementary_decomposition() const
{
    std::size_t const n = factorization.size();
    // Projections of bond b (between tensors b and b+1) onto its k-th sector, as seen from the
    // wR leg of tensor b (masks_R) and from the wL leg of tensor b + 1 (masks_L). These legs are
    // dual to each other, but share the basis order, so the same indices select the sector.
    std::vector<std::vector<MaskPtr>> masks_R;
    std::vector<std::vector<MaskPtr>> masks_L;
    for (std::size_t b = 0; b + 1 < n; ++b) {
        auto leg_R = std::dynamic_pointer_cast<ElementarySpace>(factorization[b]->get_leg("wR"));
        auto leg_L =
          std::dynamic_pointer_cast<ElementarySpace>(factorization[b + 1]->get_leg("wL"));
        if (!leg_R || !leg_L) {
            throw std::runtime_error("virtual legs must be ElementarySpaces");
        }
        auto const& leg = leg_L;
        std::map<Sector, std::vector<int64>> indices_of_sector;
        auto const basis_sectors = leg->sectors_of_basis();
        for (std::size_t i = 0; i < basis_sectors.size(); ++i) {
            indices_of_sector[basis_sectors[i]].push_back(static_cast<int64>(i));
        }
        auto& bond_masks_R = masks_R.emplace_back();
        auto& bond_masks_L = masks_L.emplace_back();
        for (auto const& [sector, inds] : indices_of_sector) {
            bond_masks_R.push_back(
              Mask::from_indices(py::cast(inds), leg_R, factorization[b]->backend));
            bond_masks_L.push_back(
              Mask::from_indices(py::cast(inds), leg_L, factorization[b]->backend));
        }
    }

    std::vector<Coupling> result;
    std::vector<std::size_t> choice(masks_R.size(), 0);
    for (auto const& bond_masks : masks_R) {
        if (bond_masks.empty()) {
            return result;
        }
    }
    while (true) {
        std::vector<SymmetricTensorPtr> tensors;
        bool vanishes = false;
        for (std::size_t i = 0; i < n && !vanishes; ++i) {
            TensorPtr W = factorization[i];
            if (i > 0) {
                W = apply_mask(W, masks_L[i - 1][choice[i - 1]], "wL");
            }
            if (i + 1 < n) {
                W = apply_mask(W, masks_R[i][choice[i]], "wR");
            }
            vanishes = norm(TensorCPtr{ W }).as_float64() == 0.;
            tensors.push_back(as_symmetric(W));
        }
        if (!vanishes) {
            result.emplace_back(sites, std::move(tensors), name);
        }
        // advance the odometer over the sectors of all bonds
        std::size_t b = 0;
        while (b < choice.size() && ++choice[b] == masks_R[b].size()) {
            choice[b++] = 0;
        }
        if (b == choice.size()) {
            break;
        }
    }
    return result;
}

Coupling
Coupling::permute(std::vector<int64> const& permutation, std::optional<LevelsSpec> levels) const
{
    int64 const n = static_cast<int64>(sites.size());
    if (!is_permutation(permutation)) {
        throw std::invalid_argument(
          std::format("`permutation` must be a permutation of range({}), got {}",
                      n,
                      format_like_list(py::cast(permutation))));
    }

    LevelsSpec levels_state;
    if (levels.has_value()) {
        levels_state = *levels;
    } else {
        levels_state.reserve(static_cast<std::size_t>(n));
        for (auto lv : _levels) {
            levels_state.push_back(lv);
        }
    }
    if (static_cast<int64>(levels_state.size()) != n) {
        throw std::invalid_argument(
          std::format("need {} `levels`, one per site, got {}", n, levels_state.size()));
    }

    for (auto const& [cached_key, cached_coupling] : _permuted) {
        if (cached_key == permutation) {
            return cached_coupling;
        }
    }

    std::vector<int64> swap_positions = permutation_as_swaps(permutation);

    std::vector<Site::Ptr> permuted_sites = sites;
    std::vector<SymmetricTensorPtr> permuted_factorization = factorization;

    for (int64 pos : swap_positions) {
        auto const lo = static_cast<std::size_t>(pos);
        auto const hi = lo + 1;
        if (!levels_state[lo].has_value() || !levels_state[hi].has_value()) {
            throw BraidChiralityUnspecifiedError("Sites that braid must have specified levels.");
        }
        int64 const level_1 = *levels_state[lo];
        int64 const level_2 = *levels_state[hi];
        if (level_1 == level_2) {
            throw BraidChiralityUnspecifiedError("Sites that braid can not have the same level.");
        }
        bool const over = level_1 > level_2;

        auto [Wleft, Wright] =
          swap_adjacent_factors(permuted_factorization[lo], permuted_factorization[hi], over);
        permuted_factorization[lo] = std::move(Wleft);
        permuted_factorization[hi] = std::move(Wright);

        std::swap(permuted_sites[lo], permuted_sites[hi]);
        std::swap(levels_state[lo], levels_state[hi]);
    }

    Coupling result(permuted_sites, permuted_factorization, name);
    result._levels.resize(permutation.size());
    for (std::size_t new_pos = 0; new_pos < permutation.size(); ++new_pos) {
        result._levels[new_pos] = _levels[static_cast<std::size_t>(permutation[new_pos])];
    }
    _permuted.emplace_back(permutation, result);
    return result;
}

std::tuple<py::object, std::vector<Site::Ptr>, std::vector<SymmetricTensorPtr>>
Coupling::key() const
{
    return { structural_key(*this), sites, factorization };
}

bool
Coupling::operator==(Coupling const& other) const
{
    if (structural_key(*this).not_equal(structural_key(other))) {
        return false;
    }
    for (std::size_t i = 0; i < factorization.size(); ++i) {
        if (factorization[i].get() == other.factorization[i].get()) {
            continue;
        }
        if (!almost_equal(factorization[i], other.factorization[i])) {
            return false;
        }
    }
    return true;
}

size_t
Coupling::hash() const
{
    return static_cast<size_t>(py::hash(structural_key(*this)));
}

std::string
Coupling::repr() const
{
    std::string site_names;
    for (std::size_t i = 0; i < sites.size(); ++i) {
        if (i > 0) {
            site_names += ", ";
        }
        site_names += sites[i]->repr();
    }
    std::string shapes;
    for (std::size_t i = 0; i < factorization.size(); ++i) {
        if (i > 0) {
            shapes += ", ";
        }
        shapes += "(";
        for (std::size_t j = 0; j < factorization[i]->shape.size(); ++j) {
            if (j > 0) {
                shapes += ", ";
            }
            shapes += std::to_string(static_cast<int64>(factorization[i]->shape[j]));
        }
        shapes += ")";
    }
    std::string name_repr = name.has_value() ? std::format("'{}'", *name) : "None";
    return std::format(
      "Coupling(name={}, sites=[{}], shapes=[{}])", name_repr, site_names, shapes);
}

Coupling
spin_spin_coupling(std::vector<Site::Ptr> sites,
                   float64 Jx,
                   float64 Jy,
                   float64 Jz,
                   py::object /*backend*/,
                   py::object /*device*/,
                   py::object name)
{
    check_site_count(sites, 2);
    auto spin0 = std::dynamic_pointer_cast<SpinDOF>(sites[0]);
    auto spin1 = std::dynamic_pointer_cast<SpinDOF>(sites[1]);
    if (!spin0 || !spin1) {
        throw std::invalid_argument("spin_spin_coupling requires SpinDOF sites");
    }
    auto np = numpy();
    py::object h = np.attr("zeros")(
      py::make_tuple(site_dim(spin0), site_dim(spin0), site_dim(spin1), site_dim(spin1)));
    for (int axis = 0; axis < 3; ++axis) {
        float64 pref = axis == 0 ? Jx : (axis == 1 ? Jy : Jz);
        if (pref == 0.) {
            continue;
        }
        py::object s1 =
          spin0->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), axis));
        py::object s2 =
          spin1->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), axis));
        h = np_add(h, np_multiply(py::float_(pref), np.attr("tensordot")(s1, s2, 0)));
    }
    h = np.attr("transpose")(h, py::make_tuple(0, 2, 3, 1));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "spin-spin"));
}

Coupling
spin_field_coupling(std::vector<Site::Ptr> sites,
                    float64 hx,
                    float64 hy,
                    float64 hz,
                    py::object /*backend*/,
                    py::object /*device*/,
                    py::object name)
{
    check_site_count(sites, 1);
    auto spin = std::dynamic_pointer_cast<SpinDOF>(sites[0]);
    if (!spin) {
        throw std::invalid_argument("spin_field_coupling requires SpinDOF sites");
    }
    auto np = numpy();
    py::object h = np.attr("zeros")(py::make_tuple(site_dim(spin), site_dim(spin)));
    if (hx != 0.) {
        h = np_add(h,
                   np_multiply(py::float_(hx),
                               spin->spin_vector.attr("__getitem__")(
                                 py::make_tuple(py::slice(), py::slice(), 0))));
    }
    if (hy != 0.) {
        h = np_add(h,
                   np_multiply(py::float_(hy),
                               spin->spin_vector.attr("__getitem__")(
                                 py::make_tuple(py::slice(), py::slice(), 1))));
    }
    if (hz != 0.) {
        h = np_add(h,
                   np_multiply(py::float_(hz),
                               spin->spin_vector.attr("__getitem__")(
                                 py::make_tuple(py::slice(), py::slice(), 2))));
    }
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "spin-field"));
}

Coupling
aklt_coupling(std::vector<Site::Ptr> sites,
              float64 J,
              py::object /*backend*/,
              py::object /*device*/,
              py::object name)
{
    check_site_count(sites, 2);
    auto spin0 = std::dynamic_pointer_cast<SpinDOF>(sites[0]);
    auto spin1 = std::dynamic_pointer_cast<SpinDOF>(sites[1]);
    if (!spin0 || !spin1) {
        throw std::invalid_argument("aklt_coupling requires SpinDOF sites");
    }
    auto np = numpy();
    py::object S_dot_S =
      np.attr("tensordot")(spin0->spin_vector,
                           spin1->spin_vector,
                           py::make_tuple(py::make_tuple(2), py::make_tuple(2)));
    S_dot_S = np.attr("transpose")(S_dot_S, py::make_tuple(0, 2, 3, 1));
    py::object S_dot_S_square = np.attr("tensordot")(
      S_dot_S, S_dot_S, py::make_tuple(py::make_tuple(3, 2), py::make_tuple(0, 1)));
    py::object h = np_add(np_multiply(py::float_(J), S_dot_S),
                          np_multiply(py::float_(J / 3.0), S_dot_S_square));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "AKLT"));
}

Coupling
heisenberg_coupling(std::vector<Site::Ptr> sites,
                    float64 J,
                    py::object backend,
                    py::object device,
                    py::object name)
{
    return spin_spin_coupling(
      sites, J, J, J, backend, device, name.is_none() ? py::str("S.S") : name);
}

Coupling
chiral_3spin_coupling(std::vector<Site::Ptr> sites,
                      float64 chi,
                      py::object /*backend*/,
                      py::object /*device*/,
                      py::object name)
{
    check_site_count(sites, 3);
    auto spin0 = std::dynamic_pointer_cast<SpinDOF>(sites[0]);
    auto spin1 = std::dynamic_pointer_cast<SpinDOF>(sites[1]);
    auto spin2 = std::dynamic_pointer_cast<SpinDOF>(sites[2]);
    if (!spin0 || !spin1 || !spin2) {
        throw std::invalid_argument("chiral_3spin_coupling requires SpinDOF sites");
    }
    auto np = numpy();
    py::object SxS = np.attr("cross")(
      spin1->spin_vector.attr("__getitem__")(
        py::make_tuple(py::slice(), py::none(), py::none(), py::slice(), py::slice())),
      spin2->spin_vector.attr("__getitem__")(
        py::make_tuple(py::none(), py::slice(), py::slice(), py::none(), py::slice())),
      4);
    py::object h = np_multiply(
      py::float_(chi), np.attr("tensordot")(spin0->spin_vector, SxS, py::make_tuple(-1, -1)));
    h = np.attr("transpose")(h, py::make_tuple(0, 2, 3, 4, 5, 1));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "S.SxS"));
}

Coupling
chemical_potential(std::vector<Site::Ptr> sites,
                   float64 mu,
                   py::object species,
                   py::object /*backend*/,
                   py::object /*device*/,
                   py::object name)
{
    check_site_count(sites, 1);
    auto occ = std::dynamic_pointer_cast<OccupationDOF>(sites[0]);
    if (!occ) {
        throw std::invalid_argument("chemical_potential requires bosonic or fermionic sites");
    }
    auto np = numpy();
    py::object h = np_multiply(py::float_(-mu), occ->get_occupation_numpy(species));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "chem. pot."));
}

Coupling
onsite_interaction(std::vector<Site::Ptr> sites,
                   float64 U,
                   py::object species,
                   py::object /*backend*/,
                   py::object /*device*/,
                   py::object name)
{
    check_site_count(sites, 1);
    auto occ = std::dynamic_pointer_cast<OccupationDOF>(sites[0]);
    if (!occ) {
        throw std::invalid_argument("onsite_interaction requires bosonic or fermionic sites");
    }
    auto np = numpy();
    py::object n_i = occ->get_occupation_numpy(species);
    py::object h = np_multiply(py::float_(0.5 * U), np.attr("matmul")(n_i, n_i));
    return coupling_from_dense_block(
      h.cast<py::array>(), sites, parse_name(name, "onsite interaction"));
}

Coupling
density_density_interaction(std::vector<Site::Ptr> sites,
                            float64 V,
                            py::object species_i,
                            py::object species_j,
                            py::object /*backend*/,
                            py::object /*device*/,
                            py::object name)
{
    check_site_count(sites, 2);
    auto occ0 = std::dynamic_pointer_cast<OccupationDOF>(sites[0]);
    auto occ1 = std::dynamic_pointer_cast<OccupationDOF>(sites[1]);
    if (!occ0 || !occ1) {
        throw std::invalid_argument(
          "density_density_interaction requires bosonic or fermionic sites");
    }
    bool boson0 = static_cast<bool>(std::dynamic_pointer_cast<BosonicDOF>(sites[0]));
    bool boson1 = static_cast<bool>(std::dynamic_pointer_cast<BosonicDOF>(sites[1]));
    if (boson0 != boson1) {
        throw SymmetryError("Bosonic and fermionic sites are incompatible and cannot be combined "
                            "for constructing couplings.");
    }
    auto np = numpy();
    py::object n_i = occ0->get_occupation_numpy(species_i);
    py::object n_j = occ1->get_occupation_numpy(species_j);
    py::object h = np_multiply(
      py::float_(V),
      np.attr("multiply")(
        n_i.attr("__getitem__")(py::make_tuple(py::slice(), py::none(), py::none(), py::slice())),
        n_j.attr("__getitem__")(
          py::make_tuple(py::none(), py::slice(), py::slice(), py::none()))));
    return coupling_from_dense_block(
      h.cast<py::array>(), sites, parse_name(name, "density-density"));
}

Coupling
hopping(std::vector<Site::Ptr> sites,
        float64 t,
        py::object species,
        py::object /*backend*/,
        py::object /*device*/,
        py::object name)
{
    py::array h = quadratic_coupling_numpy(sites, false, species);
    auto np = numpy();
    return coupling_from_dense_block(
      np_multiply(py::float_(-t), np.attr("array")(h)).cast<py::array>(),
      sites,
      parse_name(name, "hopping"));
}

Coupling
pairing(std::vector<Site::Ptr> sites,
        float64 Delta,
        py::object species,
        py::object /*backend*/,
        py::object /*device*/,
        py::object name)
{
    py::array h = quadratic_coupling_numpy(sites, true, species);
    auto np = numpy();
    return coupling_from_dense_block(
      np_multiply(py::float_(Delta), np.attr("array")(h)).cast<py::array>(),
      sites,
      parse_name(name, "pairing"));
}

Coupling
onsite_pairing(std::vector<Site::Ptr> sites,
               float64 Delta,
               py::object species,
               py::object /*backend*/,
               py::object /*device*/,
               py::object name)
{
    check_site_count(sites, 1);
    auto occ = std::dynamic_pointer_cast<OccupationDOF>(sites[0]);
    if (!occ) {
        throw std::invalid_argument("onsite_pairing requires bosonic or fermionic sites");
    }
    py::tuple species_tuple =
      species.is_none() ? default_species_pair() : species.cast<py::tuple>();
    py::list species_1;
    py::list species_2;
    if (is_all_species(species_tuple[0].cast<py::object>())) {
        for (int64 k = 0; k < occ->num_species; ++k) {
            species_1.append(k);
        }
    } else {
        species_1 = py::list(species_tuple[0]);
    }
    if (is_all_species(species_tuple[1].cast<py::object>())) {
        for (int64 k = 0; k < occ->num_species; ++k) {
            species_2.append(k);
        }
    } else {
        species_2 = py::list(species_tuple[1]);
    }

    auto np = numpy();
    py::object h = np.attr("zeros")(py::make_tuple(site_dim(occ), site_dim(occ)));
    for (py::ssize_t i = 0; i < py::len(species_1); ++i) {
        py::object k1 = species_1[py::int_(i)];
        py::object k2 = species_2[py::int_(i)];
        py::object a_i_hc = occ->get_creator_numpy(k1, true);
        py::object a_j_hc = occ->get_creator_numpy(k2, true);
        h = np_add(h, np_multiply(py::float_(Delta), np.attr("matmul")(a_i_hc, a_j_hc)));
    }
    h = h.attr("__iadd__")(np.attr("transpose")(h.attr("conj")()));
    return coupling_from_dense_block(
      h.cast<py::array>(), sites, parse_name(name, "onsite pairing"));
}

Coupling
clock_clock_coupling(std::vector<Site::Ptr> sites,
                     float64 Jx,
                     float64 Jz,
                     py::object /*backend*/,
                     py::object /*device*/,
                     py::object name)
{
    check_site_count(sites, 2);
    auto clock0 = std::dynamic_pointer_cast<ClockDOF>(sites[0]);
    auto clock1 = std::dynamic_pointer_cast<ClockDOF>(sites[1]);
    if (!clock0 || !clock1) {
        throw std::invalid_argument("clock_clock_coupling requires ClockDOF sites");
    }
    auto np = numpy();
    py::object X_i =
      clock0->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0));
    py::object Z_i =
      clock0->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    py::object X_j =
      clock1->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0));
    py::object Z_j =
      clock1->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    py::object h = np.attr("zeros")(
      py::make_tuple(site_dim(clock0), site_dim(clock1), site_dim(clock1), site_dim(clock0)));
    if (Jx != 0.) {
        h = np_add(h,
                   np_multiply(
                     py::float_(Jx),
                     np.attr("multiply")(X_i.attr("__getitem__")(py::make_tuple(
                                           py::slice(), py::none(), py::none(), py::slice())),
                                         np.attr("transpose")(X_j.attr("conj")())
                                           .attr("__getitem__")(py::make_tuple(
                                             py::none(), py::slice(), py::slice(), py::none())))));
    }
    if (Jz != 0.) {
        h = np_add(h,
                   np_multiply(
                     py::float_(Jz),
                     np.attr("multiply")(Z_i.attr("__getitem__")(py::make_tuple(
                                           py::slice(), py::none(), py::none(), py::slice())),
                                         np.attr("transpose")(Z_j.attr("conj")())
                                           .attr("__getitem__")(py::make_tuple(
                                             py::none(), py::slice(), py::slice(), py::none())))));
    }
    h = h.attr("__iadd__")(np.attr("transpose")(h.attr("conj")(), py::make_tuple(3, 2, 1, 0)));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "clock-clock"));
}

Coupling
clock_field_coupling(std::vector<Site::Ptr> sites,
                     std::optional<float64> hx,
                     std::optional<float64> hz,
                     py::object /*backend*/,
                     py::object /*device*/,
                     py::object name)
{
    check_site_count(sites, 1);
    auto clock = std::dynamic_pointer_cast<ClockDOF>(sites[0]);
    if (!clock) {
        throw std::invalid_argument("clock_field_coupling requires ClockDOF sites");
    }
    float64 hx_val = hx.value_or(0.);
    float64 hz_val = hz.value_or(0.);
    auto np = numpy();
    py::object X =
      clock->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0));
    py::object Z =
      clock->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    py::object h =
      np_add(np_multiply(py::float_(hx_val), np_add(X, np.attr("transpose")(X.attr("conj")()))),
             np_multiply(py::float_(hz_val), np_add(Z, np.attr("transpose")(Z.attr("conj")()))));
    return coupling_from_dense_block(h.cast<py::array>(), sites, parse_name(name, "clock-field"));
}

Coupling
sector_projection_coupling(std::vector<Site::Ptr> sites,
                           float64 J,
                           Sector sector,
                           py::object name,
                           py::object /*backend*/,
                           py::object /*device*/)
{
    auto backend = same_backend(sites);
    std::string device = sites.front()->default_device;
    for (auto const& s : sites) {
        if (s->default_device != device) {
            throw std::invalid_argument("All sites must share the same default_device");
        }
    }

    std::vector<Leg::Ptr> legs;
    legs.reserve(sites.size());
    for (auto const& s : sites) {
        legs.push_back(as_leg(s->leg));
    }
    auto codomain = std::make_shared<TensorProduct>(legs);
    OptionalLabels labels;
    for (std::size_t i = 0; i < sites.size(); ++i) {
        labels.emplace_back(std::format("p{}", i));
    }
    for (std::size_t i = sites.size(); i-- > 0;) {
        labels.emplace_back(std::format("p{}*", i));
    }

    SymmetricTensorPtr projector = SymmetricTensor::from_sector_projection(
      codomain, sector, backend, labels, std::nullopt, device);
    auto scaled =
      as_symmetric(scalar_multiply(backend->block_backend->as_scalar(J), TensorCPtr{ projector }));
    std::optional<std::string> parsed_name;
    if (!name.is_none()) {
        parsed_name = name.cast<std::string>();
    }
    return Coupling::from_tensor(scaled, sites, parsed_name);
}

Coupling
gold_coupling(std::vector<Site::Ptr> sites,
              float64 J,
              py::object backend,
              py::object device,
              py::object name)
{
    check_site_count(sites, 2);
    auto fib_sym = fibonacci_symmetry();
    for (auto const& site : sites) {
        if (!site->symmetry()->is_equivalent_to(*fib_sym)) {
            throw std::invalid_argument("gold_coupling requires Fibonacci anyon sites");
        }
        if (!site->leg->sector_decomposition_where(FibonacciAnyonCategory::tau).has_value()) {
            throw std::invalid_argument("gold_coupling requires sites with a tau sector");
        }
    }
    return sector_projection_coupling(sites,
                                      -J,
                                      FibonacciAnyonCategory::vacuum,
                                      name.is_none() ? py::str("gold") : name,
                                      backend,
                                      device);
}

} // namespace lasyten
