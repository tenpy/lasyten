#include <lasyten/models/sites.h>

#include <lasyten/symmetries/factors/fermion_parity.h>
#include <lasyten/symmetries/factors/fibonacci_anyon_category.h>
#include <lasyten/symmetries/factors/ising_anyon_category.h>
#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/symmetries/factors/su2.h>
#include <lasyten/symmetries/factors/su2_k_anyon_category.h>
#include <lasyten/symmetries/factors/sun.h>
#include <lasyten/symmetries/factors/u1.h>
#include <lasyten/symmetries/factors/zn.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tools.h>

#include <cmath>
#include <format>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace lasyten {

namespace {

int64
py_object_to_int64(py::handle obj)
{
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int64>();
    }
    if (py::hasattr(obj, "item")) {
        return obj.attr("item")().cast<int64>();
    }
    return py::module_::import("builtins").attr("int")(obj).cast<int64>();
}

py::module_
numpy_module()
{
    return py::module_::import("numpy");
}

py::module_
itertools_module()
{
    return py::module_::import("itertools");
}

int64
leg_dim(ElementarySpace::Ptr const& leg)
{
    return static_cast<int64>(leg->Space::dim);
}

Symmetry::Ptr
leg_symmetry(ElementarySpace::Ptr const& leg)
{
    return leg->Space::symmetry;
}

float64
parse_spin_S(std::optional<std::string> const& S)
{
    if (!S) {
        return 0.5;
    }
    return py::float_(py::str(*S)).cast<float64>();
}

SymmetryFactor::Ptr
first_factor(Symmetry::Ptr const& sym)
{
    if (!sym || sym->num_factors() == 0) {
        throw std::invalid_argument("Expected symmetry with at least one factor.");
    }
    return sym->factors[0];
}

bool
factor_is_test_su2(SymmetryFactor::Ptr const& factor)
{
    return dynamic_cast<_SU2 const*>(factor.get()) != nullptr;
}

bool
factor_is_sun2(SymmetryFactor::Ptr const& factor)
{
    auto const* sun = dynamic_cast<SUN const*>(factor.get());
    return sun != nullptr && sun->N == 2;
}

bool
factor_is_su2(SymmetryFactor::Ptr const& factor)
{
    return factor_is_test_su2(factor) || factor_is_sun2(factor);
}

bool
factor_is_u1(SymmetryFactor::Ptr const& factor)
{
    return dynamic_cast<U1 const*>(factor.get()) != nullptr;
}

bool
factor_is_zn(SymmetryFactor::Ptr const& factor)
{
    return dynamic_cast<ZN const*>(factor.get()) != nullptr;
}

bool
factor_is_no_symmetry(SymmetryFactor::Ptr const& factor)
{
    return dynamic_cast<NoSymmetry const*>(factor.get()) != nullptr;
}

Symmetry::Ptr
symmetry_from_factor(SymmetryFactor::Ptr factor)
{
    return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ std::move(factor) });
}

Symmetry::Ptr
symmetry_from_factors(std::vector<SymmetryFactor::Ptr> factors)
{
    return std::make_shared<Symmetry>(std::move(factors));
}

Symmetry::Ptr
fermion_conservation_law_to_symmetry(py::object conserve)
{
    if (py::isinstance<py::str>(conserve)) {
        return FermionicDOF::conservation_law_to_symmetry(conserve.cast<std::string>());
    }
    if (is_iterable(conserve)) {
        std::vector<SymmetryFactor::Ptr> sym_factors;
        int64 num_no_sym = 0;
        for (py::ssize_t k = 0; k < py::len(conserve); ++k) {
            py::object conserve_k = conserve[py::int_(k)];
            std::string ck = py::str(conserve_k).cast<std::string>();
            if (ck == "N" || ck == "Nk" || ck == "N_k") {
                sym_factors.push_back(
                  std::make_shared<U1>(std::format("species{}_fermion_occupation", k)));
            } else if (ck == "parity" || ck == "P" || ck == "Pi" || ck == "P_i") {
                sym_factors.push_back(
                  std::make_shared<ZN>(2, std::format("species{}_fermion_parity", k)));
            } else if (ck == "None" || ck == "none" || conserve_k.is_none()) {
                sym_factors.push_back(std::make_shared<NoSymmetry>());
                ++num_no_sym;
            } else {
                throw std::invalid_argument(std::format("Invalid entry in `conserve`: {}", ck));
            }
        }
        if (num_no_sym == py::len(conserve)) {
            return symmetry_from_factor(std::make_shared<FermionParity>("total_fermion_parity"));
        }
        sym_factors.push_back(std::make_shared<FermionParity>("total_fermion_parity"));
        return symmetry_from_factors(std::move(sym_factors));
    }
    throw std::invalid_argument(
      std::format("Invalid `conserve`: {}", py::str(conserve).cast<std::string>()));
}

bool
symmetry_is_fermion_parity(Symmetry::Ptr const& sym)
{
    return sym->num_factors() == 1 &&
           dynamic_cast<FermionParity const*>(sym->factors[0].get()) != nullptr;
}

ElementarySpace::Ptr
leg_from_spin_symmetry(Symmetry::Ptr sym, int64 two_S, int64 dim)
{
    auto const factor = first_factor(sym);
    if (factor_is_test_su2(factor)) {
        SectorArray defining = SectorArray::empty(sym->sector_ind_len);
        defining.push_back(Sector{ static_cast<int16_t>(two_S) });
        return ElementarySpace::from_defining_sectors(sym, defining);
    }
    if (factor_is_sun2(factor)) {
        // SUN(N=2) GT pattern for spin S is [2S, 0].
        SectorArray defining = SectorArray::empty(sym->sector_ind_len);
        defining.push_back(Sector{ static_cast<int16_t>(two_S), static_cast<int16_t>(0) });
        return ElementarySpace::from_defining_sectors(sym, defining);
    }
    if (factor_is_u1(factor)) {
        auto np = numpy_module();
        auto sectors =
          np.attr("arange")(-two_S, two_S + 2, 2).attr("reshape")(py::make_tuple(-1, 1));
        return ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    }
    if (factor_is_zn(factor)) {
        auto np = numpy_module();
        auto sectors =
          np.attr("mod")(np.attr("arange")(dim).attr("reshape")(py::make_tuple(-1, 1)), 2);
        return ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    }
    if (factor_is_no_symmetry(factor)) {
        return ElementarySpace::from_trivial_sector(dim, sym);
    }
    throw std::invalid_argument("Invalid spin conservation law for SpinSite.");
}

TensorProduct::Ptr
leg_as_product(ElementarySpace::Ptr leg)
{
    return std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ leg });
}

void
add_anyon_projectors(Site& site,
                     std::vector<std::string> const& names,
                     TensorBackend::Ptr backend,
                     std::optional<std::string> const& default_device)
{
    auto anyon = site.require_anyon();
    anyon->sector_names = names;
    auto const& decomposition = site.leg->sector_decomposition;
    if (names.size() != decomposition.size()) {
        throw std::invalid_argument("sector_names length must match leg.num_sectors.");
    }
    auto leg_tp = leg_as_product(site.leg);
    for (std::size_t i = 0; i < names.size(); ++i) {
        auto op =
          SymmetricTensor::from_sector_projection(leg_tp,
                                                  decomposition[i],
                                                  backend,
                                                  std::optional<OptionalLabels>{ { "p", "p*" } },
                                                  std::nullopt,
                                                  default_device);
        site.onsite_operators["P_" + names[i]] = std::move(op);
    }
}

void
add_filling_ops(Site& site,
                py::array n_tot,
                float64 filling,
                int64 dim,
                bool understood_braiding = false)
{
    auto np = numpy_module();
    auto dN_diag = np.attr("diag")(n_tot) - np.attr("multiply")(filling, np.attr("ones")(dim));
    auto dN = np.attr("diag")(dN_diag);
    auto dNdN = np.attr("diag")(np.attr("square")(dN_diag));
    site.add_onsite_operator("dN", dN, true, understood_braiding);
    site.add_onsite_operator("dNdN", dNdN, true, understood_braiding);
}

py::list
iter_product_unpacked(py::list const& factors)
{
    py::list result;
    auto product = itertools_module().attr("product");
    py::tuple args(py::len(factors));
    for (py::ssize_t i = 0; i < py::len(factors); ++i) {
        args[i] = factors[i];
    }
    for (py::handle item : product(*args)) {
        result.append(py::reinterpret_borrow<py::object>(item));
    }
    return result;
}

py::list
iter_product(py::iterable args)
{
    py::list result;
    for (py::handle item : itertools_module().attr("product")(args)) {
        result.append(py::reinterpret_borrow<py::object>(item));
    }
    return result;
}

std::string
format_string_list(std::vector<std::string> const& items)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << items[i];
    }
    out << "]";
    return out.str();
}

} // namespace

SpinSite::Prepared
SpinSite::prepare(float64 S, std::optional<std::string> conserve, bool use_test_su2)
{
    auto np = numpy_module();
    float64 const spin_S = S;
    int64 const two_S = static_cast<int64>(std::llround(2. * spin_S));
    int64 const dim = two_S + 1;
    if (two_S < 0) {
        throw std::invalid_argument("Negative spin.");
    }
    if (!np.attr("allclose")(two_S / 2., spin_S).cast<bool>()) {
        throw std::invalid_argument("total_spin must be half integer: 0, 1/2, 1, 3/2, ...");
    }

    auto Sz = np.attr("diag")(np.attr("add")(-spin_S, np.attr("arange")(dim)));
    auto Sp = np.attr("zeros")(py::make_tuple(dim, dim));
    for (int64 n = 0; n < dim - 1; ++n) {
        float64 const m = static_cast<float64>(n) - spin_S;
        float64 const coeff = std::sqrt(spin_S * (spin_S + 1.) - m * (m + 1.));
        Sp.attr("__setitem__")(py::make_tuple(n + 1, n), coeff);
    }
    auto spin_vector = SpinDOF::spin_vector_from_Sp(Sz, Sp);
    auto sym = SpinDOF::conservation_law_to_symmetry(conserve, use_test_su2);
    auto leg = leg_from_spin_symmetry(sym, two_S, dim);

    std::map<std::string, int64> state_labels;
    for (int64 n = 0; n < dim; ++n) {
        state_labels[py::str(py::cast(static_cast<float64>(n) - spin_S)).cast<std::string>()] = n;
    }
    state_labels["down"] = 0;
    state_labels["up"] = dim - 1;
    return { leg, spin_vector, state_labels, sym, two_S };
}

SpinlessBosonSite::Prepared
SpinlessBosonSite::prepare(py::object Nmax, py::object conserve)
{
    auto np = numpy_module();
    auto Nmax_arr =
      np.attr("atleast_1d")(np.attr("asarray")(Nmax, py::arg("dtype") = "int")).cast<py::array>();
    if (!np.attr("allclose")(Nmax_arr, np.attr("asarray")(Nmax_arr)).cast<bool>()) {
        throw std::invalid_argument("Invalid `Nmax`");
    }
    int64 const num_species = py_object_to_int64(Nmax_arr.attr("size"));
    if (!py::isinstance<py::str>(conserve) && !conserve.is_none()) {
        if (py::len(conserve) != num_species) {
            throw std::invalid_argument(
              std::format("Invalid number of entries in `conserve`: {} != {}",
                          static_cast<int64>(py::len(conserve)),
                          num_species));
        }
    }
    py::list state_ranges;
    for (int64 i = 0; i < num_species; ++i) {
        state_ranges.append(
          np.attr("arange")(py_object_to_int64(Nmax_arr.attr("__getitem__")(i)) + 1));
    }
    auto dims = np.attr("ones_like")(Nmax_arr) + Nmax_arr;
    int64 const total_dim = py_object_to_int64(np.attr("prod")(dims));

    auto sym = BosonicDOF::conservation_law_to_symmetry(conserve);
    ElementarySpace::Ptr leg;
    if (!py::isinstance<py::str>(conserve) && !conserve.is_none() && is_iterable(conserve) &&
        sym->num_factors() == py::len(conserve)) {
        py::list no_sym_idcs;
        py::list parity_sym_idcs;
        for (py::ssize_t i = 0; i < py::len(conserve); ++i) {
            auto const& factor = sym->factors[static_cast<std::size_t>(i)];
            py::object conserve_k = conserve[py::int_(i)];
            if (factor_is_no_symmetry(factor)) {
                no_sym_idcs.append(i);
            } else if (factor_is_zn(factor)) {
                parity_sym_idcs.append(i);
            } else if (!factor_is_u1(factor)) {
                throw std::invalid_argument(
                  std::format("Entry in `conserve` invalid for `SpinlessBosonSite`: {}",
                              py::str(conserve_k).cast<std::string>()));
            }
        }
        py::list sector_rows;
        for (py::handle occ : iter_product_unpacked(state_ranges)) {
            py::list occ_list;
            for (py::handle x : occ) {
                occ_list.append(x);
            }
            py::object sector = np.attr("asarray")(occ_list, py::arg("dtype") = "int");
            for (py::handle idx : no_sym_idcs) {
                sector.attr("__setitem__")(idx, 0);
            }
            for (py::handle idx : parity_sym_idcs) {
                py::object idx_val = sector.attr("__getitem__")(idx);
                sector.attr("__setitem__")(idx, np.attr("mod")(idx_val, 2));
            }
            sector_rows.append(sector);
        }
        auto sectors = np.attr("asarray")(sector_rows, py::arg("dtype") = "int");
        leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    } else {
        auto const factor = first_factor(sym);
        if (factor_is_u1(factor) || factor_is_zn(factor)) {
            bool const is_zn = factor_is_zn(factor);
            py::list sector_values;
            for (py::handle occ : iter_product_unpacked(state_ranges)) {
                py::list occ_list;
                for (py::handle x : occ) {
                    occ_list.append(x);
                }
                sector_values.append(np.attr("sum")(occ_list));
            }
            auto sectors = np.attr("asarray")(sector_values, py::arg("dtype") = "int")
                             .attr("reshape")(py::make_tuple(-1, 1));
            if (is_zn) {
                sectors = np.attr("mod")(sectors, 2);
            }
            leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
        } else if (factor_is_no_symmetry(factor)) {
            leg = ElementarySpace::from_trivial_sector(total_dim, sym);
        } else {
            throw std::invalid_argument(
              std::format("`conserve` invalid for `SpinlessBosonSite`: {}",
                          py::str(conserve).cast<std::string>()));
        }
    }

    py::list dim_prod_list;
    for (int64 i = 0; i < num_species; ++i) {
        int64 prod = 1;
        for (int64 j = i + 1; j < num_species; ++j) {
            prod *= py_object_to_int64(Nmax_arr.attr("__getitem__")(j)) + 1;
        }
        dim_prod_list.append(prod);
    }
    auto dim_prod = np.attr("asarray")(dim_prod_list, py::arg("dtype") = "int");
    std::map<std::string, int64> state_labels;
    for (py::handle occ : iter_product_unpacked(state_ranges)) {
        py::list occ_list;
        for (py::handle x : occ) {
            occ_list.append(x);
        }
        std::string label;
        if (num_species == 1) {
            label = std::to_string(py_object_to_int64(occ_list[0]));
        } else {
            py::list label_parts;
            for (py::handle x : occ_list) {
                label_parts.append(py_object_to_int64(x));
            }
            label = py::str(py::tuple(label_parts)).cast<std::string>();
        }
        auto occ_arr = np.attr("asarray")(occ_list, py::arg("dtype") = "int");
        state_labels[label] =
          py_object_to_int64(np.attr("sum")(np.attr("multiply")(occ_arr, dim_prod)));
    }
    state_labels["vac"] = 0;

    auto ops = BosonicDOF::creation_annihilation_ops_from_Nmax(Nmax_arr, total_dim);
    return { leg, Nmax_arr, ops.first, ops.second, state_labels, total_dim };
}

SpinlessFermionSite::Prepared
SpinlessFermionSite::prepare(int64 num_species, py::object conserve)
{
    if (num_species <= 0) {
        throw std::invalid_argument("Must have at least a single fermion species");
    }
    auto np = numpy_module();
    auto sym = fermion_conservation_law_to_symmetry(conserve);
    ElementarySpace::Ptr leg;
    if (symmetry_is_fermion_parity(sym)) {
        py::list sector_values;
        for (py::handle occ : itertools_module().attr("product")(
               py::make_tuple(0, 1), py::arg("repeat") = num_species)) {
            py::list occ_list;
            for (py::handle x : occ) {
                occ_list.append(x);
            }
            sector_values.append(np.attr("sum")(occ_list).attr("__mod__")(2));
        }
        auto sectors = np.attr("asarray")(sector_values, py::arg("dtype") = "int")
                         .attr("reshape")(py::make_tuple(-1, 1));
        leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    } else if (!py::isinstance<py::str>(conserve) && !conserve.is_none()) {
        if (!std::dynamic_pointer_cast<Symmetry>(sym)) {
            throw std::invalid_argument(
              "Expected product symmetry for multi-species conserve sequence");
        }
        py::list no_sym_idcs;
        py::list parity_sym_idcs;
        for (py::ssize_t i = 0; i < py::len(conserve); ++i) {
            auto const& factor = sym->factors[static_cast<std::size_t>(i)];
            py::object conserve_k = conserve[py::int_(i)];
            if (factor_is_no_symmetry(factor)) {
                no_sym_idcs.append(i);
            } else if (factor_is_zn(factor)) {
                parity_sym_idcs.append(i);
            } else if (!factor_is_u1(factor)) {
                throw std::invalid_argument(
                  std::format("Entry in `conserve` invalid for `SpinlessFermionSite`: {}",
                              py::str(conserve_k).cast<std::string>()));
            }
        }
        py::list sector_rows;
        for (py::handle occ : itertools_module().attr("product")(
               py::make_tuple(0, 1), py::arg("repeat") = num_species)) {
            py::list occ_list;
            for (py::handle x : occ) {
                occ_list.append(x);
            }
            py::object sector = np.attr("asarray")(occ_list, py::arg("dtype") = "int");
            sector = np.attr("append")(sector, np.attr("sum")(occ_list).attr("__mod__")(2));
            for (py::handle idx : no_sym_idcs) {
                sector.attr("__setitem__")(idx, 0);
            }
            sector_rows.append(sector);
        }
        auto sectors = np.attr("asarray")(sector_rows, py::arg("dtype") = "int");
        leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    } else if (factor_is_u1(first_factor(sym))) {
        py::list sector_values;
        for (py::handle occ : itertools_module().attr("product")(
               py::make_tuple(0, 1), py::arg("repeat") = num_species)) {
            py::list occ_list;
            for (py::handle x : occ) {
                occ_list.append(x);
            }
            auto fermion_number = np.attr("sum")(occ_list);
            sector_values.append(
              py::make_tuple(fermion_number, np.attr("mod")(fermion_number, 2)));
        }
        auto sectors = np.attr("asarray")(sector_values, py::arg("dtype") = "int");
        leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    } else {
        throw std::invalid_argument(std::format("`conserve` invalid for `SpinlessFermionSite`: {}",
                                                py::str(conserve).cast<std::string>()));
    }

    std::map<std::string, int64> state_labels;
    for (py::handle occ : itertools_module().attr("product")(py::make_tuple(0, 1),
                                                             py::arg("repeat") = num_species)) {
        py::list occ_list;
        for (py::handle x : occ) {
            occ_list.append(x);
        }
        std::string label;
        if (num_species == 1) {
            label = std::to_string(py_object_to_int64(occ_list[0]));
        } else {
            py::list label_parts;
            for (py::handle x : occ_list) {
                label_parts.append(py_object_to_int64(x));
            }
            label = py::str(py::tuple(label_parts)).cast<std::string>();
        }
        int64 idx = 0;
        for (py::handle x : occ_list) {
            idx = (idx << 1) | py_object_to_int64(x);
        }
        state_labels[label] = idx;
    }
    state_labels["vac"] = 0;
    state_labels["empty"] = 0;
    if (num_species == 1) {
        state_labels["full"] = 1;
    }
    auto ops = FermionicDOF::creation_annihilation_ops(num_species);
    return { leg, ops.first, ops.second, state_labels };
}

SpinHalfFermionSite::Prepared
SpinHalfFermionSite::prepare(std::string const& conserve_N,
                             std::optional<std::string> const& conserve_S,
                             bool use_test_su2)
{
    auto np = numpy_module();
    auto sym_N = FermionicDOF::conservation_law_to_symmetry(conserve_N);

    py::list parity_sectors;
    parity_sectors.append(py::make_tuple(0, 0));
    parity_sectors.append(py::make_tuple(-1, 1));
    parity_sectors.append(py::make_tuple(1, 1));
    parity_sectors.append(py::make_tuple(0, 0));
    auto sectors = np.attr("asarray")(parity_sectors, py::arg("dtype") = "int")
                     .attr("reshape")(py::make_tuple(4, 2));
    if (factor_is_u1(first_factor(sym_N))) {
        py::list number_sectors;
        number_sectors.append(py::make_tuple(0, 0, 0));
        number_sectors.append(py::make_tuple(-1, 1, 1));
        number_sectors.append(py::make_tuple(1, 1, 1));
        number_sectors.append(py::make_tuple(0, 2, 0));
        sectors = np.attr("asarray")(number_sectors, py::arg("dtype") = "int")
                    .attr("reshape")(py::make_tuple(4, 3));
    } else if (!symmetry_is_fermion_parity(sym_N)) {
        throw std::invalid_argument(
          std::format("`conserve_N` invalid for `SpinHalfFermionSite`: {}", conserve_N));
    }

    auto sym_S = SpinDOF::conservation_law_to_symmetry(conserve_S, use_test_su2);
    auto sym_S_factor = first_factor(sym_S);
    if (factor_is_u1(sym_S_factor)) {
    } else if (factor_is_zn(sym_S_factor)) {
        sectors.attr("__setitem__")(
          py::make_tuple(py::slice(), 0),
          np.attr("mod")(sectors.attr("__getitem__")(py::make_tuple(py::slice(), 0)), 2));
    } else if (factor_is_test_su2(sym_S_factor)) {
        // _SU2: empty/full jj=0, down/up jj=1 (patch down from -1 to 1).
        sectors.attr("__setitem__")(py::make_tuple(1, 0), 1);
    } else if (factor_is_sun2(sym_S_factor)) {
        // SUN(N=2): replace leading spin column with GT patterns [0,0] / [1,0].
        auto rest = sectors.attr("__getitem__")(
          py::make_tuple(py::slice(), py::slice(py::int_(1), py::none(), py::none())));
        py::list spin_gt;
        spin_gt.append(py::make_tuple(0, 0)); // empty
        spin_gt.append(py::make_tuple(1, 0)); // down
        spin_gt.append(py::make_tuple(1, 0)); // up
        spin_gt.append(py::make_tuple(0, 0)); // full
        auto spin_arr = np.attr("asarray")(spin_gt, py::arg("dtype") = "int")
                          .attr("reshape")(py::make_tuple(4, 2));
        sectors = np.attr("hstack")(py::make_tuple(spin_arr, rest));
    } else if (factor_is_no_symmetry(sym_S_factor)) {
        sectors = sectors.attr("__getitem__")(
          py::make_tuple(py::slice(), py::slice(py::int_(1), py::none(), py::none())));
    } else {
        throw std::invalid_argument(std::format(
          "`conserve_S` invalid for `SpinHalfFermionSite`: {}", conserve_S.value_or("None")));
    }

    Symmetry::Ptr sym;
    if (factor_is_no_symmetry(sym_S_factor)) {
        sym = sym_N;
    } else {
        std::vector<SymmetryFactor::Ptr> factors;
        factors.push_back(sym_S_factor);
        for (auto const& f : sym_N->factors) {
            factors.push_back(f);
        }
        sym = std::make_shared<Symmetry>(std::move(factors));
    }
    auto leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));

    py::list sz_diag;
    sz_diag.append(0);
    sz_diag.append(-0.5);
    sz_diag.append(0.5);
    sz_diag.append(0);
    auto Sz = np.attr("diag")(sz_diag);
    auto Sp = np.attr("zeros")(py::make_tuple(4, 4));
    Sp.attr("__setitem__")(py::make_tuple(2, 1), 1);
    auto spin_vector = SpinDOF::spin_vector_from_Sp(Sz, Sp);
    auto ops = FermionicDOF::creation_annihilation_ops(2);

    std::map<std::string, int64> state_labels{
        { "(0, 0)", 0 }, { "(0, 1)", 1 }, { "(1, 0)", 2 }, { "(1, 1)", 3 }, { "empty", 0 },
        { "vac", 0 },    { "down", 1 },   { "up", 2 },     { "full", 3 },
    };
    return { leg, spin_vector, ops.first, ops.second, state_labels, sym_S_factor };
}

ClockSite::Prepared
ClockSite::prepare(int64 q, std::optional<std::string> conserve)
{
    auto np = numpy_module();
    ElementarySpace::Ptr leg;
    if (conserve &&
        (conserve == "Z_N" || conserve == "ZN" || conserve == "Z_q" || conserve == "Zq")) {
        auto sym = std::make_shared<Symmetry>(
          std::vector<SymmetryFactor::Ptr>{ std::make_shared<ZN>(static_cast<int>(q), "Z_q") });
        auto sectors = np.attr("arange")(q).attr("reshape")(py::make_tuple(-1, 1));
        leg = ElementarySpace::from_basis(sym, sector_array_from_numpy(sectors));
    } else if (!conserve || conserve == "None" || conserve == "none") {
        auto sym = std::make_shared<Symmetry>(
          std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() });
        leg = ElementarySpace::from_trivial_sector(q, sym);
    } else {
        throw std::invalid_argument(std::format("Invalid `conserve`: {}", *conserve));
    }

    auto X = np.attr("eye")(q, py::arg("k") = 1) + np.attr("eye")(q, py::arg("k") = 1 - q);
    auto phase = np.attr("exp")(np.attr("multiply")(
      np.attr("divide")(np.attr("multiply")(2.0, np.attr("pi")), static_cast<float64>(q)),
      np.attr("multiply")(np.attr("arange")(q), py::cast(std::complex(0., 1.)))));
    auto Z = np.attr("diag")(phase);
    auto clock_operators =
      np.attr("stack")(py::make_tuple(X, Z), py::arg("axis") = 2).cast<py::array>();

    std::map<std::string, int64> state_labels;
    for (int64 n = 0; n < q; ++n) {
        state_labels[std::to_string(n)] = n;
    }
    state_labels["up"] = 0;
    if (q % 2 == 0) {
        state_labels["down"] = q / 2;
    }
    return { leg, clock_operators, state_labels };
}

namespace {

ElementarySpace::Ptr
build_golden_site_leg(std::string const& handedness)
{
    auto cat = std::make_shared<FibonacciAnyonCategory>(handedness);
    auto sym = std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ cat });
    return ElementarySpace::from_defining_sectors(
      sym, SectorArray::from_sector(FibonacciAnyonCategory::tau));
}

ElementarySpace::Ptr
build_su2k_spin1_leg(int64 k)
{
    if (k < 2) {
        throw std::invalid_argument("SU2kSpin1Site requires k >= 2");
    }
    auto cat = std::make_shared<SU2_kAnyonCategory>(static_cast<int>(k), "left");
    auto sym = std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ cat });
    if (!cat->spin_one) {
        throw std::runtime_error("SU2_kAnyonCategory spin_one sector is not defined.");
    }
    return ElementarySpace::from_defining_sectors(sym, SectorArray::from_sector(*cat->spin_one));
}

} // namespace

FibonacciAnyonSite::Prepared
FibonacciAnyonSite::prepare()
{
    auto symmetry = std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<FibonacciAnyonCategory>("left") });
    auto leg = ElementarySpace::from_defining_sectors(symmetry, symmetry->all_sectors());
    return { std::move(symmetry), std::move(leg) };
}

IsingAnyonSite::Prepared
IsingAnyonSite::prepare(int nu)
{
    auto symmetry = std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<IsingAnyonCategory>(nu) });
    auto leg = ElementarySpace::from_defining_sectors(symmetry, symmetry->all_sectors());
    return { std::move(symmetry), std::move(leg) };
}

SpinSite::SpinSite(float64 S,
                   std::optional<std::string> conserve,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device,
                   bool use_test_su2)
  : SpinSite(prepare(S, conserve, use_test_su2),
             S,
             conserve,
             std::move(backend),
             std::move(default_device),
             use_test_su2)
{
}

SpinSite::SpinSite(Prepared&& prepared,
                   float64 S,
                   std::optional<std::string> conserve,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device,
                   bool use_test_su2)
  : Site(prepared.leg, std::move(prepared.state_labels), {}, backend, default_device)
{
    auto spin = std::make_shared<SpinDOF>(prepared.leg, std::move(prepared.spin_vector));
    set_dof(DOF_SPIN, spin);
    this->S = S;
    this->double_total_spin = prepared.two_S;
    this->conserve = std::move(conserve);
    this->use_test_su2 = use_test_su2;
    auto const factor = first_factor(prepared.sym);
    auto np = numpy_module();
    if (!factor_is_su2(factor)) {
        add_onsite_operator(
          "Sz",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 2)),
          true);
        if (prepared.two_S == 1) {
            add_onsite_operator(
              "Sigmaz",
              np.attr("multiply")(2.0,
                                  spin->spin_vector.attr("__getitem__")(
                                    py::make_tuple(py::slice(), py::slice(), 2))),
              true);
        }
    }
    if (factor_is_no_symmetry(factor)) {
        add_onsite_operator(
          "Sx",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)));
        add_onsite_operator(
          "Sy",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)));
        add_onsite_operator(
          "Sp",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)) +
            py::cast(std::complex(0., 1.)) *
              spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)));
        add_onsite_operator(
          "Sm",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)) -
            py::cast(std::complex(0., 1.)) *
              spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)));
        if (prepared.two_S == 1) {
            add_onsite_operator(
              "Sigmax",
              np.attr("multiply")(2.0,
                                  spin->spin_vector.attr("__getitem__")(
                                    py::make_tuple(py::slice(), py::slice(), 0))));
            add_onsite_operator(
              "Sigmay",
              np.attr("multiply")(2.0,
                                  spin->spin_vector.attr("__getitem__")(
                                    py::make_tuple(py::slice(), py::slice(), 1))));
        }
    }
}

void
SpinSite::test_sanity()
{
    Site::test_sanity();
    auto spin = require_spin();
    auto np = numpy_module();
    auto S_sq = np.attr("tensordot")(spin->spin_vector,
                                     spin->spin_vector,
                                     py::make_tuple(py::make_tuple(-1, 1), py::make_tuple(-1, 0)));
    int64 const two_S = leg_dim(leg) - 1;
    auto eigenvalue = np.attr("multiply")(two_S * (two_S + 2) / 4., np.attr("eye")(two_S + 1));
    if (!np.attr("allclose")(S_sq, eigenvalue).cast<bool>()) {
        throw std::runtime_error("SpinSite sanity check failed: S^2 eigenvalue mismatch.");
    }
}

std::string
SpinSite::repr() const
{
    return std::format("SpinSite(S={}, conserve={})", S, conserve.value_or("None"));
}

SpinlessBosonSite::SpinlessBosonSite(py::object Nmax,
                                     py::object conserve,
                                     std::optional<float64> filling,
                                     TensorBackend::Ptr backend,
                                     std::optional<std::string> default_device)
  : SpinlessBosonSite(prepare(Nmax, conserve),
                      conserve,
                      filling,
                      std::move(backend),
                      std::move(default_device))
{
}

SpinlessBosonSite::SpinlessBosonSite(Prepared&& prepared,
                                     py::object conserve,
                                     std::optional<float64> filling,
                                     TensorBackend::Ptr backend,
                                     std::optional<std::string> default_device)
  : Site(prepared.leg, std::move(prepared.state_labels), {}, backend, default_device)
{
    auto occ = std::make_shared<BosonicDOF>(prepared.leg,
                                            std::move(prepared.Nmax_arr),
                                            std::move(prepared.creators),
                                            std::move(prepared.annihilators));
    set_dof(DOF_OCCUPATION, occ);
    this->conserve = std::move(conserve);
    this->filling = filling;
    occ->add_individual_occupation_ops(*this);
    occ->add_total_occupation_ops(*this);
    if (filling) {
        add_filling_ops(*this, occ->n_tot, *filling, prepared.total_dim);
    }
}

std::string
SpinlessBosonSite::repr() const
{
    auto bos = std::dynamic_pointer_cast<BosonicDOF>(require_occupation());
    return std::format("SpinlessBosonSite(Nmax={}, conserve={}, filling={})",
                       py::str(bos->Nmax).cast<std::string>(),
                       py::str(conserve).cast<std::string>(),
                       filling ? std::to_string(*filling) : "None");
}

SpinlessFermionSite::SpinlessFermionSite(int64 num_species,
                                         py::object conserve,
                                         std::optional<float64> filling,
                                         TensorBackend::Ptr backend,
                                         std::optional<std::string> default_device)
  : SpinlessFermionSite(prepare(num_species, conserve),
                        num_species,
                        conserve,
                        filling,
                        std::move(backend),
                        std::move(default_device))
{
}

SpinlessFermionSite::SpinlessFermionSite(Prepared&& prepared,
                                         int64 num_species,
                                         py::object conserve,
                                         std::optional<float64> filling,
                                         TensorBackend::Ptr backend,
                                         std::optional<std::string> default_device)
  : Site(prepared.leg, std::move(prepared.state_labels), {}, backend, default_device)
{
    auto occ = std::make_shared<FermionicDOF>(
      prepared.leg, std::move(prepared.creators), std::move(prepared.annihilators));
    set_dof(DOF_OCCUPATION, occ);
    this->num_species = num_species;
    this->conserve = std::move(conserve);
    this->filling = filling;
    occ->add_individual_occupation_ops(*this);
    occ->add_total_occupation_ops(*this);
    if (filling) {
        add_filling_ops(*this, occ->n_tot, *filling, leg_dim(leg), true);
    }
}

std::string
SpinlessFermionSite::repr() const
{
    return std::format("SpinlessFermionSite(num_species={}, conserve={}, filling={})",
                       num_species,
                       py::str(conserve).cast<std::string>(),
                       filling ? std::to_string(*filling) : "None");
}

SpinHalfFermionSite::SpinHalfFermionSite(std::string conserve_N,
                                         std::optional<std::string> conserve_S,
                                         std::optional<float64> filling,
                                         TensorBackend::Ptr backend,
                                         std::optional<std::string> default_device,
                                         bool use_test_su2)
  : SpinHalfFermionSite(prepare(conserve_N, conserve_S, use_test_su2),
                        conserve_N,
                        conserve_S,
                        filling,
                        std::move(backend),
                        std::move(default_device),
                        use_test_su2)
{
}

SpinHalfFermionSite::SpinHalfFermionSite(Prepared&& prepared,
                                         std::string conserve_N,
                                         std::optional<std::string> conserve_S,
                                         std::optional<float64> filling,
                                         TensorBackend::Ptr backend,
                                         std::optional<std::string> default_device,
                                         bool use_test_su2)
  : Site(prepared.leg, std::move(prepared.state_labels), {}, backend, default_device)
{
    auto spin = std::make_shared<SpinDOF>(prepared.leg, std::move(prepared.spin_vector));
    auto occ = std::make_shared<FermionicDOF>(
      prepared.leg,
      std::move(prepared.creators),
      std::move(prepared.annihilators),
      std::vector<std::optional<std::string>>{ std::optional<std::string>{ "up" },
                                               std::optional<std::string>{ "down" } });
    set_dof(DOF_SPIN, spin);
    set_dof(DOF_OCCUPATION, occ);

    this->conserve_N = std::move(conserve_N);
    this->conserve_S = std::move(conserve_S);
    this->filling = filling;
    this->use_test_su2 = use_test_su2;
    auto const& sym_S_factor = prepared.sym_S_factor;
    auto np = numpy_module();

    if (!factor_is_su2(sym_S_factor)) {
        occ->add_individual_occupation_ops(*this);
        if (onsite_operators.contains("N0")) {
            rename_op("N0", "Nup");
        }
        if (onsite_operators.contains("N1")) {
            rename_op("N1", "Ndown");
        }
    }
    occ->add_total_occupation_ops(*this);

    auto NtotNtot = np.attr("matmul")(occ->n_tot, occ->n_tot);
    auto NupNdown = np.attr("multiply")(0.5, NtotNtot - occ->n_tot);
    add_onsite_operator("NupNdown", NupNdown, true);

    std::map<std::string, py::object> ops;
    if (!factor_is_su2(sym_S_factor)) {
        ops["Sz"] =
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 2));
        ops["Sigmaz"] = np.attr("multiply")(
          2.0, spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 2)));
    }
    if (factor_is_no_symmetry(sym_S_factor)) {
        add_onsite_operator(
          "Sx",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)),
          std::nullopt,
          true);
        add_onsite_operator(
          "Sy",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)),
          std::nullopt,
          true);
        add_onsite_operator(
          "Sp",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)) +
            py::cast(std::complex(0., 1.)) *
              spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)),
          std::nullopt,
          true);
        add_onsite_operator(
          "Sm",
          spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)) -
            py::cast(std::complex(0., 1.)) *
              spin->spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1)),
          std::nullopt,
          true);
        add_onsite_operator("Sigmax",
                            np.attr("multiply")(2.0,
                                                spin->spin_vector.attr("__getitem__")(
                                                  py::make_tuple(py::slice(), py::slice(), 0))),
                            std::nullopt,
                            true);
        add_onsite_operator("Sigmay",
                            np.attr("multiply")(2.0,
                                                spin->spin_vector.attr("__getitem__")(
                                                  py::make_tuple(py::slice(), py::slice(), 1))),
                            std::nullopt,
                            true);
    }
    if (filling) {
        auto dN_diag =
          np.attr("diag")(occ->n_tot) - np.attr("multiply")(*filling, np.attr("ones")(4));
        ops["dN"] = np.attr("diag")(dN_diag);
        ops["dNdN"] = np.attr("diag")(np.attr("square")(dN_diag));
    }
    for (auto const& [name, op] : ops) {
        add_onsite_operator(name, op, true, true);
    }
}

void
SpinHalfFermionSite::test_sanity()
{
    Site::test_sanity();
}

std::string
SpinHalfFermionSite::repr() const
{
    return std::format("SpinHalfFermionSite(conserve_N={}, conserve_S={}, filling={})",
                       conserve_N,
                       conserve_S.value_or("None"),
                       filling ? std::to_string(*filling) : "None");
}

ClockSite::ClockSite(int64 q,
                     std::optional<std::string> conserve,
                     TensorBackend::Ptr backend,
                     std::optional<std::string> default_device)
  : ClockSite(prepare(q, conserve), q, conserve, std::move(backend), std::move(default_device))
{
}

ClockSite::ClockSite(Prepared&& prepared,
                     int64 q,
                     std::optional<std::string> conserve,
                     TensorBackend::Ptr backend,
                     std::optional<std::string> default_device)
  : Site(prepared.leg, std::move(prepared.state_labels), {}, backend, default_device)
{
    auto clock = std::make_shared<ClockDOF>(prepared.leg, std::move(prepared.clock_operators));
    set_dof(DOF_CLOCK, clock);
    clock->register_default_ops(*this);
    this->q = q;
    this->conserve = std::move(conserve);
    auto np = numpy_module();
    auto sym = leg_symmetry(leg);
    auto Xhc = np.attr("conj")(
      clock->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0))
        .attr("T"));
    if (factor_is_no_symmetry(first_factor(sym))) {
        add_onsite_operator(
          "X",
          clock->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)));
        add_onsite_operator("Xhc", Xhc);
        add_onsite_operator(
          "Xphc",
          clock->clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0)) +
            Xhc);
    }
}

std::string
ClockSite::repr() const
{
    return std::format("ClockSite(q={}, conserve={})", q, conserve.value_or("None"));
}

AnyonSite::AnyonSite(Symmetry::Ptr symmetry,
                     TensorBackend::Ptr backend,
                     std::optional<std::string> default_device)
  : AnyonSite(ElementarySpace::from_defining_sectors(symmetry, symmetry->all_sectors()),
              std::move(backend),
              std::move(default_device))
{
}

AnyonSite::AnyonSite(ElementarySpace::Ptr leg,
                     TensorBackend::Ptr backend,
                     std::optional<std::string> default_device)
  : Site(leg, {}, {}, backend, default_device)
{
    auto anyon = std::make_shared<AnyonDOF>(leg);
    set_dof(DOF_ANYON, anyon);
    anyon->register_default_ops(*this, backend, default_device);
}

std::string
AnyonSite::repr() const
{
    auto anyon = require_anyon();
    return std::format("AnyonSite(symmetry={}, sector_names={})",
                       leg_symmetry(leg)->repr(),
                       format_string_list(anyon->sector_names));
}

FibonacciAnyonSite::FibonacciAnyonSite(TensorBackend::Ptr backend,
                                       std::optional<std::string> default_device)
  : FibonacciAnyonSite(prepare(), std::move(backend), std::move(default_device))
{
}

FibonacciAnyonSite::FibonacciAnyonSite(Prepared&& prepared,
                                       TensorBackend::Ptr backend,
                                       std::optional<std::string> default_device)
  : AnyonSite(prepared.leg, backend, default_device)
{
    add_anyon_projectors(*this, { "vac", "tau" }, backend, default_device);
}

std::string
FibonacciAnyonSite::repr() const
{
    auto const* cat =
      dynamic_cast<FibonacciAnyonCategory const*>(leg_symmetry(leg)->factors[0].get());
    return std::format("FibonacciAnyonSite(handedness={})", cat ? cat->handedness : "left");
}

IsingAnyonSite::IsingAnyonSite(int nu,
                               TensorBackend::Ptr backend,
                               std::optional<std::string> default_device)
  : IsingAnyonSite(prepare(nu), std::move(backend), std::move(default_device))
{
}

IsingAnyonSite::IsingAnyonSite(Prepared&& prepared,
                               TensorBackend::Ptr backend,
                               std::optional<std::string> default_device)
  : AnyonSite(prepared.leg, backend, default_device)
{
    add_anyon_projectors(*this, { "vac", "sigma", "psi" }, backend, default_device);
}

std::string
IsingAnyonSite::repr() const
{
    auto const* cat = dynamic_cast<IsingAnyonCategory const*>(leg_symmetry(leg)->factors[0].get());
    return std::format("IsingAnyonSite(nu={})", cat ? cat->nu : 1);
}

GoldenSite::GoldenSite(std::string handedness,
                       TensorBackend::Ptr backend,
                       std::optional<std::string> default_device)
  : GoldenSite(build_golden_site_leg(handedness), std::move(backend), std::move(default_device))
{
}

GoldenSite::GoldenSite(ElementarySpace::Ptr leg,
                       TensorBackend::Ptr backend,
                       std::optional<std::string> default_device)
  : Site(leg, {}, {}, backend, default_device)
{
    auto anyon = std::make_shared<AnyonDOF>(leg);
    set_dof(DOF_ANYON, anyon);
    anyon->register_default_ops(*this, backend, default_device);
}

std::string
GoldenSite::repr() const
{
    auto const* cat =
      dynamic_cast<FibonacciAnyonCategory const*>(leg_symmetry(leg)->factors[0].get());
    return std::format("GoldenSite(handedness={})", cat ? cat->handedness : "left");
}

SU2kSpin1Site::SU2kSpin1Site(int64 k,
                             TensorBackend::Ptr backend,
                             std::optional<std::string> default_device)
  : SU2kSpin1Site(build_su2k_spin1_leg(k), k, std::move(backend), std::move(default_device))
{
}

SU2kSpin1Site::SU2kSpin1Site(ElementarySpace::Ptr leg,
                             int64 k,
                             TensorBackend::Ptr backend,
                             std::optional<std::string> default_device)
  : Site(leg, {}, {}, backend, default_device)
{
    auto anyon = std::make_shared<AnyonDOF>(leg);
    set_dof(DOF_ANYON, anyon);
    anyon->register_default_ops(*this, backend, default_device);
    this->k = k;
}

std::string
SU2kSpin1Site::repr() const
{
    auto const* cat = dynamic_cast<SU2_kAnyonCategory const*>(leg_symmetry(leg)->factors[0].get());
    if (cat) {
        return std::format("SU2kSpin1Site(k={}, handedness={})", cat->k, cat->handedness);
    }
    return std::format("SU2kSpin1Site(k={}, handedness=left)", k);
}

namespace {

py::object
optional_str_to_py(std::optional<std::string> const& value)
{
    if (!value)
        return py::none();
    return py::str(*value);
}

py::object
optional_float_to_py(std::optional<float64> const& value)
{
    if (!value)
        return py::none();
    return py::float_(*value);
}

} // namespace

py::dict
SpinSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["S"] = S;
    d["conserve"] = optional_str_to_py(conserve);
    d["use_test_su2"] = use_test_su2;
    return d;
}

py::dict
SpinlessBosonSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    auto bos = std::dynamic_pointer_cast<BosonicDOF>(require_occupation());
    d["Nmax"] = bos->Nmax;
    d["conserve"] = conserve;
    d["filling"] = optional_float_to_py(filling);
    return d;
}

py::dict
SpinlessFermionSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["num_species"] = num_species;
    d["conserve"] = conserve;
    d["filling"] = optional_float_to_py(filling);
    return d;
}

py::dict
SpinHalfFermionSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["conserve_N"] = py::str(conserve_N);
    d["conserve_S"] = optional_str_to_py(conserve_S);
    d["filling"] = optional_float_to_py(filling);
    d["use_test_su2"] = use_test_su2;
    return d;
}

py::dict
ClockSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["q"] = q;
    d["conserve"] = optional_str_to_py(conserve);
    return d;
}

py::dict
AnyonSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["symmetry"] = py::cast(symmetry());
    return d;
}

py::dict
FibonacciAnyonSite::hdf5_init_kwargs() const
{
    return hdf5_backend_kwargs();
}

py::dict
IsingAnyonSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    auto const* cat = dynamic_cast<IsingAnyonCategory const*>(symmetry()->factors[0].get());
    d["nu"] = cat ? cat->nu : 1;
    return d;
}

py::dict
GoldenSite::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    auto const* cat = dynamic_cast<FibonacciAnyonCategory const*>(symmetry()->factors[0].get());
    d["handedness"] = cat ? cat->handedness : "left";
    return d;
}

py::dict
SU2kSpin1Site::hdf5_init_kwargs() const
{
    py::dict d = hdf5_backend_kwargs();
    d["k"] = k;
    return d;
}

namespace {

ElementarySpace::Ptr
as_es(Space::Ptr space)
{
    auto leg = std::dynamic_pointer_cast<ElementarySpace>(space);
    if (!leg) {
        throw std::invalid_argument("GroupedSite: expected ElementarySpace");
    }
    return leg;
}

std::vector<std::variant<int64, std::string>> const kPpLabels{ "p", "p*" };

OptionalLabels
pp_optional_labels()
{
    return OptionalLabels{ "p", "p*" };
}

Site::Ptr
clone_site_same_leg(Site const& site)
{
    auto cloned = std::make_shared<Site>(site.leg,
                                         site.state_labels,
                                         std::map<std::string, SymmetricTensorPtr>{},
                                         site.backend,
                                         site.default_device);
    for (auto const& [name, op] : site.onsite_operators) {
        if (name == "Id") {
            continue;
        }
        bool const is_diag = static_cast<bool>(std::dynamic_pointer_cast<DiagonalTensor>(op));
        py::array dense = op->to_numpy(kPpLabels, py::none(), true);
        cloned->add_onsite_operator(name, dense, is_diag, true, std::nullopt);
    }
    cloned->hc_ops = site.hc_ops;
    for (auto const& [name, dof] : site.dofs) {
        if (auto spin = std::dynamic_pointer_cast<SpinDOF>(dof)) {
            cloned->set_dof(name, std::make_shared<SpinDOF>(cloned->leg, spin->spin_vector));
        } else if (auto clock = std::dynamic_pointer_cast<ClockDOF>(dof)) {
            cloned->set_dof(name, std::make_shared<ClockDOF>(cloned->leg, clock->clock_operators));
        } else if (auto anyon = std::dynamic_pointer_cast<AnyonDOF>(dof)) {
            cloned->set_dof(name, std::make_shared<AnyonDOF>(cloned->leg, anyon->sector_names));
        } else if (auto ferm = std::dynamic_pointer_cast<FermionicDOF>(dof)) {
            auto copy = std::make_shared<FermionicDOF>(
              cloned->leg, ferm->creators, ferm->annihilators, ferm->species_names);
            copy->partial_JWs = ferm->partial_JWs;
            copy->JW = ferm->JW;
            cloned->set_dof(name, std::move(copy));
        } else if (auto bos = std::dynamic_pointer_cast<BosonicDOF>(dof)) {
            auto copy = std::make_shared<BosonicDOF>(
              cloned->leg, bos->Nmax, bos->creators, bos->annihilators, bos->species_names);
            copy->JW = bos->JW;
            cloned->set_dof(name, std::move(copy));
        }
    }
    return cloned;
}

Site::Ptr
clone_site_with_leg(Site const& site, ElementarySpace::Ptr new_leg)
{
    auto cloned = clone_site_same_leg(site);
    cloned->change_leg(std::move(new_leg));
    return cloned;
}

py::array
numpy_eye(int64 dim)
{
    return numpy_module().attr("eye")(dim);
}

py::array
kron_embed_matrix(py::array mat, int64 site_idx, std::vector<int64> const& dims)
{
    auto np = numpy_module();
    int64 left = 1;
    for (int64 i = 0; i < site_idx; ++i) {
        left *= dims[static_cast<std::size_t>(i)];
    }
    int64 right = 1;
    for (std::size_t i = static_cast<std::size_t>(site_idx) + 1; i < dims.size(); ++i) {
        right *= dims[i];
    }
    py::object out = mat;
    if (left > 1) {
        out = np.attr("kron")(numpy_eye(left), out);
    }
    if (right > 1) {
        out = np.attr("kron")(out, numpy_eye(right));
    }
    return as_immutable_array(out);
}

py::array
kron_embed_stack3(py::array stack, int64 site_idx, std::vector<int64> const& dims)
{
    auto np = numpy_module();
    py::ssize_t n_ops = stack.attr("shape").attr("__getitem__")(2).cast<py::ssize_t>();
    py::list mats;
    for (py::ssize_t a = 0; a < n_ops; ++a) {
        py::array mat = stack.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), a));
        mats.append(kron_embed_matrix(mat, site_idx, dims));
    }
    return as_immutable_array(np.attr("stack")(mats, py::arg("axis") = 2));
}

std::map<std::string, int64>
combined_state_labels(std::vector<Site::Ptr> const& working,
                      std::vector<std::string> const& labels,
                      AbelianLegPipe const& pipe)
{
    std::map<std::string, int64> out;
    if (working.empty()) {
        return out;
    }
    for (auto const& s : working) {
        if (s->state_labels.empty()) {
            return out; // product over empty → no labels
        }
    }
    std::vector<int64> dims;
    dims.reserve(working.size());
    for (auto const& s : working) {
        dims.push_back(static_cast<int64>(s->dim()));
    }
    std::vector<int64> strides(dims.size(), 1);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] =
          strides[static_cast<std::size_t>(i + 1)] * dims[static_cast<std::size_t>(i + 1)];
    }
    auto perm = pipe.basis_perm();

    // Nested iteration over state_labels of each site via recursive lambda.
    std::vector<std::pair<std::string, int64>> chosen;
    chosen.reserve(working.size());
    std::function<void(std::size_t)> rec = [&](std::size_t site_i) {
        if (site_i == working.size()) {
            int64 pipe_idx = 0;
            std::ostringstream name;
            for (std::size_t i = 0; i < chosen.size(); ++i) {
                pipe_idx += chosen[i].second * strides[i];
                if (i > 0) {
                    name << ' ';
                }
                name << chosen[i].first << '_' << labels[i];
            }
            int64 mapped = pipe_idx;
            if (!perm.empty()) {
                mapped = perm[static_cast<std::size_t>(pipe_idx)];
            }
            out[name.str()] = mapped;
            return;
        }
        for (auto const& [lbl, idx] : working[site_i]->state_labels) {
            chosen.emplace_back(lbl, idx);
            rec(site_i + 1);
            chosen.pop_back();
        }
    };
    rec(0);
    return out;
}

} // namespace

GroupedSite::Prepared
GroupedSite::prepare(std::vector<Site::Ptr> sites,
                     std::optional<std::vector<std::string>> labels_opt,
                     std::string symmetry_combine)
{
    if (sites.empty()) {
        throw std::invalid_argument("GroupedSite: need at least one site");
    }
    for (auto const& s : sites) {
        if (!s) {
            throw std::invalid_argument("GroupedSite: sites must not contain null");
        }
    }
    auto const n = static_cast<int64>(sites.size());
    std::vector<std::string> labels;
    if (labels_opt) {
        labels = *labels_opt;
        if (static_cast<int64>(labels.size()) != n) {
            throw std::invalid_argument("GroupedSite: labels length must match number of sites");
        }
    } else {
        labels.reserve(static_cast<std::size_t>(n));
        for (int64 i = 0; i < n; ++i) {
            labels.push_back(std::to_string(i));
        }
    }

    auto backend = sites[0]->backend;
    auto device = sites[0]->default_device;
    for (auto const& s : sites) {
        if (s->backend.get() != backend.get()) {
            throw std::invalid_argument("GroupedSite: all sites must share the same backend");
        }
    }

    std::vector<Site::Ptr> working;
    working.reserve(sites.size());
    std::vector<ElementarySpace::Ptr> legs;

    if (symmetry_combine == "same") {
        auto const& sym0 = *sites[0]->symmetry();
        for (auto const& s : sites) {
            if (!s->symmetry()->is_equivalent_to(sym0, /*strict_ordering=*/true)) {
                throw std::invalid_argument(
                  "GroupedSite(symmetry_combine='same'): sites must share equivalent symmetries; "
                  "call set_common_symmetry first");
            }
            if (!s->symmetry()->is_abelian()) {
                throw std::invalid_argument("GroupedSite: only abelian symmetries are supported");
            }
            working.push_back(s);
            legs.push_back(s->leg);
        }
    } else if (symmetry_combine == "drop") {
        for (auto const& s : sites) {
            auto new_leg = as_es(s->leg->drop_symmetry());
            working.push_back(clone_site_with_leg(*s, new_leg));
            legs.push_back(working.back()->leg);
        }
    } else if (symmetry_combine == "independent") {
        std::vector<Symmetry::Ptr> all_symmetries;
        all_symmetries.reserve(sites.size());
        for (auto const& s : sites) {
            if (!s->symmetry()->is_abelian()) {
                throw std::invalid_argument("GroupedSite: only abelian symmetries are supported");
            }
            all_symmetries.push_back(s->symmetry());
        }
        for (std::size_t i = 0; i < sites.size(); ++i) {
            std::vector<ElementarySpace::Ptr> independent_gradings;
            independent_gradings.reserve(sites.size());
            auto const dim_i = static_cast<int64>(sites[i]->dim());
            for (std::size_t j = 0; j < sites.size(); ++j) {
                if (j == i) {
                    independent_gradings.push_back(sites[i]->leg);
                } else {
                    independent_gradings.push_back(
                      ElementarySpace::from_trivial_sector(dim_i, all_symmetries[j]));
                }
            }
            auto new_leg = ElementarySpace::from_independent_symmetries(independent_gradings);
            working.push_back(clone_site_with_leg(*sites[i], new_leg));
            legs.push_back(working.back()->leg);
        }
    } else {
        throw std::invalid_argument(
          std::format("Unknown option for symmetry_combine: {}", symmetry_combine));
    }

    auto pipe = std::make_shared<AbelianLegPipe>(legs, /*is_dual=*/false, /*combine_cstyle=*/true);
    auto state_labels = combined_state_labels(working, labels, *pipe);
    return Prepared{ std::move(pipe),   std::move(state_labels),     std::move(working),
                     std::move(labels), std::move(symmetry_combine), std::move(backend),
                     std::move(device) };
}

GroupedSite::GroupedSite(Prepared&& prepared)
  : Site(prepared.pipe,
         std::move(prepared.state_labels),
         {},
         prepared.backend,
         prepared.default_device)
  , n_sites(static_cast<int64>(prepared.working_sites.size()))
  , sites(std::move(prepared.working_sites))
  , labels(std::move(prepared.labels))
  , symmetry_combine(std::move(prepared.symmetry_combine))
{
    add_kronecker_onsite_ops();
    register_embedded_dofs();
}

GroupedSite::GroupedSite(std::vector<Site::Ptr> sites_,
                         std::optional<std::vector<std::string>> labels_,
                         std::string symmetry_combine_)
  : GroupedSite(prepare(std::move(sites_), std::move(labels_), std::move(symmetry_combine_)))
{
}

SymmetricTensorPtr
GroupedSite::kroneckerproduct(std::vector<SymmetricTensorPtr> ops) const
{
    if (static_cast<int64>(ops.size()) != n_sites) {
        throw std::invalid_argument("kroneckerproduct: need one operator per grouped site");
    }
    for (auto& op : ops) {
        if (std::dynamic_pointer_cast<DiagonalTensor>(op)) {
            op = op->as_SymmetricTensor();
        }
    }
    TensorPtr tens = ops[0]->copy(/*deep=*/false);
    tens->relabel({ { "p", "p0" }, { "p*", "p0*" } });
    for (int64 i = 1; i < n_sites; ++i) {
        std::map<std::string, std::string> relabel2{
            { "p", std::format("p{}", i) },
            { "p*", std::format("p{}*", i) },
        };
        tens = outer(tens, ops[static_cast<std::size_t>(i)], std::nullopt, relabel2);
    }
    std::vector<LegRef> cod;
    std::vector<LegRef> dom;
    cod.reserve(static_cast<std::size_t>(n_sites));
    dom.reserve(static_cast<std::size_t>(n_sites));
    for (int64 i = 0; i < n_sites; ++i) {
        cod.emplace_back(std::format("p{}", i));
    }
    // Domain group is right-to-left in `tensor.legs` after `outer` (… p1*, p0*), so reverse
    // site order. See `combine_legs` docs: first group entry is the right-most product factor.
    for (int64 i = n_sites - 1; i >= 0; --i) {
        dom.emplace_back(std::format("p{}*", i));
    }
    auto pipe = std::dynamic_pointer_cast<AbelianLegPipe>(leg);
    if (!pipe) {
        throw std::logic_error("GroupedSite.leg must be an AbelianLegPipe");
    }
    // Pipe dualities {false, true} match Site matrix convention: codomain/domain both store
    // the non-dual `pipe` (public legs appear as pipe / pipe.dual).
    auto combined = combine_legs(tens,
                                 { std::move(cod), std::move(dom) },
                                 PipeDualities{ std::vector<bool>{ false, true } },
                                 std::vector<Leg::Ptr>{ pipe, pipe });
    combined->set_labels(pp_optional_labels());
    return std::dynamic_pointer_cast<SymmetricTensor>(combined);
}

void
GroupedSite::add_kronecker_onsite_ops()
{
    std::vector<SymmetricTensorPtr> ids;
    ids.reserve(static_cast<std::size_t>(n_sites));
    for (auto const& s : sites) {
        ids.push_back(s->onsite_operators.at("Id"));
    }
    for (int64 i = 0; i < n_sites; ++i) {
        auto const& site = *sites[static_cast<std::size_t>(i)];
        for (auto const& [name, op] : site.onsite_operators) {
            if (name == "Id") {
                continue;
            }
            auto ops = ids;
            ops[static_cast<std::size_t>(i)] = op;
            auto kron = kroneckerproduct(ops);
            std::string new_name = name + labels[static_cast<std::size_t>(i)];
            std::optional<std::string> hc;
            auto hc_it = site.hc_ops.find(name);
            if (hc_it != site.hc_ops.end()) {
                hc = hc_it->second + labels[static_cast<std::size_t>(i)];
            }
            add_onsite_operator(new_name, py::cast(kron), std::nullopt, true, hc);
        }
    }
}

void
GroupedSite::register_embedded_dofs()
{
    std::vector<int64> dims;
    dims.reserve(static_cast<std::size_t>(n_sites));
    for (auto const& s : sites) {
        dims.push_back(static_cast<int64>(s->dim()));
    }
    for (int64 i = 0; i < n_sites; ++i) {
        auto const& site = *sites[static_cast<std::size_t>(i)];
        auto const& lbl = labels[static_cast<std::size_t>(i)];
        for (auto const& [dof_name, dof] : site.dofs) {
            std::string reg = std::format("{}.{}", lbl, dof_name);
            if (auto spin = std::dynamic_pointer_cast<SpinDOF>(dof)) {
                auto embedded = kron_embed_stack3(spin->spin_vector, i, dims);
                set_dof(reg, std::make_shared<SpinDOF>(leg, embedded));
            } else if (auto clock = std::dynamic_pointer_cast<ClockDOF>(dof)) {
                auto embedded = kron_embed_stack3(clock->clock_operators, i, dims);
                set_dof(reg, std::make_shared<ClockDOF>(leg, embedded));
            } else if (auto anyon = std::dynamic_pointer_cast<AnyonDOF>(dof)) {
                set_dof(reg, std::make_shared<AnyonDOF>(leg, anyon->sector_names));
            } else if (auto ferm = std::dynamic_pointer_cast<FermionicDOF>(dof)) {
                auto creators = kron_embed_stack3(ferm->creators, i, dims);
                auto annihilators = kron_embed_stack3(ferm->annihilators, i, dims);
                auto copy =
                  std::make_shared<FermionicDOF>(leg, creators, annihilators, ferm->species_names);
                copy->JW = kron_embed_matrix(ferm->JW, i, dims);
                // partial_JWs: embed each species slice
                copy->partial_JWs = kron_embed_stack3(ferm->partial_JWs, i, dims);
                set_dof(reg, std::move(copy));
            } else if (auto bos = std::dynamic_pointer_cast<BosonicDOF>(dof)) {
                auto creators = kron_embed_stack3(bos->creators, i, dims);
                auto annihilators = kron_embed_stack3(bos->annihilators, i, dims);
                auto copy = std::make_shared<BosonicDOF>(
                  leg, bos->Nmax, creators, annihilators, bos->species_names);
                copy->JW = kron_embed_matrix(bos->JW, i, dims);
                set_dof(reg, std::move(copy));
            }
        }
    }
}

std::string
GroupedSite::repr() const
{
    return std::format("GroupedSite(n_sites={}, symmetry_combine='{}', labels={})",
                       n_sites,
                       symmetry_combine,
                       labels.size());
}

std::vector<GroupedSite::Ptr>
group_sites(std::vector<Site::Ptr> const& sites,
            int64 n,
            std::optional<std::vector<std::string>> labels,
            std::string symmetry_combine)
{
    if (n < 1) {
        throw std::invalid_argument("group_sites: n must be >= 1");
    }
    std::vector<GroupedSite::Ptr> out;
    for (std::size_t start = 0; start < sites.size(); start += static_cast<std::size_t>(n)) {
        std::size_t stop = std::min(start + static_cast<std::size_t>(n), sites.size());
        std::vector<Site::Ptr> chunk(sites.begin() + static_cast<std::ptrdiff_t>(start),
                                     sites.begin() + static_cast<std::ptrdiff_t>(stop));
        std::optional<std::vector<std::string>> chunk_labels;
        if (labels) {
            chunk_labels = std::vector<std::string>(
              labels->begin(), labels->begin() + static_cast<std::ptrdiff_t>(chunk.size()));
        }
        out.push_back(
          std::make_shared<GroupedSite>(std::move(chunk), chunk_labels, symmetry_combine));
    }
    return out;
}

} // namespace lasyten
