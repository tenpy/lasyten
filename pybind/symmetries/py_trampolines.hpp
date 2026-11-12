#pragma once

#include <lasyten/symmetries/abelian_group.h>
#include <lasyten/symmetries/base_symmetry.h>
#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/symmetries/group.h>
#include <lasyten/symmetries/sector_numpy.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry_factor.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "symmetries/casters.hpp"

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// pybind11 trampoline so Python subclasses (SymmetryFactor, Symmetry, …) can override.
class PyBaseSymmetry
  : public BaseSymmetry
  , public py::trampoline_self_life_support
{
  public:
    using BaseSymmetry::BaseSymmetry;

    // Note: do NOT trampoline can_be_dropped / has_*_braid / is_abelian / has_unique_fusion.
    // Those are bound as def_property_readonly; PYBIND11_OVERRIDE then finds the property
    // and raises TypeError: bool is not an instance of function.

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, BaseSymmetry, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, BaseSymmetry, _n_symbol, a, b, c);
    }
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, BaseSymmetry, _f_symbol, a, b, c, d, e, f);
    }
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, BaseSymmetry, _r_symbol, a, b, c);
    }
    std::shared_ptr<Symmetry> as_Symmetry() override
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Symmetry>, BaseSymmetry, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, BaseSymmetry, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, BaseSymmetry, fusion_outcomes, a, b);
    }

    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, BaseSymmetry, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    FusionSymbol swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, BaseSymmetry, swap_gate, a, b);
    }
    FusionSymbol Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, BaseSymmetry, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        // Python overrides take ``*sectors``; unpack instead of passing one list.
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<BaseSymmetry const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return BaseSymmetry::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, BaseSymmetry, sector_dim, a);
    }
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_sector_dim");
        if (ov) {
            return vector_i64_from_numpy(ov(a));
        }
        return BaseSymmetry::batch_sector_dim(a);
    }
    std::vector<float64> batch_qdim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_qdim");
        if (ov) {
            return vector_f64_from_numpy(ov(a));
        }
        return BaseSymmetry::batch_qdim(a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, BaseSymmetry, sector_str, a);
    }
    Sector get_sector(std::string const& s) const override
    {
        PYBIND11_OVERRIDE(Sector, BaseSymmetry, get_sector, s);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, BaseSymmetry, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, BaseSymmetry, qdim, a); }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, BaseSymmetry, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, BaseSymmetry, inv_sqrt_qdim, a);
    }
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, BaseSymmetry, _b_symbol, a, b, c);
    }
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, BaseSymmetry, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, BaseSymmetry, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of SymmetryFactor (Group, anyons, …).
class PySymmetryFactor
  : public SymmetryFactor
  , public py::trampoline_self_life_support
{
  public:
    using SymmetryFactor::SymmetryFactor;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, SymmetryFactor, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, SymmetryFactor, _n_symbol, a, b, c);
    }
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, SymmetryFactor, _f_symbol, a, b, c, d, e, f);
    }
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, SymmetryFactor, _r_symbol, a, b, c);
    }
    std::shared_ptr<Symmetry> as_Symmetry() override
    {
        PYBIND11_OVERRIDE(std::shared_ptr<Symmetry>, SymmetryFactor, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, SymmetryFactor, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, SymmetryFactor, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, SymmetryFactor, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, SymmetryFactor, _is_equivalent_factor, other);
    }

    // Optional overrides commonly customized by factors (mirror PyBaseSymmetry).
    // Do not trampoline property-bound methods (is_abelian, can_be_dropped, …).
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, SymmetryFactor, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    FusionSymbol swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, SymmetryFactor, swap_gate, a, b);
    }
    FusionSymbol Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, SymmetryFactor, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, SymmetryFactor, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        // Python overrides take ``*sectors``; unpack instead of passing one list.
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<SymmetryFactor const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return SymmetryFactor::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, SymmetryFactor, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, SymmetryFactor, sector_dim, a);
    }
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_sector_dim");
        if (ov) {
            return vector_i64_from_numpy(ov(a));
        }
        return SymmetryFactor::batch_sector_dim(a);
    }
    std::vector<float64> batch_qdim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_qdim");
        if (ov) {
            return vector_f64_from_numpy(ov(a));
        }
        return SymmetryFactor::batch_qdim(a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, SymmetryFactor, sector_str, a);
    }
    Sector get_sector(std::string const& s) const override
    {
        PYBIND11_OVERRIDE(Sector, SymmetryFactor, get_sector, s);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, SymmetryFactor, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, SymmetryFactor, qdim, a); }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, SymmetryFactor, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, SymmetryFactor, inv_sqrt_qdim, a);
    }
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, SymmetryFactor, _b_symbol, a, b, c);
    }
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, SymmetryFactor, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, SymmetryFactor, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of Group (_SU2, SUN, …).
class PyGroup
  : public Group
  , public py::trampoline_self_life_support
{
  public:
    using Group::Group;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, Group, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, Group, _n_symbol, a, b, c);
    }
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, Group, _f_symbol, a, b, c, d, e, f);
    }
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, Group, _r_symbol, a, b, c);
    }
    std::shared_ptr<Symmetry> as_Symmetry() override
    {
        PYBIND11_OVERRIDE(std::shared_ptr<Symmetry>, Group, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, Group, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, Group, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, Group, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, Group, _is_equivalent_factor, other);
    }

    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE_PURE(FusionSymbol, Group, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    FusionSymbol swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, Group, swap_gate, a, b);
    }
    FusionSymbol Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, Group, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, Group, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<Group const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return Group::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, Group, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override { PYBIND11_OVERRIDE(int64, Group, sector_dim, a); }
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_sector_dim");
        if (ov) {
            return vector_i64_from_numpy(ov(a));
        }
        return Group::batch_sector_dim(a);
    }
    std::vector<float64> batch_qdim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_qdim");
        if (ov) {
            return vector_f64_from_numpy(ov(a));
        }
        return Group::batch_qdim(a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, Group, sector_str, a);
    }
    Sector get_sector(std::string const& s) const override
    {
        PYBIND11_OVERRIDE(Sector, Group, get_sector, s);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, Group, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, Group, qdim, a); }
    float64 sqrt_qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, Group, sqrt_qdim, a); }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, Group, inv_sqrt_qdim, a);
    }
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, Group, _b_symbol, a, b, c);
    }
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, Group, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, Group, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of AbelianGroup (if any remain after concretes).
class PyAbelianGroup
  : public AbelianGroup
  , public py::trampoline_self_life_support
{
  public:
    using AbelianGroup::AbelianGroup;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, AbelianGroup, dual_sector, a);
    }
    std::shared_ptr<Symmetry> as_Symmetry() override
    {
        PYBIND11_OVERRIDE(std::shared_ptr<Symmetry>, AbelianGroup, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, AbelianGroup, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, AbelianGroup, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, AbelianGroup, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, AbelianGroup, _is_equivalent_factor, other);
    }

    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, _n_symbol, a, b, c);
    }
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, _f_symbol, a, b, c, d, e, f);
    }
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, _r_symbol, a, b, c);
    }
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    FusionSymbol swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, swap_gate, a, b);
    }
    FusionSymbol Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, AbelianGroup, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<AbelianGroup const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return AbelianGroup::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, AbelianGroup, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, sector_dim, a);
    }
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_sector_dim");
        if (ov) {
            return vector_i64_from_numpy(ov(a));
        }
        return AbelianGroup::batch_sector_dim(a);
    }
    std::vector<float64> batch_qdim(SectorArray const& a) const override
    {
        py::object ov = py::get_override(this, "batch_qdim");
        if (ov) {
            return vector_f64_from_numpy(ov(a));
        }
        return AbelianGroup::batch_qdim(a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, AbelianGroup, sector_str, a);
    }
    Sector get_sector(std::string const& s) const override
    {
        PYBIND11_OVERRIDE(Sector, AbelianGroup, get_sector, s);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, AbelianGroup, qdim, a); }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, AbelianGroup, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, AbelianGroup, inv_sqrt_qdim, a);
    }
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, _b_symbol, a, b, c);
    }
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override
    {
        PYBIND11_OVERRIDE(FusionSymbol, AbelianGroup, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, AbelianGroup, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of Space (ElementarySpace, TensorProduct, …).
class PySpace
  : public Space
  , public py::trampoline_self_life_support
{
  public:
    using Space::Space;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, Space, test_sanity); }

    Ptr dual_space() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(Ptr, Space, "dual", dual_space);
    }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, Space, "__eq__", operator==, other);
    }

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, Space, as_ElementarySpace, is_dual);
    }

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE_PURE(
          Space::Ptr, Space, change_symmetry, symmetry, sector_map, injective);
    }

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE_PURE(Space::Ptr, Space, drop_symmetry, which);
    }
};

/// Trampoline for Python subclasses of LegPipe.
///
/// ``Leg`` is a value-type handle (not a polymorphic base); LegPipe is independent of Space.
class PyLegPipe
  : public LegPipe
  , public py::trampoline_self_life_support
{
  public:
    using LegPipe::LegPipe;

    // test_sanity is non-virtual on LegPipe.

    std::vector<int64> _flat_leg_permutation(int64 offset) const override
    {
        PYBIND11_OVERRIDE(std::vector<int64>, LegPipe, _flat_leg_permutation, offset);
    }

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override
    {
        PYBIND11_OVERRIDE(void, LegPipe, set_basis_perm, basis_perm);
    }

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override
    {
        PYBIND11_OVERRIDE(void, LegPipe, set_inverse_basis_perm, inverse_basis_perm);
    }
};

/// Trampoline for Python subclasses of ElementarySpace (AbelianLegPipe, …).
///
/// Note: as for the other trampolines here, only virtuals that are bound as *methods* may be
/// forwarded. Those bound as ``def_property`` (``dual``, ``is_trivial``, ``ascii_arrow``,
/// ``flat_legs``, …) must not be, since PYBIND11_OVERRIDE would find the property value instead
/// of a function.
class PyElementarySpace
  : public ElementarySpace
  , public py::trampoline_self_life_support
{
  public:
    using ElementarySpace::ElementarySpace;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, ElementarySpace, test_sanity); }

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, ElementarySpace, as_ElementarySpace, is_dual);
    }

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE(
          Space::Ptr, ElementarySpace, change_symmetry, symmetry, sector_map, injective);
    }

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE(Space::Ptr, ElementarySpace, drop_symmetry, which);
    }

    std::vector<int64> _flat_leg_permutation(int64 offset) const override
    {
        PYBIND11_OVERRIDE(std::vector<int64>, ElementarySpace, _flat_leg_permutation, offset);
    }

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override
    {
        PYBIND11_OVERRIDE(void, ElementarySpace, set_basis_perm, basis_perm);
    }

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override
    {
        PYBIND11_OVERRIDE(void, ElementarySpace, set_inverse_basis_perm, inverse_basis_perm);
    }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, ElementarySpace, "__eq__", operator==, other);
    }
};

/// Trampoline for Python subclasses of AbelianLegPipe.
///
/// Note: as for the other trampolines here, only virtuals that are bound as *methods* may be
/// forwarded. Those bound as ``def_property`` (``dual``, ``is_trivial``, ``flat_spaces``,
/// ``ascii_arrow``) must not be, since PYBIND11_OVERRIDE would find the property value instead
/// of a function. ``is_abelian_leg_pipe`` is deliberately not forwarded either; it must stay
/// ``true``.
class PyAbelianLegPipe
  : public AbelianLegPipe
  , public py::trampoline_self_life_support
{
  public:
    using AbelianLegPipe::AbelianLegPipe;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, AbelianLegPipe, test_sanity); }

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, AbelianLegPipe, as_ElementarySpace, is_dual);
    }

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE(
          Space::Ptr, AbelianLegPipe, change_symmetry, symmetry, sector_map, injective);
    }

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE(Space::Ptr, AbelianLegPipe, drop_symmetry, which);
    }

    std::vector<int64> _flat_leg_permutation(int64 offset) const override
    {
        PYBIND11_OVERRIDE(std::vector<int64>, AbelianLegPipe, _flat_leg_permutation, offset);
    }

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override
    {
        PYBIND11_OVERRIDE(void, AbelianLegPipe, set_basis_perm, basis_perm);
    }

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override
    {
        PYBIND11_OVERRIDE(void, AbelianLegPipe, set_inverse_basis_perm, inverse_basis_perm);
    }

    ElementarySpace::Ptr take_slice(py::array blockmask) const override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, AbelianLegPipe, take_slice, blockmask);
    }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, AbelianLegPipe, "__eq__", operator==, other);
    }
};

/// Trampoline for Python subclasses of DirectSumSpace.
class PyDirectSumSpace
  : public DirectSumSpace
  , public py::trampoline_self_life_support
{
  public:
    using DirectSumSpace::DirectSumSpace;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, DirectSumSpace, test_sanity); }

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, DirectSumSpace, as_ElementarySpace, is_dual);
    }

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE(
          Space::Ptr, DirectSumSpace, change_symmetry, symmetry, sector_map, injective);
    }

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE(Space::Ptr, DirectSumSpace, drop_symmetry, which);
    }

    ElementarySpace::Ptr take_slice(py::array blockmask) const override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, DirectSumSpace, take_slice, blockmask);
    }

    ElementarySpace::Ptr with_opposite_duality() const override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, DirectSumSpace, with_opposite_duality);
    }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, DirectSumSpace, "__eq__", operator==, other);
    }
};

/// Trampoline for Python subclasses of TensorProduct.
///
/// Note: as for the other trampolines here, only virtuals that are bound as *methods* may be
/// forwarded. ``dual_space`` is bound as the ``dual`` property and must therefore be left out;
/// since :class:`TensorProduct` implements it, the trampoline is still concrete.
class PyTensorProduct
  : public TensorProduct
  , public py::trampoline_self_life_support
{
  public:
    using TensorProduct::TensorProduct;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, TensorProduct, test_sanity); }

    Space::Ptr change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE(
          Space::Ptr, TensorProduct, change_symmetry, symmetry, sector_map, injective);
    }

    Space::Ptr drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE(Space::Ptr, TensorProduct, drop_symmetry, which);
    }

    ElementarySpace::Ptr as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(ElementarySpace::Ptr, TensorProduct, as_ElementarySpace, is_dual);
    }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, TensorProduct, "__eq__", operator==, other);
    }
};

} // namespace lasyten
