#include <lasyten/symmetries/symmetry_factor.h>

#include <lasyten/symmetries/symmetry.h>

#include <lasyten/tools/hdf5.h>
#include <stdexcept>
#include <tenpy_hdf5/h5_ops.h>

namespace lasyten {

SymmetryFactor::SymmetryFactor(FusionStyle fusion_style,
                               BraidingStyle braiding_style,
                               Sector trivial_sector,
                               std::string group_name_,
                               float64 num_sectors,
                               bool has_complex_topological_data,
                               std::optional<std::string> descriptive_name_,
                               bool trivial_shift)
  : BaseSymmetry(fusion_style,
                 braiding_style,
                 trivial_sector,
                 num_sectors,
                 has_complex_topological_data,
                 trivial_shift)
  , group_name(std::move(group_name_))
  , descriptive_name(std::move(descriptive_name_))
{
}

bool
SymmetryFactor::is_equivalent_to(BaseSymmetry const& other) const
{
    if (auto const* factor = dynamic_cast<SymmetryFactor const*>(&other)) {
        return _is_equivalent_factor(*factor);
    }
    // Product Symmetry: ask the other side (Symmetry::is_equivalent_to handles factors).
    return false;
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::as_Symmetry()
{
    // Prefer the Python binding (takes py::object) for trampoline instances: smart_holder
    // does not always initialize enable_shared_from_this. C++-only shared_ptr owners work here.
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ self });
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::as_Symmetry: call via Python bindings (no shared_from_this)");
    }
}

std::string
SymmetryFactor::str() const
{
    if (descriptive_name.has_value()) {
        return group_name + " (\"" + *descriptive_name + "\")";
    }
    return group_name;
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::mul(Ptr other)
{
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        return std::make_shared<Symmetry>(
          std::vector<SymmetryFactor::Ptr>{ self, std::move(other) });
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::mul: call via Python bindings (no shared_from_this)");
    }
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::mul(Symmetry const& other)
{
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        std::vector<SymmetryFactor::Ptr> factors;
        factors.reserve(1 + other.factors.size());
        factors.push_back(self);
        factors.insert(factors.end(), other.factors.begin(), other.factors.end());
        return std::make_shared<Symmetry>(std::move(factors));
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::mul: call via Python bindings (no shared_from_this)");
    }
}

bool
SymmetryFactor::equals(SymmetryFactor const& other) const
{
    if (descriptive_name != other.descriptive_name) {
        return false;
    }
    return _is_equivalent_factor(other);
}

void
SymmetryFactor::save_hdf5(lasyten::hdf5::Saver& saver,
                          HighFive::Group& h5gr,
                          std::string const& subpath) const
{
    saver.save_string(subpath + "group_name", group_name);
    saver.save_int64(subpath + "fusion_style", static_cast<std::int64_t>(fusion_style));
    saver.save_int64(subpath + "braiding_style", static_cast<std::int64_t>(braiding_style));
    saver.save_instance(subpath + "trivial_sector",
                        "lasyten._core",
                        "Sector",
                        &trivial_sector,
                        [&](lasyten::hdf5::Saver& s, HighFive::Group& g, std::string const& sub) {
                            trivial_sector.save_hdf5(s, g, sub);
                        });
    saver.save_float64(subpath + "num_sectors", num_sectors);
    saver.save_int64(subpath + "sector_ind_len", static_cast<std::int64_t>(sector_ind_len));
    saver.save_bool(subpath + "trivial_shift", trivial_shift);
    std::string descr = descriptive_name.has_value() ? *descriptive_name : "None";
    tenpy_hdf5::h5_set_attr(h5gr.getId(), "descriptive_name", descr);
    tenpy_hdf5::h5_set_attr(
      h5gr.getId(), "has_complex_topological_data", has_complex_topological_data);
}

void
SymmetryFactor::load_hdf5_common(lasyten::hdf5::Loader& loader,
                                 HighFive::Group& h5gr,
                                 std::string const& subpath)
{
    {
        hid_t id = loader.open(subpath + "group_name");
        group_name = loader.load_string(id);
        H5Idec_ref(id);
    }
    {
        hid_t id = loader.open(subpath + "fusion_style");
        fusion_style = static_cast<FusionStyle>(loader.load_int64(id));
        H5Idec_ref(id);
    }
    {
        hid_t id = loader.open(subpath + "braiding_style");
        braiding_style = static_cast<BraidingStyle>(loader.load_int64(id));
        H5Idec_ref(id);
    }
    {
        hid_t id = loader.open(subpath + "trivial_sector");
        HighFive::Group sector_gr = tenpy_hdf5::group_from_hid(id);
        trivial_sector = Sector::from_hdf5(
          loader, sector_gr, lasyten::hdf5::ensure_slash(subpath + "trivial_sector"));
    }
    {
        hid_t id = loader.open(subpath + "num_sectors");
        num_sectors = loader.load_float64(id);
        H5Idec_ref(id);
    }
    {
        hid_t id = loader.open(subpath + "sector_ind_len");
        sector_ind_len = static_cast<std::uint8_t>(loader.load_int64(id));
        H5Idec_ref(id);
    }
    empty_sector_array = SectorArray::empty(sector_ind_len);
    // trivial_shift was added later; default true if missing.
    trivial_shift = trivial_shift_from_hdf5(loader, subpath);
    descriptive_name = descriptive_name_from_hdf5_attrs(h5gr);
    auto has_complex = tenpy_hdf5::h5_get_attr_int64(h5gr.getId(), "has_complex_topological_data");
    has_complex_topological_data = has_complex.value_or(0) != 0;
}

std::optional<std::string>
descriptive_name_from_hdf5_attrs(HighFive::Group& h5gr)
{
    auto descr = tenpy_hdf5::h5_get_attr_string(h5gr.getId(), "descriptive_name");
    if (!descr || *descr == "None") {
        return std::nullopt;
    }
    return descr;
}

bool
trivial_shift_from_hdf5(lasyten::hdf5::Loader& loader, std::string const& subpath)
{
    if (!tenpy_hdf5::h5_contains(loader.root(), subpath + "trivial_shift")) {
        return true;
    }
    hid_t id = loader.open(subpath + "trivial_shift");
    bool v = loader.load_bool(id);
    H5Idec_ref(id);
    return v;
}

} // namespace lasyten
