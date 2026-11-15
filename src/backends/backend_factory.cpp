#include <lasyten/backends/backend_factory.h>
#if LASYTEN_HAS_PYTHON
#include <lasyten/backends/abelian.h>
#include <lasyten/backends/fusion_tree_backend.h>
#endif
#include <lasyten/backends/no_symmetry.h>
#include <lasyten/block_backend/eigen.h>
#if LASYTEN_WITH_NUMPY_BACKEND
#include <lasyten/block_backend/numpy.h>
#endif
#if LASYTEN_WITH_TORCH_BACKEND
#include <lasyten/block_backend/torch.h>
#endif
#include <lasyten/config.h>
#include <lasyten/symmetries/factors/no_symmetry.h>
#include <lasyten/tools.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lasyten {

namespace {

/// Map a ``block_backend`` name to a (cached singleton) backend instance.
///
/// Preferred names:
///   * ``"eigen"``  -- native C++ Eigen/LAPACK CPU backend (no Python needed; the long-term
///                     default for ``"cpu"``).
///   * ``"torch"``, ``"gpu"``, ``"apple_silicon"`` -- libtorch (GPU / ATen).
///
/// Transitional (convert_nanobind.md, leftover #5; compiled only with
/// ``LASYTEN_WITH_NUMPY_BACKEND``, which needs ``LASYTEN_WITH_PYTHON``):
///   * ``"numpy"``  -- NumpyBlockBackend (dispatches into numpy/scipy through the interpreter).
///   * ``"cpu"``    -- still NumpyBlockBackend while the numpy backend is built, because
///                     EigenBlockBackend has not passed the full test-suite yet. Without
///                     ``LASYTEN_WITH_NUMPY_BACKEND`` it is EigenBlockBackend.
/// ``ArrayApiBlockBackend`` is not reachable through this factory (construct it explicitly).
std::shared_ptr<BlockBackend>
make_block_backend(std::string const& block_backend)
{
    if (block_backend == "eigen") {
        return EigenBlockBackend::from_factory_shared("eigen");
    }
    if (block_backend == "cpu") {
        // NumpyBlockBackend while it is built (TODO(#255): EigenBlockBackend once it has full
        // test-suite parity), else EigenBlockBackend; see default_cpu_block_backend_shared().
        return default_cpu_block_backend_shared();
    }
    if (block_backend == "numpy") {
#if LASYTEN_WITH_NUMPY_BACKEND
        return NumpyBlockBackend::from_factory_shared("cpu");
#else
        throw NotImplemented(
          "block backend \"numpy\" (compiled out: LASYTEN_WITH_NUMPY_BACKEND=OFF); use \"eigen\"");
#endif
    }
    if (block_backend == "torch" || block_backend == "gpu" || block_backend == "apple_silicon") {
#if LASYTEN_WITH_TORCH_BACKEND
        if (block_backend == "torch")
            return TorchBlockBackend::from_factory_shared("cpu:0");
        if (block_backend == "gpu")
            return TorchBlockBackend::from_factory_shared("cuda");
        return TorchBlockBackend::from_factory_shared("mps");
#else
        throw NotImplemented(std::string("block backend ") + block_backend +
                             " (compiled out: LASYTEN_WITH_PYTHON=OFF)");
#endif
    }
    if (block_backend == "tensorflow" || block_backend == "jax" || block_backend == "tpu") {
        throw NotImplemented(std::string("block backend ") + block_backend);
    }
    throw std::invalid_argument("Unknown block_backend: " + block_backend);
}

bool
is_no_symmetry(Symmetry const& symmetry)
{
    Symmetry no_sym{ std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() } };
    return symmetry.is_equivalent_to(no_sym);
}

struct BackendCacheKey
{
    std::string tensor_backend;
    std::string block_backend;

    bool operator==(BackendCacheKey const& other) const noexcept
    {
        return tensor_backend == other.tensor_backend && block_backend == other.block_backend;
    }
};

struct BackendCacheKeyHash
{
    std::size_t operator()(BackendCacheKey const& key) const noexcept
    {
        return std::hash<std::string>{}(key.tensor_backend) ^
               (std::hash<std::string>{}(key.block_backend) << 1);
    }
};

std::unordered_map<BackendCacheKey, TensorBackend::Ptr, BackendCacheKeyHash>&
backend_cache()
{
    static std::unordered_map<BackendCacheKey, TensorBackend::Ptr, BackendCacheKeyHash> cache;
    return cache;
}

std::mutex&
backend_cache_mutex()
{
    static std::mutex mu;
    return mu;
}

TensorBackend::Ptr
make_tensor_backend(std::string const& tensor_backend,
                    std::shared_ptr<BlockBackend> block_backend_instance)
{
    if (tensor_backend == "no_symmetry") {
        return std::make_shared<NoSymmetryBackend>(std::move(block_backend_instance));
    }
    if (tensor_backend == "abelian" || tensor_backend == "fusion_tree") {
#if LASYTEN_HAS_PYTHON
        if (tensor_backend == "abelian") {
            return std::make_shared<AbelianBackend>(std::move(block_backend_instance));
        }
        return std::make_shared<FusionTreeBackend>(std::move(block_backend_instance));
#else
        throw NotImplemented("tensor backend \"" + tensor_backend +
                             "\" (compiled out: LASYTEN_WITH_PYTHON=OFF); only \"no_symmetry\"");
#endif
    }
    throw std::invalid_argument("Unknown tensor_backend: " + tensor_backend);
}

} // namespace

namespace {

/// Python-free core: ``tensor_backend`` / ``block_backend_str`` are already resolved names.
TensorBackend::Ptr
get_backend_by_name(std::string const& tensor_backend,
                    std::string const& block_backend_str,
                    Symmetry::Ptr const& sym_ptr)
{
    BackendCacheKey key{ tensor_backend, block_backend_str };

    {
        std::lock_guard<std::mutex> lock(backend_cache_mutex());
        auto& cache = backend_cache();
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    auto block_backend_instance = make_block_backend(block_backend_str);
    auto backend = make_tensor_backend(tensor_backend, std::move(block_backend_instance));

    if (sym_ptr) {
        if (!backend->supports_symmetry(sym_ptr)) {
            throw std::runtime_error("backend does not support the given symmetry");
        }
    }

    {
        std::lock_guard<std::mutex> lock(backend_cache_mutex());
        auto& cache = backend_cache();
        auto [it, inserted] = cache.emplace(key, backend);
        if (!inserted) {
            backend = it->second;
        }
    }
    return backend;
}

/// Simplest tensor backend that supports ``sym_ptr``.
std::string
tensor_backend_for_symmetry(Symmetry const& sym)
{
    if (is_no_symmetry(sym)) {
        return "no_symmetry";
    }
    if (sym.is_abelian() && sym.has_trivial_braid()) {
        return "abelian";
    }
    return "fusion_tree";
}

} // namespace

TensorBackend::Ptr
get_backend(Symmetry::Ptr symmetry, std::optional<std::string> block_backend)
{
    std::string tensor_backend =
      symmetry ? tensor_backend_for_symmetry(*symmetry) : get_config().default_tensor_backend;
    std::string block_backend_str =
      block_backend.has_value() ? *block_backend : get_config().default_block_backend;
    return get_backend_by_name(tensor_backend, block_backend_str, symmetry);
}

#if LASYTEN_HAS_PYTHON
py::object
get_backend(py::object symmetry, py::object block_backend)
{
    // --- hints from Python get_backend ---
    // figure out minimal symmetry_backend that supports that symmetry
    // ---
    if (symmetry.is_none()) {
        symmetry = py::cast(get_config().default_tensor_backend);
    }
    if (block_backend.is_none()) {
        block_backend = py::cast(get_config().default_block_backend);
    }

    std::string tensor_backend;
    Symmetry::Ptr sym_ptr;
    if (py::isinstance<Symmetry>(symmetry)) {
        sym_ptr = py::cast<Symmetry::Ptr>(symmetry);
        tensor_backend = tensor_backend_for_symmetry(*sym_ptr);
    } else if (py::isinstance<py::str>(symmetry)) {
        tensor_backend = py::cast<std::string>(symmetry);
    } else {
        throw py::type_error("Invalid type for symmetry. Expected Symmetry or str");
    }

    return py::cast(
      get_backend_by_name(tensor_backend, py::cast<std::string>(block_backend), sym_ptr));
}
#endif // LASYTEN_HAS_PYTHON

} // namespace lasyten
