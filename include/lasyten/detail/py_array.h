#pragma once

/// NumPy array adapter on top of nanobind (``lasyten::py::array`` / ``array_t<T>`` / ``dtype``).
///
/// nanobind has no equivalent of ``pybind11::array`` (``nb::ndarray`` is an interchange view,
/// not an owning ``numpy.ndarray`` handle). liblasyten still passes ``numpy.ndarray`` objects
/// through a few APIs (``to_numpy``, the transitional ``NumpyBlockBackend``, models, ...), so this
/// header provides a small pybind11-style adapter built *only* on the Python C-API subset that is
/// part of the Stable ABI (``PyObject_GetBuffer`` + attribute calls into ``numpy``); it needs
/// neither the NumPy C-API nor ``Python.h`` internals, hence works with ``Py_LIMITED_API``.
///
/// Included from ``lasyten/detail/python_api.h`` (only with ``LASYTEN_HAS_PYTHON``); do not
/// include directly.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace lasyten::py {

namespace nb = ::nanobind;

using ssize_t = ::Py_ssize_t;

class array;
class dtype;
template<typename T, int ExtraFlags>
class array_t;

namespace detail {

/// ``numpy`` module (borrowed, never released: avoids ``Py_DECREF`` after ``Py_Finalize``).
inline nb::handle
numpy_module()
{
    static PyObject* mod = nb::module_::import_("numpy").release().ptr();
    return nb::handle(mod);
}

inline bool
array_check(PyObject* o)
{
    static PyObject* ndarray_type = nb::getattr(numpy_module(), "ndarray").release().ptr();
    int const r = PyObject_IsInstance(o, ndarray_type);
    if (r < 0) {
        PyErr_Clear();
        return false;
    }
    return r == 1;
}

inline bool
dtype_check(PyObject* o)
{
    static PyObject* dtype_type = nb::getattr(numpy_module(), "dtype").release().ptr();
    int const r = PyObject_IsInstance(o, dtype_type);
    if (r < 0) {
        PyErr_Clear();
        return false;
    }
    return r == 1;
}

/// NumPy dtype name for a C++ scalar type.
template<typename T>
constexpr char const*
numpy_dtype_name()
{
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<U, std::complex<float>>) {
        return "complex64";
    } else if constexpr (std::is_same_v<U, std::complex<double>>) {
        return "complex128";
    } else if constexpr (std::is_floating_point_v<U>) {
        return sizeof(U) == 4 ? "float32" : "float64";
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
        return sizeof(U) == 1   ? "int8"
               : sizeof(U) == 2 ? "int16"
               : sizeof(U) == 4 ? "int32"
                                : "int64";
    } else {
        static_assert(std::is_integral_v<U>, "unsupported array element type");
        return sizeof(U) == 1   ? "uint8"
               : sizeof(U) == 2 ? "uint16"
               : sizeof(U) == 4 ? "uint32"
                                : "uint64";
    }
}

} // namespace detail

/// Result of ``array::request()`` (a snapshot of the buffer protocol view).
struct buffer_info
{
    void* ptr = nullptr;
    ssize_t itemsize = 0;
    std::string format;
    ssize_t ndim = 0;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides; ///< in bytes
    bool readonly = false;

    ssize_t size = 0; ///< number of elements

    /// Is the buffer's element type (``format`` / ``itemsize``) the C++ scalar type ``T``?
    template<typename T>
    [[nodiscard]] bool item_type_is_equivalent_to() const
    {
        using U = std::remove_cv_t<T>;
        std::size_t k = 0;
        while (k < format.size() && (format[k] == '@' || format[k] == '=' || format[k] == '<')) {
            ++k;
        }
        std::string const f = format.substr(k);
        if (f.empty()) {
            return false;
        }
        if constexpr (std::is_same_v<U, bool>) {
            return f == "?";
        } else if constexpr (std::is_same_v<U, std::complex<float>> ||
                             std::is_same_v<U, std::complex<double>>) {
            return f.size() == 2 && f[0] == 'Z' && itemsize == static_cast<ssize_t>(sizeof(U));
        } else if constexpr (std::is_floating_point_v<U>) {
            return f.size() == 1 && (f[0] == 'f' || f[0] == 'd') &&
                   itemsize == static_cast<ssize_t>(sizeof(U));
        } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
            return f.size() == 1 &&
                   std::string_view("bhilqn").find(f[0]) != std::string_view::npos &&
                   itemsize == static_cast<ssize_t>(sizeof(U));
        } else {
            return f.size() == 1 &&
                   std::string_view("BHILQN").find(f[0]) != std::string_view::npos &&
                   itemsize == static_cast<ssize_t>(sizeof(U));
        }
    }
};

namespace detail {

/// Snapshot the buffer view of a Python object (the pointer stays valid as long as the object
/// is alive and not resized, as for numpy arrays).
inline buffer_info
get_buffer_info(PyObject* o, bool writable)
{
    Py_buffer view;
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (PyObject_GetBuffer(o, &view, flags) != 0) {
        throw nb::python_error();
    }
    buffer_info info;
    info.ptr = view.buf;
    info.itemsize = view.itemsize;
    info.format = view.format ? view.format : "B";
    info.ndim = view.ndim;
    info.readonly = view.readonly != 0;
    info.shape.assign(view.shape, view.shape + view.ndim);
    info.size = 1;
    for (auto sh : info.shape) {
        info.size *= sh;
    }
    if (view.strides) {
        info.strides.assign(view.strides, view.strides + view.ndim);
    } else {
        info.strides.resize(static_cast<std::size_t>(view.ndim));
        ssize_t st = view.itemsize;
        for (ssize_t i = view.ndim - 1; i >= 0; --i) {
            info.strides[static_cast<std::size_t>(i)] = st;
            st *= info.shape[static_cast<std::size_t>(i)];
        }
    }
    PyBuffer_Release(&view);
    return info;
}

inline nb::tuple
shape_tuple(std::vector<ssize_t> const& shape)
{
    nb::builder<nb::tuple> b(shape.size());
    for (auto s : shape) {
        b.put(nb::int_(s));
    }
    return b.commit();
}

} // namespace detail

/// ``numpy.dtype`` object.
class dtype : public nb::object
{
  public:
    static constexpr bool nb_typed = true;
    static constexpr auto Name = nb::detail::const_name("numpy.dtype");
    dtype() = default;
    dtype(nb::handle h, nb::detail::borrow_t)
      : nb::object(h, nb::detail::borrow_t{})
    {
    }
    dtype(nb::handle h, nb::detail::steal_t)
      : nb::object(h, nb::detail::steal_t{})
    {
    }
    static bool check_(nb::handle h) { return detail::dtype_check(h.ptr()); }

    /// ``numpy.dtype(arg)`` (``arg``: name / type / dtype / descriptor).
    explicit dtype(nb::handle arg)
      : nb::object(nb::steal(detail::numpy_module().attr("dtype")(arg).release()))
    {
    }
    explicit dtype(char const* name)
      : dtype(nb::handle(nb::str(name).ptr()))
    {
    }
    explicit dtype(std::string const& name)
      : dtype(name.c_str())
    {
    }

    template<typename T>
    static dtype of()
    {
        return dtype(detail::numpy_dtype_name<T>());
    }

    [[nodiscard]] char kind() const
    {
        std::string const k = nb::cast<std::string>(attr("kind"));
        return k.empty() ? '\0' : k[0];
    }
    [[nodiscard]] ssize_t itemsize() const { return nb::cast<ssize_t>(attr("itemsize")); }
    [[nodiscard]] char byteorder() const
    {
        std::string const k = nb::cast<std::string>(attr("byteorder"));
        return k.empty() ? '\0' : k[0];
    }
};

/// ``numpy.ndarray`` object (pybind11-style adapter, see file comment).
class array : public nb::object
{
  public:
    static constexpr bool nb_typed = true;
    static constexpr auto Name = nb::detail::const_name("numpy.ndarray");

    enum
    {
        c_style = 1,
        f_style = 2,
        forcecast = 16,
    };

    array() = default;
    array(nb::handle h, nb::detail::borrow_t)
      : nb::object(h, nb::detail::borrow_t{})
    {
    }
    array(nb::handle h, nb::detail::steal_t)
      : nb::object(h, nb::detail::steal_t{})
    {
    }
    static bool check_(nb::handle h) { return detail::array_check(h.ptr()); }

    /// Convert any array-like (``numpy.asarray``); raises on failure.
    explicit array(nb::handle h)
      : nb::object(nb::steal(detail::numpy_module().attr("asarray")(h).release()))
    {
    }
    // implicit, as in pybind11: ``py::array a = some_object;`` converts via ``numpy.asarray``
    array(nb::object const& o)
      : array(static_cast<nb::handle>(o))
    {
    }
    array(nb::object&& o)
      : array(static_cast<nb::handle>(o))
    {
    }

    /// New uninitialised array (or a copy of ``ptr``'s data, if given).
    array(py::dtype const& dt, std::vector<ssize_t> const& shape, void const* ptr = nullptr)
      : nb::object(nb::steal(
          detail::numpy_module().attr("empty")(detail::shape_tuple(shape), dt).release()))
    {
        if (ptr != nullptr) {
            auto info = detail::get_buffer_info(m_ptr, true);
            std::memcpy(info.ptr, ptr, static_cast<std::size_t>(info.size * info.itemsize));
        }
    }
    array(py::dtype const& dt, ssize_t count, void const* ptr = nullptr)
      : array(dt, std::vector<ssize_t>{ count }, ptr)
    {
    }

    /// Zero-copy view of foreign memory (``ptr``, strides in *bytes*; empty ``strides`` = C
    /// order), keeping ``base`` alive. The array is writable and aliases ``ptr``.
    array(py::dtype const& dt,
          std::vector<ssize_t> const& shape,
          std::vector<ssize_t> const& strides,
          void* ptr,
          nb::handle base)
      : nb::object(make_view_(dt, shape, strides, ptr, base))
    {
    }

    /// ``numpy.asarray(h)`` (``ensure(...)``-style: returns a null array on failure).
    static array ensure(nb::handle h)
    {
        if (!h.is_valid()) {
            return array();
        }
        try {
            return nb::steal<array>(detail::numpy_module().attr("asarray")(h).release());
        } catch (nb::python_error&) {
            PyErr_Clear();
            return array();
        }
    }

    [[nodiscard]] buffer_info request(bool writable = false) const
    {
        return detail::get_buffer_info(m_ptr, writable);
    }

    [[nodiscard]] ssize_t ndim() const { return nb::cast<ssize_t>(attr("ndim")); }
    [[nodiscard]] ssize_t size() const { return nb::cast<ssize_t>(attr("size")); }
    [[nodiscard]] ssize_t itemsize() const { return nb::cast<ssize_t>(attr("itemsize")); }
    [[nodiscard]] ssize_t nbytes() const { return nb::cast<ssize_t>(attr("nbytes")); }
    [[nodiscard]] std::vector<ssize_t> shape_vector() const
    {
        return nb::cast<std::vector<ssize_t>>(attr("shape"));
    }
    [[nodiscard]] ssize_t shape(ssize_t dim) const
    {
        auto s = shape_vector();
        if (dim < 0 || dim >= static_cast<ssize_t>(s.size())) {
            throw nb::index_error("invalid axis");
        }
        return s[static_cast<std::size_t>(dim)];
    }
    // pybind11 spelling ``arr.dtype()``; inside this class the name ``dtype`` is the member
    // function, so the type is always spelled ``py::dtype`` here.
    [[nodiscard]] py::dtype dtype() const { return nb::borrow<py::dtype>(attr("dtype")); }

    [[nodiscard]] void const* data() const { return request().ptr; }
    [[nodiscard]] void* mutable_data() { return request(true).ptr; }

    [[nodiscard]] array reshape(std::vector<ssize_t> const& new_shape) const
    {
        return nb::steal<array>(attr("reshape")(detail::shape_tuple(new_shape)).release());
    }
    [[nodiscard]] array squeeze() const { return nb::steal<array>(attr("squeeze")().release()); }

  private:
    static nb::object make_view_(py::dtype const& dt,
                                 std::vector<ssize_t> const& shape,
                                 std::vector<ssize_t> const& strides,
                                 void* ptr,
                                 nb::handle base)
    {
        using nb::dlpack::dtype_code;
        ssize_t const itemsize = dt.itemsize();
        char const kind = dt.kind();
        std::uint8_t code = static_cast<std::uint8_t>(dtype_code::Float);
        std::uint8_t bits = static_cast<std::uint8_t>(itemsize * 8);
        switch (kind) {
            case 'b':
                code = static_cast<std::uint8_t>(dtype_code::Bool);
                break;
            case 'i':
                code = static_cast<std::uint8_t>(dtype_code::Int);
                break;
            case 'u':
                code = static_cast<std::uint8_t>(dtype_code::UInt);
                break;
            case 'f':
                code = static_cast<std::uint8_t>(dtype_code::Float);
                break;
            case 'c':
                code = static_cast<std::uint8_t>(dtype_code::Complex);
                break;
            default:
                throw nb::type_error("array: unsupported dtype for a zero-copy view");
        }
        std::vector<std::size_t> shp(shape.begin(), shape.end());
        std::vector<std::int64_t> st;
        for (auto b : strides) {
            st.push_back(b / itemsize); // nb::ndarray strides are in elements
        }
        nb::ndarray<nb::numpy> a(ptr,
                                 shp.size(),
                                 shp.data(),
                                 base,
                                 st.empty() ? nullptr : st.data(),
                                 nb::dlpack::dtype{ code, bits, 1 });
        return nb::cast(a);
    }
};

namespace detail {

/// Strided element access to array data (``unchecked<T, N>()`` / ``mutable_unchecked``).
template<typename T, ssize_t Dims, bool Mutable>
class unchecked_ref
{
  public:
    unchecked_ref(array arr)
      : arr_(std::move(arr))
    {
        auto info = arr_.request(Mutable);
        if (Dims >= 0 && info.ndim != Dims) {
            throw nb::index_error("array has incorrect number of dimensions");
        }
        data_ = static_cast<std::byte*>(info.ptr);
        shape_ = std::move(info.shape);
        strides_ = std::move(info.strides);
    }

    template<typename... Ix>
    std::conditional_t<Mutable, T&, T const&> operator()(Ix... idx) const
    {
        ssize_t const ix[] = { static_cast<ssize_t>(idx)..., 0 };
        std::byte* p = data_;
        for (std::size_t d = 0; d < sizeof...(Ix); ++d) {
            p += ix[d] * strides_[d];
        }
        return *reinterpret_cast<T*>(p);
    }

    [[nodiscard]] ssize_t shape(ssize_t d) const { return shape_[static_cast<std::size_t>(d)]; }
    [[nodiscard]] ssize_t ndim() const { return static_cast<ssize_t>(shape_.size()); }
    [[nodiscard]] ssize_t size() const
    {
        ssize_t n = 1;
        for (auto s : shape_) {
            n *= s;
        }
        return n;
    }

  private:
    array arr_; // keeps the data alive
    std::byte* data_ = nullptr;
    std::vector<ssize_t> shape_;
    std::vector<ssize_t> strides_;
};

} // namespace detail

/// ``numpy.ndarray`` of a fixed element type ``T``. ``ExtraFlags``: ``array::c_style``,
/// ``array::forcecast`` (default, as in pybind11) ...
template<typename T, int ExtraFlags = array::forcecast>
class array_t : public array
{
  public:
    using value_type = T;
    static constexpr auto Name = nb::detail::const_name("numpy.ndarray");

    array_t() = default;
    array_t(nb::handle h, nb::detail::borrow_t)
      : array(h, nb::detail::borrow_t{})
    {
    }
    array_t(nb::handle h, nb::detail::steal_t)
      : array(h, nb::detail::steal_t{})
    {
    }

    /// Convert any array-like to this element type (raises on failure).
    explicit array_t(nb::handle h)
      : array(ensure(h))
    {
        if (!m_ptr) {
            throw nb::type_error("cannot convert object to array of the requested dtype");
        }
    }
    array_t(nb::object const& o)
      : array_t(static_cast<nb::handle>(o))
    {
    }
    array_t(array const& a)
      : array_t(static_cast<nb::handle>(a))
    {
    }
    template<int F2, std::enable_if_t<F2 != ExtraFlags, int> = 0>
    array_t(array_t<T, F2> const& a)
      : array_t(static_cast<nb::handle>(a))
    {
    }

    explicit array_t(std::vector<ssize_t> const& shape, T const* ptr = nullptr)
      : array(py::dtype::of<T>(), shape, ptr)
    {
    }
    array_t(std::initializer_list<ssize_t> shape, T const* ptr = nullptr)
      : array(py::dtype::of<T>(), std::vector<ssize_t>(shape), ptr)
    {
    }
    explicit array_t(ssize_t count, T const* ptr = nullptr)
      : array(py::dtype::of<T>(), std::vector<ssize_t>{ count }, ptr)
    {
    }

    /// Does ``h`` already have a matching dtype / layout (no conversion needed)?
    static bool check_(nb::handle h)
    {
        if (!array::check_(h)) {
            return false;
        }
        auto dt = nb::borrow<py::dtype>(nb::getattr(h, "dtype"));
        auto want = py::dtype::of<T>();
        if (!dt.equal(want)) {
            return false;
        }
        if constexpr ((ExtraFlags & array::c_style) != 0) {
            if (!nb::cast<bool>(nb::getattr(nb::getattr(h, "flags"), "c_contiguous"))) {
                return false;
            }
        }
        if constexpr ((ExtraFlags & array::f_style) != 0) {
            if (!nb::cast<bool>(nb::getattr(nb::getattr(h, "flags"), "f_contiguous"))) {
                return false;
            }
        }
        return true;
    }

    /// Convert to this element type (+ layout); a null array on failure.
    static array_t ensure(nb::handle h)
    {
        if (!h.is_valid()) {
            return array_t();
        }
        try {
            nb::object np = nb::borrow(detail::numpy_module());
            nb::object src = np.attr("asarray")(h);
            nb::object want = np.attr("dtype")(detail::numpy_dtype_name<T>());
            nb::object cur = src.attr("dtype");
            nb::object out = src;
            if (!cur.equal(want)) {
                if ((ExtraFlags & array::forcecast) == 0 &&
                    !nb::cast<bool>(np.attr("can_cast")(cur, want, "safe"))) {
                    return array_t();
                }
                out = src.attr("astype")(want);
            }
            if constexpr ((ExtraFlags & array::c_style) != 0) {
                out = np.attr("ascontiguousarray")(out);
            } else if constexpr ((ExtraFlags & array::f_style) != 0) {
                out = np.attr("asfortranarray")(out);
            }
            return nb::steal<array_t>(out.release());
        } catch (nb::python_error&) {
            PyErr_Clear();
            return array_t();
        }
    }

    [[nodiscard]] T const* data() const { return static_cast<T const*>(request().ptr); }
    [[nodiscard]] T* mutable_data() { return static_cast<T*>(request(true).ptr); }

    template<typename... Ix>
    T const& operator()(Ix... idx) const
    {
        return detail::unchecked_ref<T, -1, false>(*this)(idx...);
    }

    /// Bounds-checked element access (``at``) / write access (``mutable_at``).
    template<typename... Ix>
    [[nodiscard]] T const& at(Ix... idx) const
    {
        check_bounds_({ static_cast<ssize_t>(idx)... });
        return (*this)(idx...);
    }
    template<typename... Ix>
    [[nodiscard]] T& mutable_at(Ix... idx)
    {
        check_bounds_({ static_cast<ssize_t>(idx)... });
        return const_cast<T&>(detail::unchecked_ref<T, -1, true>(*this)(idx...));
    }

    template<ssize_t Dims = -1>
    [[nodiscard]] detail::unchecked_ref<T, Dims, false> unchecked() const
    {
        return detail::unchecked_ref<T, Dims, false>(*this);
    }
    template<ssize_t Dims = -1>
    [[nodiscard]] detail::unchecked_ref<T, Dims, true> mutable_unchecked()
    {
        return detail::unchecked_ref<T, Dims, true>(*this);
    }

    [[nodiscard]] array_t reshape(std::vector<ssize_t> const& new_shape) const
    {
        return nb::steal<array_t>(attr("reshape")(detail::shape_tuple(new_shape)).release());
    }

  private:
    void check_bounds_(std::initializer_list<ssize_t> idx) const
    {
        auto const sh = shape_vector();
        if (idx.size() != sh.size()) {
            throw nb::index_error("index dimension mismatch");
        }
        std::size_t d = 0;
        for (auto i : idx) {
            if (i < 0 || i >= sh[d++]) {
                throw nb::index_error("index out of range");
            }
        }
    }
};

} // namespace lasyten::py

// ---------------------------------------------------------------------------------------------
// nanobind type casters: ``None`` handling is left to the call; implicit conversion
// (``numpy.asarray`` / ``astype``) is applied when nanobind's ``convert`` flag is set, as in the
// pybind11 casters these types replace.
// ---------------------------------------------------------------------------------------------
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template<>
struct type_caster<lasyten::py::dtype>
{
    NB_TYPE_CASTER(lasyten::py::dtype, lasyten::py::dtype::Name)

    type_caster()
      : value(nullptr, steal_t())
    {
    }

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        try {
            if (lasyten::py::dtype::check_(src)) {
                value = borrow<lasyten::py::dtype>(src);
                return true;
            }
            if (flags & (uint32_t)cast_flags::convert) {
                value = lasyten::py::dtype(src);
                return true;
            }
        } catch (...) {
            PyErr_Clear();
        }
        return false;
    }

    static handle from_cpp(lasyten::py::dtype const& src, rv_policy, cleanup_list*) noexcept
    {
        return src.inc_ref();
    }
    static handle from_cpp(lasyten::py::dtype&& src, rv_policy, cleanup_list*) noexcept
    {
        return src.release();
    }
};

template<>
struct type_caster<lasyten::py::array>
{
    NB_TYPE_CASTER(lasyten::py::array, lasyten::py::array::Name)

    type_caster()
      : value(nullptr, steal_t())
    {
    }

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        try {
            if (lasyten::py::array::check_(src)) {
                value = borrow<lasyten::py::array>(src);
                return true;
            }
            if (flags & (uint32_t)cast_flags::convert) {
                value = lasyten::py::array::ensure(src);
                return value.is_valid();
            }
        } catch (...) {
            PyErr_Clear();
        }
        return false;
    }

    static handle from_cpp(lasyten::py::array const& src, rv_policy, cleanup_list*) noexcept
    {
        return src.inc_ref();
    }
    static handle from_cpp(lasyten::py::array&& src, rv_policy, cleanup_list*) noexcept
    {
        return src.release();
    }
};

template<typename T, int F>
struct type_caster<lasyten::py::array_t<T, F>>
{
    using Arr = lasyten::py::array_t<T, F>;
    NB_TYPE_CASTER(Arr, Arr::Name)

    type_caster()
      : value(nullptr, steal_t())
    {
    }

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        try {
            if (Arr::check_(src)) {
                value = borrow<Arr>(src);
                return true;
            }
            if (flags & (uint32_t)cast_flags::convert) {
                value = Arr::ensure(src);
                return value.is_valid();
            }
        } catch (...) {
            PyErr_Clear();
        }
        return false;
    }

    static handle from_cpp(Arr const& src, rv_policy, cleanup_list*) noexcept
    {
        return src.inc_ref();
    }
    static handle from_cpp(Arr&& src, rv_policy, cleanup_list*) noexcept { return src.release(); }
};

// ``None`` is accepted for these (the casters above decide; ``None`` -> ``asarray(None)`` only
// with ``convert``).
template<>
struct has_arg_defaults<lasyten::py::array> : std::true_type
{};
template<>
struct has_arg_defaults<lasyten::py::dtype> : std::true_type
{};
template<typename T, int F>
struct has_arg_defaults<lasyten::py::array_t<T, F>> : std::true_type
{};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
