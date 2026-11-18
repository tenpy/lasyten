// Native Eigen + LAPACK(E) BlockBackend.
//
// Blocks own contiguous C-ordered CPU buffers. All compute (elementwise, reductions,
// contractions, decompositions) is plain C++: Eigen for products / matrix functions and
// LAPACKE for SVD, QR, eigh and eig. Python objects are only used at the boundary
// (to_numpy / block_from_numpy / as_block / as_scalar(py::object) / repr), and that boundary is
// compiled only with LASYTEN_HAS_PYTHON; HDF5 I/O is Python-free. The nb::ndarray exchange for
// the Python bindings lives in pybind/block_backend/eigen_ndarray.h.

#include <lasyten/block_backend/eigen.h>
#include <lasyten/tools.h>
#include <lasyten/tools/hdf5_export.h>
#include <lasyten/tools/warn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <variant>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// LAPACKE with std::complex as the complex type (must be defined before the include).
#ifndef LAPACK_COMPLEX_CUSTOM
#define LAPACK_COMPLEX_CUSTOM
#endif
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define lapack_complex_float_real(z) ((z).real())
#define lapack_complex_float_imag(z) ((z).imag())
#define lapack_complex_double_real(z) ((z).real())
#define lapack_complex_double_imag(z) ((z).imag())
#include <lapacke.h>

namespace lasyten {

namespace dtype {

std::size_t
itemsize(Dtype d)
{
    switch (d) {
        case Dtype::Bool:
            return 1;
        case Dtype::Int64:
            return 8;
        case Dtype::Float32:
            return 4;
        case Dtype::Complex64:
            return 8;
        case Dtype::Float64:
            return 8;
        case Dtype::Complex128:
            return 16;
        default:
            throw std::invalid_argument("unknown Dtype");
    }
}

} // namespace dtype

namespace {

using Blk = EigenBlockBackend::Block;
using BPtr = std::shared_ptr<Blk>;
using CBPtr = std::shared_ptr<const Blk>;
using Shape = std::vector<int64>;
using BlockIndex = BlockBackend::BlockIndex;
using Scalar = BlockBackend::Scalar;

// -----------------------------------------------------------------------------
// type utilities
// -----------------------------------------------------------------------------

template<class T>
struct is_cplx : std::false_type
{};
template<class R>
struct is_cplx<std::complex<R>> : std::true_type
{};
template<class T>
inline constexpr bool is_cplx_v = is_cplx<T>::value;

template<class T>
struct real_of
{
    using type = T;
};
template<class R>
struct real_of<std::complex<R>>
{
    using type = R;
};
template<class T>
using real_t = typename real_of<T>::type;

/// Call ``f(T{})`` with ``T`` the C++ type of the dtype.
template<class F>
void
visit_dtype(Dtype d, F&& f)
{
    switch (d) {
        case Dtype::Bool:
            f(bool{});
            return;
        case Dtype::Int64:
            f(int64{});
            return;
        case Dtype::Float32:
            f(float32{});
            return;
        case Dtype::Complex64:
            f(complex64{});
            return;
        case Dtype::Float64:
            f(float64{});
            return;
        case Dtype::Complex128:
            f(complex128{});
            return;
        default:
            throw std::invalid_argument("unknown Dtype");
    }
}

/// Like `visit_dtype`, but only for float32/float64/complex64/complex128.
template<class F>
void
visit_inexact(Dtype d, F&& f)
{
    switch (d) {
        case Dtype::Float32:
            f(float32{});
            return;
        case Dtype::Complex64:
            f(complex64{});
            return;
        case Dtype::Float64:
            f(float64{});
            return;
        case Dtype::Complex128:
            f(complex128{});
            return;
        default:
            throw std::invalid_argument("operation requires a floating point or complex dtype");
    }
}

/// Convert one entry.
template<class To, class From>
To
conv(const From& v)
{
    if constexpr (std::is_same_v<To, From>) {
        return v;
    } else if constexpr (is_cplx_v<From>) {
        if constexpr (is_cplx_v<To>)
            return To(static_cast<real_t<To>>(v.real()), static_cast<real_t<To>>(v.imag()));
        else if constexpr (std::is_same_v<To, bool>)
            return v != From{};
        else
            return static_cast<To>(v.real());
    } else if constexpr (is_cplx_v<To>) {
        return To(static_cast<real_t<To>>(v), real_t<To>(0));
    } else if constexpr (std::is_same_v<To, bool>) {
        return v != From{};
    } else {
        return static_cast<To>(v);
    }
}

template<class T>
bool
less_than(const T& x, const T& y)
{
    if constexpr (is_cplx_v<T>)
        return x.real() < y.real() || (x.real() == y.real() && x.imag() < y.imag());
    else
        return x < y;
}

/// |x| as a float64 for any entry type.
template<class T>
float64
abs_f64(const T& x)
{
    if constexpr (std::is_same_v<T, bool>)
        return x ? 1.0 : 0.0;
    else if constexpr (is_cplx_v<T>)
        return static_cast<float64>(std::abs(x));
    else
        return std::abs(static_cast<float64>(x));
}

/// dtype used for operations that need an inexact type (ints and bools become float64).
Dtype
float_dtype(Dtype d)
{
    return (d == Dtype::Bool || d == Dtype::Int64) ? Dtype::Float64 : d;
}

/// Real dtype with the same precision (for abs, norm, ...); ints and bools become float64.
Dtype
real_dtype(Dtype d)
{
    d = float_dtype(d);
    return dtype::is_complex(d) ? dtype::to_real(d) : d;
}

// -----------------------------------------------------------------------------
// shapes and strides
// -----------------------------------------------------------------------------

int64
prod(const Shape& s)
{
    int64 n = 1;
    for (int64 v : s)
        n *= v;
    return n;
}

Shape
c_strides(const Shape& s)
{
    Shape st(s.size(), 1);
    for (int64 i = static_cast<int64>(s.size()) - 2; i >= 0; --i)
        st[i] = st[i + 1] * s[i + 1];
    return st;
}

std::string
shape_str(const Shape& s)
{
    std::string out = "(";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(s[i]);
    }
    if (s.size() == 1)
        out += ",";
    return out + ")";
}

Shape
broadcast_shapes(const Shape& a, const Shape& b)
{
    size_t nd = std::max(a.size(), b.size());
    Shape out(nd, 1);
    for (size_t i = 0; i < nd; ++i) {
        int64 da = i < nd - a.size() ? 1 : a[i - (nd - a.size())];
        int64 db = i < nd - b.size() ? 1 : b[i - (nd - b.size())];
        if (da == db || db == 1)
            out[i] = da;
        else if (da == 1)
            out[i] = db;
        else
            throw std::invalid_argument("shapes " + shape_str(a) + " and " + shape_str(b) +
                                        " can not be broadcast together");
    }
    return out;
}

/// Element strides to read an array of shape ``in`` as if it had shape ``out``.
Shape
broadcast_strides(const Shape& in, const Shape& out)
{
    if (in.size() > out.size())
        throw std::invalid_argument("can not broadcast " + shape_str(in) + " to " +
                                    shape_str(out));
    Shape st_in = c_strides(in);
    Shape st(out.size(), 0);
    size_t off = out.size() - in.size();
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == out[off + i])
            st[off + i] = st_in[i];
        else if (in[i] == 1)
            st[off + i] = 0;
        else
            throw std::invalid_argument("can not broadcast " + shape_str(in) + " to " +
                                        shape_str(out));
    }
    return st;
}

/// Loop over a C-ordered ``shape``; call ``f(flat_out_index, offsets)`` where ``offsets[k]`` is
/// the (element) offset into the k-th operand given by ``strides[k]``.
template<size_t K, class F>
void
strided_loop(const Shape& shape, const std::array<Shape, K>& strides, F&& f)
{
    const int64 nd = static_cast<int64>(shape.size());
    if (prod(shape) == 0)
        return;
    std::array<int64, K> off{};
    if (nd == 0) {
        f(int64(0), off);
        return;
    }
    const int64 inner = shape[nd - 1];
    std::array<int64, K> istr{};
    for (size_t k = 0; k < K; ++k)
        istr[k] = strides[k][nd - 1];
    Shape idx(nd, 0);
    int64 out = 0;
    while (true) {
        std::array<int64, K> o = off;
        for (int64 j = 0; j < inner; ++j) {
            f(out++, o);
            for (size_t k = 0; k < K; ++k)
                o[k] += istr[k];
        }
        int64 d = nd - 2;
        for (; d >= 0; --d) {
            ++idx[d];
            for (size_t k = 0; k < K; ++k)
                off[k] += strides[k][d];
            if (idx[d] < shape[d])
                break;
            for (size_t k = 0; k < K; ++k)
                off[k] -= strides[k][d] * shape[d];
            idx[d] = 0;
        }
        if (d < 0)
            break;
    }
}

struct AxisSplit
{
    int64 outer, n, inner;
};

AxisSplit
split_axis(const Shape& s, int64 axis)
{
    AxisSplit r{ 1, s[axis], 1 };
    for (int64 i = 0; i < axis; ++i)
        r.outer *= s[i];
    for (size_t i = axis + 1; i < s.size(); ++i)
        r.inner *= s[i];
    return r;
}

// -----------------------------------------------------------------------------
// block helpers
// -----------------------------------------------------------------------------

BPtr
make(Dtype d, Shape s)
{
    return std::make_shared<Blk>(d, std::move(s));
}

BPtr
copy_of(const Blk& a)
{
    return std::make_shared<Blk>(a.dtype(), a.shape(), a.bytes());
}

BPtr
with_shape(const Blk& a, Shape s)
{
    if (prod(s) != a.numel())
        throw std::invalid_argument("can not view block of shape " + shape_str(a.shape()) +
                                    " as shape " + shape_str(s));
    return std::make_shared<Blk>(a.dtype(), std::move(s), a.bytes());
}

Scalar
mk_scalar(BPtr b)
{
    return Scalar(std::shared_ptr<BlockBackend::Block>(std::move(b)));
}

template<class T>
BPtr
scalar_block(T v)
{
    BPtr b = make(dtype::of<T>(), {});
    *b->data<T>() = v;
    return b;
}

BPtr
cast_copy(const Blk& a, Dtype d)
{
    if (a.dtype() == d)
        return copy_of(a);
    BPtr res = make(d, a.shape());
    const int64 n = a.numel();
    visit_dtype(a.dtype(), [&](auto ft) {
        using F = decltype(ft);
        visit_dtype(d, [&](auto tt) {
            using T = decltype(tt);
            const F* src = a.data<F>();
            T* dst = res->data<T>();
            for (int64 i = 0; i < n; ++i)
                dst[i] = conv<T>(src[i]);
        });
    });
    return res;
}

/// ``a`` converted to dtype ``d``; aliases ``a`` (no copy) if the dtype already matches.
CBPtr
as_dtype(const Blk& a, Dtype d)
{
    if (a.dtype() == d)
        return CBPtr(CBPtr{}, &a);
    return cast_copy(a, d);
}

const Blk&
eb_ref(const BlockCPtr& b)
{
    auto const* p = dynamic_cast<const Blk*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not an EigenBlockBackend::Block");
    return *p;
}

/// Accept blocks of any backend (foreign blocks are converted through numpy).
CBPtr
coerce(const BlockBackend::Block& o)
{
    if (auto const* p = dynamic_cast<const Blk*>(&o))
        return CBPtr(CBPtr{}, p);
#if LASYTEN_HAS_PYTHON
    return std::make_shared<Blk>(o.to_numpy());
#else
    throw std::invalid_argument(
      "EigenBlockBackend: block of a foreign backend (needs LASYTEN_WITH_PYTHON to convert)");
#endif
}

int64
wrap_axis(int64 axis, int64 nd)
{
    return to_valid_idx(axis, nd);
}

std::vector<int64>
int64_vector(const Blk& b)
{
    if (b.dtype() != Dtype::Int64)
        throw std::invalid_argument("expected an int64 block");
    return std::vector<int64>(b.data<int64>(), b.data<int64>() + b.numel());
}

// -----------------------------------------------------------------------------
// elementwise kernels
// -----------------------------------------------------------------------------

template<class TIn, class TOut, class F>
BPtr
map_unary(const Blk& a, Dtype out_dtype, F&& f)
{
    BPtr res = make(out_dtype, a.shape());
    const TIn* pa = a.data<TIn>();
    TOut* po = res->data<TOut>();
    const int64 n = a.numel();
    for (int64 i = 0; i < n; ++i)
        po[i] = f(pa[i]);
    return res;
}

template<class TIn, class TOut, class F>
BPtr
map_binary(const Blk& a, const Blk& b, Dtype out_dtype, F&& f)
{
    const Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    BPtr res = make(out_dtype, out_shape);
    const TIn* pa = a.data<TIn>();
    const TIn* pb = b.data<TIn>();
    TOut* po = res->data<TOut>();
    if (a.shape() == b.shape()) {
        const int64 n = a.numel();
        for (int64 i = 0; i < n; ++i)
            po[i] = f(pa[i], pb[i]);
    } else {
        std::array<Shape, 2> st{ broadcast_strides(a.shape(), out_shape),
                                 broadcast_strides(b.shape(), out_shape) };
        strided_loop<2>(out_shape, st, [&](int64 i, const std::array<int64, 2>& o) {
            po[i] = f(pa[o[0]], pb[o[1]]);
        });
    }
    return res;
}

template<class T>
T
pow_elem(const T& x, const T& y)
{
    if constexpr (std::is_same_v<T, int64>) {
        if (y < 0)
            throw std::invalid_argument("Integers to negative integer powers are not allowed.");
        int64 result = 1;
        int64 base = x;
        int64 e = y;
        while (e > 0) {
            if (e & 1)
                result *= base;
            base *= base;
            e >>= 1;
        }
        return result;
    } else if constexpr (std::is_same_v<T, bool>) {
        return x || !y;
    } else {
        return std::pow(x, y);
    }
}

enum class BinOp
{
    Add,
    Sub,
    Mul,
    Div,
    Pow
};

BPtr
arith(const Blk& a, const Blk& b, BinOp op)
{
    Dtype P = dtype::common({ a.dtype(), b.dtype() });
    if (op == BinOp::Div && (P == Dtype::Bool || P == Dtype::Int64))
        P = Dtype::Float64;
    if (op == BinOp::Pow && P == Dtype::Bool)
        P = Dtype::Int64;
    if (op == BinOp::Sub && P == Dtype::Bool)
        throw std::invalid_argument("subtraction of boolean blocks is not supported");
    CBPtr ca = as_dtype(a, P);
    CBPtr cb = as_dtype(b, P);
    BPtr res;
    visit_dtype(P, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (std::is_same_v<T, bool>) {
            if (op == BinOp::Add)
                res = map_binary<T, T>(*ca, *cb, P, [](bool x, bool y) { return x || y; });
            else
                res = map_binary<T, T>(*ca, *cb, P, [](bool x, bool y) { return x && y; });
        } else {
            switch (op) {
                case BinOp::Add:
                    res = map_binary<T, T>(*ca, *cb, P, [](T x, T y) { return T(x + y); });
                    break;
                case BinOp::Sub:
                    res = map_binary<T, T>(*ca, *cb, P, [](T x, T y) { return T(x - y); });
                    break;
                case BinOp::Mul:
                    res = map_binary<T, T>(*ca, *cb, P, [](T x, T y) { return T(x * y); });
                    break;
                case BinOp::Div:
                    res = map_binary<T, T>(*ca, *cb, P, [](T x, T y) { return T(x / y); });
                    break;
                case BinOp::Pow:
                    res =
                      map_binary<T, T>(*ca, *cb, P, [](T x, T y) { return pow_elem<T>(x, y); });
                    break;
            }
        }
    });
    return res;
}

enum class CmpOp
{
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne
};

BPtr
compare(const Blk& a, const Blk& b, CmpOp op)
{
    Dtype P = dtype::common({ a.dtype(), b.dtype() });
    CBPtr ca = as_dtype(a, P);
    CBPtr cb = as_dtype(b, P);
    BPtr res;
    visit_dtype(P, [&](auto tag) {
        using T = decltype(tag);
        res = map_binary<T, bool>(*ca, *cb, Dtype::Bool, [op](T x, T y) -> bool {
            switch (op) {
                case CmpOp::Lt:
                    return less_than(x, y);
                case CmpOp::Le:
                    return less_than(x, y) || x == y;
                case CmpOp::Gt:
                    return less_than(y, x);
                case CmpOp::Ge:
                    return less_than(y, x) || x == y;
                case CmpOp::Eq:
                    return x == y;
                case CmpOp::Ne:
                    return x != y;
            }
            return false;
        });
    });
    return res;
}

/// ``f(x)`` for every entry, with ints/bools promoted to float64 first.
template<class F>
BPtr
float_unary(const Blk& a, F&& f)
{
    Dtype d = float_dtype(a.dtype());
    CBPtr ca = as_dtype(a, d);
    BPtr res;
    visit_inexact(d, [&](auto tag) {
        using T = decltype(tag);
        res = map_unary<T, T>(*ca, d, [&](T x) { return f(x); });
    });
    return res;
}

// -----------------------------------------------------------------------------
// permutation / gathering / broadcasting
// -----------------------------------------------------------------------------

std::vector<int64>
validate_permutation(const std::vector<int64>& perm, int64 nd)
{
    if (static_cast<int64>(perm.size()) != nd)
        throw std::invalid_argument("permutation must have one entry per axis");
    std::vector<int64> out(perm.size());
    std::vector<bool> seen(perm.size(), false);
    for (size_t i = 0; i < perm.size(); ++i) {
        out[i] = wrap_axis(perm[i], nd);
        if (seen[out[i]])
            throw std::invalid_argument("repeated axis in permutation");
        seen[out[i]] = true;
    }
    return out;
}

BPtr
permute_copy(const Blk& a, const std::vector<int64>& perm)
{
    const Shape in_shape = a.shape();
    const Shape in_st = c_strides(in_shape);
    Shape out_shape(perm.size());
    Shape st(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        out_shape[i] = in_shape[perm[i]];
        st[i] = in_st[perm[i]];
    }
    BPtr res = make(a.dtype(), out_shape);
    visit_dtype(a.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* src = a.data<T>();
        T* dst = res->data<T>();
        strided_loop<1>(out_shape,
                        std::array<Shape, 1>{ st },
                        [&](int64 i, const std::array<int64, 1>& o) { dst[i] = src[o[0]]; });
    });
    return res;
}

/// Permuted block; aliases ``a`` for the identity permutation.
CBPtr
permuted(const Blk& a, const std::vector<int64>& perm)
{
    bool identity = true;
    for (size_t i = 0; i < perm.size(); ++i)
        identity = identity && perm[i] == static_cast<int64>(i);
    if (identity)
        return CBPtr(CBPtr{}, &a);
    return permute_copy(a, perm);
}

/// ``a`` with the entries ``idx`` along ``axis`` (like ``np.take``).
BPtr
take_axis(const Blk& a, const std::vector<int64>& idx, int64 axis)
{
    Shape shape = a.shape();
    axis = wrap_axis(axis, static_cast<int64>(shape.size()));
    const AxisSplit sp = split_axis(shape, axis);
    const int64 K = static_cast<int64>(idx.size());
    std::vector<int64> widx(idx.size());
    for (size_t k = 0; k < idx.size(); ++k)
        widx[k] = to_valid_idx(idx[k], sp.n);
    shape[axis] = K;
    BPtr res = make(a.dtype(), shape);
    const size_t isz = dtype::itemsize(a.dtype());
    const std::byte* src = a.bytes().data();
    std::byte* dst = res->bytes().data();
    for (int64 o = 0; o < sp.outer; ++o)
        for (int64 k = 0; k < K; ++k)
            if (sp.inner > 0)
                std::memcpy(dst + (o * K + k) * sp.inner * isz,
                            src + (o * sp.n + widx[k]) * sp.inner * isz,
                            sp.inner * isz);
    return res;
}

/// ``a`` broadcast to ``target``, materialized; aliases ``a`` if the shapes agree.
CBPtr
broadcast_to(const Blk& a, const Shape& target)
{
    if (a.shape() == target)
        return CBPtr(CBPtr{}, &a);
    Shape vs = a.shape();
    while (vs.size() > target.size() && vs.front() == 1)
        vs.erase(vs.begin());
    const Shape st = broadcast_strides(vs, target);
    BPtr res = make(a.dtype(), target);
    visit_dtype(a.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* src = a.data<T>();
        T* dst = res->data<T>();
        strided_loop<1>(target,
                        std::array<Shape, 1>{ st },
                        [&](int64 i, const std::array<int64, 1>& o) { dst[i] = src[o[0]]; });
    });
    return res;
}

// -----------------------------------------------------------------------------
// indexing
// -----------------------------------------------------------------------------

struct SliceInfo
{
    int64 start, count, step;
};

SliceInfo
normalize_slice(const BlockBackend::AxisSlice& s, int64 n)
{
    const int64 step = s.step.value_or(1);
    if (step == 0)
        throw std::invalid_argument("slice step cannot be zero");
    const int64 lo = step > 0 ? 0 : -1;
    const int64 hi = step > 0 ? n : n - 1;
    auto norm = [&](std::optional<int64> v, int64 dflt) {
        int64 r = dflt;
        if (v) {
            r = *v;
            if (r < 0)
                r += n;
        }
        return std::clamp(r, lo, hi);
    };
    const int64 start = norm(s.start, step > 0 ? 0 : n - 1);
    const int64 stop = norm(s.stop, step > 0 ? n : -1);
    int64 count = 0;
    if (step > 0)
        count = stop > start ? (stop - start + step - 1) / step : 0;
    else
        count = start > stop ? (start - stop - step - 1) / (-step) : 0;
    return { start, count, step };
}

struct IndexPlan
{
    Shape result_shape;
    std::vector<int64> offsets; // source offsets (elements) in C-order of the result
};

struct IdxItem
{
    enum class Kind
    {
        Int,
        Slice,
        Array
    } kind;
    int64 axis = 0;
    int64 value = 0;
    BlockBackend::AxisSlice slice;
    std::vector<int64> arr;
    Shape arr_shape;
};

struct Segment
{
    Shape dims;
    std::vector<int64> table;
};

/// NumPy-style basic + advanced indexing (a copy is made).
IndexPlan
make_index_plan(const Shape& shape, std::span<const BlockIndex> key)
{
    const int64 nd = static_cast<int64>(shape.size());
    std::vector<IdxItem> items;
    int64 axis = 0;
    auto check_axis = [&](int64 ax) {
        if (ax >= nd)
            throw std::out_of_range("too many indices for block of dimension " +
                                    std::to_string(nd));
    };
    for (const BlockIndex& k : key) {
        if (auto const* iv = std::get_if<int64>(&k)) {
            check_axis(axis);
            IdxItem it;
            it.kind = IdxItem::Kind::Int;
            it.axis = axis++;
            it.value = *iv;
            items.push_back(std::move(it));
        } else if (auto const* sl = std::get_if<BlockBackend::AxisSlice>(&k)) {
            check_axis(axis);
            IdxItem it;
            it.kind = IdxItem::Kind::Slice;
            it.axis = axis++;
            it.slice = *sl;
            items.push_back(std::move(it));
        } else if (auto const* vec = std::get_if<std::vector<int64>>(&k)) {
            check_axis(axis);
            IdxItem it;
            it.kind = IdxItem::Kind::Array;
            it.axis = axis++;
            it.arr = *vec;
            it.arr_shape = { static_cast<int64>(vec->size()) };
            items.push_back(std::move(it));
        } else {
            const BlockCPtr& bp = std::get<BlockCPtr>(k);
            if (!bp)
                throw std::invalid_argument("EigenBlockBackend: null BlockCPtr in BlockIndex");
            CBPtr b = coerce(*bp);
            if (b->dtype() == Dtype::Bool) {
                const Shape ms = b->shape();
                const int64 kd = static_cast<int64>(ms.size());
                if (kd == 0)
                    throw std::invalid_argument("0-dimensional boolean index is not supported");
                for (int64 d = 0; d < kd; ++d) {
                    check_axis(axis + d);
                    if (shape[axis + d] != ms[d])
                        throw std::out_of_range("boolean index did not match indexed block");
                }
                std::vector<std::vector<int64>> coords(kd);
                const bool* m = b->data<bool>();
                Shape idx(kd, 0);
                const int64 n = b->numel();
                for (int64 i = 0; i < n; ++i) {
                    if (m[i])
                        for (int64 d = 0; d < kd; ++d)
                            coords[d].push_back(idx[d]);
                    for (int64 d = kd - 1; d >= 0; --d) {
                        if (++idx[d] < ms[d])
                            break;
                        idx[d] = 0;
                    }
                }
                for (int64 d = 0; d < kd; ++d) {
                    IdxItem it;
                    it.kind = IdxItem::Kind::Array;
                    it.axis = axis + d;
                    it.arr_shape = { static_cast<int64>(coords[d].size()) };
                    it.arr = std::move(coords[d]);
                    items.push_back(std::move(it));
                }
                axis += kd;
            } else if (b->dtype() == Dtype::Int64) {
                check_axis(axis);
                IdxItem it;
                it.kind = IdxItem::Kind::Array;
                it.axis = axis++;
                it.arr = int64_vector(*b);
                it.arr_shape = b->shape();
                items.push_back(std::move(it));
            } else {
                throw std::invalid_argument("index blocks must have bool or int64 dtype");
            }
        }
    }
    while (axis < nd) {
        IdxItem it;
        it.kind = IdxItem::Kind::Slice;
        it.axis = axis++;
        items.push_back(std::move(it));
    }

    const Shape st = c_strides(shape);
    int64 base = 0;
    std::vector<Segment> segs;
    auto slice_segment = [&](const IdxItem& it) {
        SliceInfo si = normalize_slice(it.slice, shape[it.axis]);
        Segment seg;
        seg.dims = { si.count };
        seg.table.resize(si.count);
        for (int64 i = 0; i < si.count; ++i)
            seg.table[i] = (si.start + i * si.step) * st[it.axis];
        return seg;
    };

    const bool has_adv = std::any_of(items.begin(), items.end(), [](const IdxItem& it) {
        return it.kind == IdxItem::Kind::Array;
    });
    if (!has_adv) {
        for (const IdxItem& it : items) {
            if (it.kind == IdxItem::Kind::Int)
                base += to_valid_idx(it.value, shape[it.axis]) * st[it.axis];
            else
                segs.push_back(slice_segment(it));
        }
    } else {
        // ints count as advanced indices once arrays are present
        int64 first_adv = -1, last_adv = -1;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].kind != IdxItem::Kind::Slice) {
                if (first_adv < 0)
                    first_adv = static_cast<int64>(i);
                last_adv = static_cast<int64>(i);
            }
        }
        bool adjacent = true;
        for (int64 i = first_adv; i <= last_adv; ++i)
            adjacent = adjacent && items[i].kind != IdxItem::Kind::Slice;
        Shape B;
        for (const IdxItem& it : items)
            if (it.kind == IdxItem::Kind::Array)
                B = broadcast_shapes(B, it.arr_shape);
        std::vector<int64> adv_off(prod(B), 0);
        for (const IdxItem& it : items) {
            if (it.kind == IdxItem::Kind::Int) {
                base += to_valid_idx(it.value, shape[it.axis]) * st[it.axis];
            } else if (it.kind == IdxItem::Kind::Array) {
                std::array<Shape, 1> bst{ broadcast_strides(it.arr_shape, B) };
                const int64 dim = shape[it.axis];
                const int64 stride = st[it.axis];
                strided_loop<1>(B, bst, [&](int64 i, const std::array<int64, 1>& o) {
                    adv_off[i] += to_valid_idx(it.arr[o[0]], dim) * stride;
                });
            }
        }
        Segment adv{ B, std::move(adv_off) };
        if (!adjacent)
            segs.push_back(adv);
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].kind == IdxItem::Kind::Slice) {
                segs.push_back(slice_segment(items[i]));
            } else if (adjacent && static_cast<int64>(i) == first_adv) {
                segs.push_back(adv);
            }
        }
    }

    IndexPlan plan;
    plan.offsets = { base };
    for (const Segment& seg : segs) {
        plan.result_shape.insert(plan.result_shape.end(), seg.dims.begin(), seg.dims.end());
        std::vector<int64> next;
        next.reserve(plan.offsets.size() * seg.table.size());
        for (int64 o : plan.offsets)
            for (int64 t : seg.table)
                next.push_back(o + t);
        plan.offsets.swap(next);
    }
    return plan;
}

// -----------------------------------------------------------------------------
// matrix products
// -----------------------------------------------------------------------------

template<class T>
void
gemm_rowmajor(const T* A, const T* B, T* C, int64 M, int64 K, int64 N)
{
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Mat> c(C, M, N);
    if (M == 0 || N == 0 || K == 0) {
        c.setZero();
        return;
    }
    Eigen::Map<const Mat> a(A, M, K);
    Eigen::Map<const Mat> b(B, K, N);
    c.noalias() = a * b;
}

BPtr
tdot_impl(const Blk& a,
          const Blk& b,
          const std::vector<int64>& idcs_a,
          const std::vector<int64>& idcs_b)
{
    if (idcs_a.size() != idcs_b.size())
        throw std::invalid_argument("tdot: idcs_a and idcs_b must have the same length");
    const int64 nda = a.ndim(), ndb = b.ndim();
    const Shape sa = a.shape(), sb = b.shape();
    std::vector<int64> ia(idcs_a.size()), ib(idcs_b.size());
    std::vector<bool> used_a(nda, false), used_b(ndb, false);
    int64 K = 1;
    for (size_t i = 0; i < ia.size(); ++i) {
        ia[i] = wrap_axis(idcs_a[i], nda);
        ib[i] = wrap_axis(idcs_b[i], ndb);
        if (used_a[ia[i]] || used_b[ib[i]])
            throw std::invalid_argument("tdot: repeated axis");
        used_a[ia[i]] = used_b[ib[i]] = true;
        if (sa[ia[i]] != sb[ib[i]])
            throw std::invalid_argument("tdot: shape mismatch in contracted axes: " +
                                        shape_str(sa) + " vs " + shape_str(sb));
        K *= sa[ia[i]];
    }
    std::vector<int64> perm_a, perm_b;
    Shape out_shape;
    int64 M = 1, N = 1;
    for (int64 i = 0; i < nda; ++i)
        if (!used_a[i]) {
            perm_a.push_back(i);
            out_shape.push_back(sa[i]);
            M *= sa[i];
        }
    perm_a.insert(perm_a.end(), ia.begin(), ia.end());
    perm_b = ib;
    for (int64 i = 0; i < ndb; ++i)
        if (!used_b[i]) {
            perm_b.push_back(i);
            out_shape.push_back(sb[i]);
            N *= sb[i];
        }

    Dtype P = dtype::common({ a.dtype(), b.dtype() });
    const bool is_bool = P == Dtype::Bool;
    if (is_bool)
        P = Dtype::Int64;
    CBPtr ca = as_dtype(a, P);
    CBPtr cb = as_dtype(b, P);
    CBPtr pa = permuted(*ca, perm_a);
    CBPtr pb = permuted(*cb, perm_b);
    BPtr res = make(P, out_shape);
    visit_dtype(P, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (!std::is_same_v<T, bool>)
            gemm_rowmajor<T>(pa->data<T>(), pb->data<T>(), res->data<T>(), M, K, N);
    });
    if (is_bool)
        return cast_copy(*res, Dtype::Bool);
    return res;
}

// -----------------------------------------------------------------------------
// LAPACKE wrappers (row-major)
// -----------------------------------------------------------------------------

template<class T>
struct Lapack;

#define LASYTEN_DEFINE_LAPACK(T, PFX, ORGQR, EVD)                                                 \
    template<>                                                                                    \
    struct Lapack<T>                                                                              \
    {                                                                                             \
        using R = real_t<T>;                                                                      \
        static lapack_int gesdd(char jobz,                                                        \
                                int64 m,                                                          \
                                int64 n,                                                          \
                                T* a,                                                             \
                                int64 lda,                                                        \
                                R* s,                                                             \
                                T* u,                                                             \
                                int64 ldu,                                                        \
                                T* vt,                                                            \
                                int64 ldvt)                                                       \
        {                                                                                         \
            return LAPACKE_##PFX##gesdd(LAPACK_ROW_MAJOR,                                         \
                                        jobz,                                                     \
                                        static_cast<lapack_int>(m),                               \
                                        static_cast<lapack_int>(n),                               \
                                        a,                                                        \
                                        static_cast<lapack_int>(lda),                             \
                                        s,                                                        \
                                        u,                                                        \
                                        static_cast<lapack_int>(ldu),                             \
                                        vt,                                                       \
                                        static_cast<lapack_int>(ldvt));                           \
        }                                                                                         \
        static lapack_int gesvd(int64 m,                                                          \
                                int64 n,                                                          \
                                T* a,                                                             \
                                int64 lda,                                                        \
                                R* s,                                                             \
                                T* u,                                                             \
                                int64 ldu,                                                        \
                                T* vt,                                                            \
                                int64 ldvt,                                                       \
                                R* superb)                                                        \
        {                                                                                         \
            return LAPACKE_##PFX##gesvd(LAPACK_ROW_MAJOR,                                         \
                                        'S',                                                      \
                                        'S',                                                      \
                                        static_cast<lapack_int>(m),                               \
                                        static_cast<lapack_int>(n),                               \
                                        a,                                                        \
                                        static_cast<lapack_int>(lda),                             \
                                        s,                                                        \
                                        u,                                                        \
                                        static_cast<lapack_int>(ldu),                             \
                                        vt,                                                       \
                                        static_cast<lapack_int>(ldvt),                            \
                                        superb);                                                  \
        }                                                                                         \
        static lapack_int geqrf(int64 m, int64 n, T* a, int64 lda, T* tau)                        \
        {                                                                                         \
            return LAPACKE_##PFX##geqrf(LAPACK_ROW_MAJOR,                                         \
                                        static_cast<lapack_int>(m),                               \
                                        static_cast<lapack_int>(n),                               \
                                        a,                                                        \
                                        static_cast<lapack_int>(lda),                             \
                                        tau);                                                     \
        }                                                                                         \
        static lapack_int orgqr(int64 m, int64 n, int64 k, T* a, int64 lda, const T* tau)         \
        {                                                                                         \
            return LAPACKE_##ORGQR(LAPACK_ROW_MAJOR,                                              \
                                   static_cast<lapack_int>(m),                                    \
                                   static_cast<lapack_int>(n),                                    \
                                   static_cast<lapack_int>(k),                                    \
                                   a,                                                             \
                                   static_cast<lapack_int>(lda),                                  \
                                   tau);                                                          \
        }                                                                                         \
        static lapack_int heev(char jobz, int64 n, T* a, int64 lda, R* w)                         \
        {                                                                                         \
            return LAPACKE_##EVD(LAPACK_ROW_MAJOR,                                                \
                                 jobz,                                                            \
                                 'L',                                                             \
                                 static_cast<lapack_int>(n),                                      \
                                 a,                                                               \
                                 static_cast<lapack_int>(lda),                                    \
                                 w);                                                              \
        }                                                                                         \
    };

LASYTEN_DEFINE_LAPACK(float32, s, sorgqr, ssyevd)
LASYTEN_DEFINE_LAPACK(float64, d, dorgqr, dsyevd)
LASYTEN_DEFINE_LAPACK(complex64, c, cungqr, cheevd)
LASYTEN_DEFINE_LAPACK(complex128, z, zungqr, zheevd)
#undef LASYTEN_DEFINE_LAPACK

/// General eigenvalue problem: real input returns (wr, wi), complex input returns w.
template<class T>
struct LapackEig;

template<>
struct LapackEig<float32>
{
    static lapack_int geev(char jobvr, int64 n, float32* a, float32* wr, float32* wi, float32* vr)
    {
        float32 dummy = 0;
        return LAPACKE_sgeev(LAPACK_ROW_MAJOR,
                             'N',
                             jobvr,
                             static_cast<lapack_int>(n),
                             a,
                             static_cast<lapack_int>(n),
                             wr,
                             wi,
                             &dummy,
                             1,
                             vr,
                             jobvr == 'V' ? static_cast<lapack_int>(n) : 1);
    }
};
template<>
struct LapackEig<float64>
{
    static lapack_int geev(char jobvr, int64 n, float64* a, float64* wr, float64* wi, float64* vr)
    {
        float64 dummy = 0;
        return LAPACKE_dgeev(LAPACK_ROW_MAJOR,
                             'N',
                             jobvr,
                             static_cast<lapack_int>(n),
                             a,
                             static_cast<lapack_int>(n),
                             wr,
                             wi,
                             &dummy,
                             1,
                             vr,
                             jobvr == 'V' ? static_cast<lapack_int>(n) : 1);
    }
};
template<>
struct LapackEig<complex64>
{
    static lapack_int geev(char jobvr, int64 n, complex64* a, complex64* w, complex64* vr)
    {
        complex64 dummy{};
        return LAPACKE_cgeev(LAPACK_ROW_MAJOR,
                             'N',
                             jobvr,
                             static_cast<lapack_int>(n),
                             a,
                             static_cast<lapack_int>(n),
                             w,
                             &dummy,
                             1,
                             vr,
                             jobvr == 'V' ? static_cast<lapack_int>(n) : 1);
    }
};
template<>
struct LapackEig<complex128>
{
    static lapack_int geev(char jobvr, int64 n, complex128* a, complex128* w, complex128* vr)
    {
        complex128 dummy{};
        return LAPACKE_zgeev(LAPACK_ROW_MAJOR,
                             'N',
                             jobvr,
                             static_cast<lapack_int>(n),
                             a,
                             static_cast<lapack_int>(n),
                             w,
                             &dummy,
                             1,
                             vr,
                             jobvr == 'V' ? static_cast<lapack_int>(n) : 1);
    }
};

void
check_lapack(lapack_int info, const char* name)
{
    if (info < 0)
        throw std::invalid_argument(
          std::string(name) + ": illegal value (or NaN/inf) in argument " + std::to_string(-info));
    if (info > 0)
        throw std::runtime_error(std::string(name) +
                                 " failed to converge (LAPACK info=" + std::to_string(info) + ")");
}

/// Dtype that linear algebra is carried out in (ints and bools become float64).
Dtype
linalg_dtype(Dtype d)
{
    return float_dtype(d);
}

void
require_ndim(const Blk& a, int64 nd, const char* what)
{
    if (a.ndim() != nd)
        throw std::invalid_argument(std::string(what) + " requires a " + std::to_string(nd) +
                                    "-dimensional block, got shape " + shape_str(a.shape()));
}

void
require_square(const Blk& a, const char* what)
{
    require_ndim(a, 2, what);
    if (a.shape()[0] != a.shape()[1])
        throw std::invalid_argument(std::string(what) + " requires a square matrix, got shape " +
                                    shape_str(a.shape()));
}

template<class T>
std::tuple<BPtr, BPtr, BPtr>
svd_typed(const Blk& a, const std::string& algo)
{
    using R = real_t<T>;
    const int64 m = a.shape()[0], n = a.shape()[1], K = std::min(m, n);
    BPtr U = make(dtype::of<T>(), { m, K });
    BPtr S = make(dtype::of<R>(), { K });
    BPtr Vt = make(dtype::of<T>(), { K, n });
    if (m == 0 || n == 0)
        return { U, S, Vt };
    const T* in = a.data<T>();
    auto run_gesdd = [&]() {
        std::vector<T> w(in, in + m * n);
        return Lapack<T>::gesdd(
          'S', m, n, w.data(), n, S->data<R>(), U->data<T>(), K, Vt->data<T>(), n);
    };
    auto run_gesvd = [&]() {
        std::vector<T> w(in, in + m * n);
        std::vector<R> superb(std::max<int64>(1, K));
        return Lapack<T>::gesvd(
          m, n, w.data(), n, S->data<R>(), U->data<T>(), K, Vt->data<T>(), n, superb.data());
    };
    lapack_int info = 0;
    if (algo == "gesdd") {
        info = run_gesdd();
        check_lapack(info, "gesdd");
    } else if (algo == "gesvd") {
        info = run_gesvd();
        check_lapack(info, "gesvd");
    } else { // robust, robust_silent
        info = run_gesdd();
        if (info > 0) {
            if (algo == "robust")
                warn("SVD with gesdd did not converge; falling back to gesvd");
            info = run_gesvd();
            check_lapack(info, "gesvd");
        } else {
            check_lapack(info, "gesdd");
        }
    }
    return { U, S, Vt };
}

template<class T>
std::tuple<BPtr, BPtr>
qr_typed(const Blk& a, bool full)
{
    const int64 m = a.shape()[0], n = a.shape()[1], K = std::min(m, n);
    const int64 qcols = full ? m : K;
    const int64 rrows = full ? m : K;
    BPtr Q = make(dtype::of<T>(), { m, qcols });
    BPtr R = make(dtype::of<T>(), { rrows, n });
    if (K == 0) {
        if (full) {
            T* q = Q->data<T>();
            for (int64 i = 0; i < m; ++i)
                q[i * m + i] = T(1);
        }
        return { Q, R };
    }
    const T* in = a.data<T>();
    std::vector<T> w(in, in + m * n);
    std::vector<T> tau(K);
    check_lapack(Lapack<T>::geqrf(m, n, w.data(), n, tau.data()), "geqrf");
    T* r = R->data<T>();
    for (int64 i = 0; i < rrows; ++i)
        for (int64 j = i; j < n; ++j)
            r[i * n + j] = w[i * n + j];
    T* q = Q->data<T>();
    for (int64 i = 0; i < m; ++i)
        for (int64 j = 0; j < K; ++j)
            q[i * qcols + j] = w[i * n + j];
    check_lapack(Lapack<T>::orgqr(m, qcols, K, q, qcols, tau.data()), "orgqr");
    return { Q, R };
}

template<class T>
std::tuple<BPtr, BPtr>
eigh_typed(const Blk& a, bool want_vecs)
{
    using R = real_t<T>;
    const int64 n = a.shape()[0];
    BPtr W = make(dtype::of<R>(), { n });
    BPtr V = want_vecs ? make(dtype::of<T>(), { n, n }) : BPtr{};
    if (n == 0)
        return { W, V };
    std::vector<T> w(a.data<T>(), a.data<T>() + n * n);
    check_lapack(Lapack<T>::heev(want_vecs ? 'V' : 'N', n, w.data(), n, W->data<R>()), "heevd");
    if (want_vecs)
        std::memcpy(V->data<T>(), w.data(), sizeof(T) * n * n);
    return { W, V };
}

template<class T>
std::tuple<BPtr, BPtr>
eig_typed(const Blk& a, bool want_vecs)
{
    using R = real_t<T>;
    using C = std::complex<R>;
    const int64 n = a.shape()[0];
    BPtr W = make(dtype::of<C>(), { n });
    BPtr V = want_vecs ? make(dtype::of<C>(), { n, n }) : BPtr{};
    if (n == 0)
        return { W, V };
    std::vector<T> w(a.data<T>(), a.data<T>() + n * n);
    const char jobvr = want_vecs ? 'V' : 'N';
    if constexpr (is_cplx_v<T>) {
        T dummy{};
        check_lapack(
          LapackEig<T>::geev(jobvr, n, w.data(), W->data<C>(), want_vecs ? V->data<C>() : &dummy),
          "geev");
    } else {
        std::vector<R> wr(n), wi(n), vr(want_vecs ? n * n : 1);
        check_lapack(LapackEig<T>::geev(jobvr, n, w.data(), wr.data(), wi.data(), vr.data()),
                     "geev");
        C* pw = W->data<C>();
        for (int64 j = 0; j < n; ++j)
            pw[j] = C(wr[j], wi[j]);
        if (want_vecs) {
            C* pv = V->data<C>();
            int64 j = 0;
            while (j < n) {
                if (wi[j] == R(0) || j + 1 >= n) {
                    for (int64 i = 0; i < n; ++i)
                        pv[i * n + j] = C(vr[i * n + j], R(0));
                    j += 1;
                } else {
                    for (int64 i = 0; i < n; ++i) {
                        pv[i * n + j] = C(vr[i * n + j], vr[i * n + j + 1]);
                        pv[i * n + j + 1] = C(vr[i * n + j], -vr[i * n + j + 1]);
                    }
                    j += 2;
                }
            }
        }
    }
    return { W, V };
}

template<class T>
BPtr
matrix_exp_typed(const Blk& a)
{
    using RMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const Shape shape = a.shape();
    const int64 n = shape.back();
    BPtr res = make(a.dtype(), shape);
    if (n == 0)
        return res;
    const int64 batch = a.numel() / (n * n);
    for (int64 b = 0; b < batch; ++b) {
        Mat m = Eigen::Map<const RMat>(a.data<T>() + b * n * n, n, n);
        Mat e = m.exp();
        Eigen::Map<RMat>(res->data<T>() + b * n * n, n, n) = e;
    }
    return res;
}

// -----------------------------------------------------------------------------
// random numbers
// -----------------------------------------------------------------------------

std::mutex&
rng_mutex()
{
    static std::mutex mu;
    return mu;
}

std::mt19937_64&
rng()
{
    static std::mt19937_64 gen{ std::random_device{}() };
    return gen;
}

// -----------------------------------------------------------------------------
// sorting / misc
// -----------------------------------------------------------------------------

Shape
unravel(int64 flat, const Shape& shape)
{
    Shape idx(shape.size(), 0);
    for (int64 i = static_cast<int64>(shape.size()) - 1; i >= 0; --i) {
        idx[i] = flat % shape[i];
        flat /= shape[i];
    }
    return idx;
}

} // namespace

// -----------------------------------------------------------------------------
// EigenBlockBackend::Block
// -----------------------------------------------------------------------------

EigenBlockBackend::Block::Block(Dtype dtype, std::vector<int64> shape)
  : dtype_(dtype)
  , shape_(std::move(shape))
{
    for (int64 s : shape_)
        if (s < 0)
            throw std::invalid_argument("negative dimension in block shape");
    data_.assign(static_cast<size_t>(numel()) * dtype::itemsize(dtype_), std::byte{ 0 });
}

EigenBlockBackend::Block::Block(Dtype dtype, std::vector<int64> shape, std::vector<std::byte> data)
  : dtype_(dtype)
  , shape_(std::move(shape))
  , data_(std::move(data))
{
    for (int64 s : shape_)
        if (s < 0)
            throw std::invalid_argument("negative dimension in block shape");
    if (data_.size() != static_cast<size_t>(numel()) * dtype::itemsize(dtype_))
        throw std::invalid_argument("EigenBlockBackend::Block: buffer size does not match shape");
}

#if LASYTEN_HAS_PYTHON
EigenBlockBackend::Block::Block(const py::array& arr)
  : dtype_(Dtype::Float64)
{
    try {
        dtype_ = dtype::from_numpy_dtype(arr.dtype());
    } catch (std::invalid_argument&) {
        // other widths: cast to the closest supported dtype
        const char kind = arr.dtype().kind();
        if (kind == 'u' && arr.dtype().itemsize() == 1)
            dtype_ = Dtype::Bool; // legacy HDF5 writers store bool as uint8
        else if (kind == 'b')
            dtype_ = Dtype::Bool;
        else if (kind == 'i' || kind == 'u')
            dtype_ = Dtype::Int64;
        else if (kind == 'f')
            dtype_ = Dtype::Float64;
        else if (kind == 'c')
            dtype_ = Dtype::Complex128;
        else
            throw std::invalid_argument("EigenBlockBackend::Block: invalid numpy dtype: " +
                                        py::cast<std::string>(py::str(arr.dtype())));
    }
    shape_.resize(static_cast<size_t>(arr.ndim()));
    for (py::ssize_t i = 0; i < arr.ndim(); ++i)
        shape_[static_cast<size_t>(i)] = static_cast<int64>(arr.shape(i));
    data_.resize(static_cast<size_t>(numel()) * dtype::itemsize(dtype_));
    visit_dtype(dtype_, [&](auto tag) {
        using T = decltype(tag);
        // C-contiguous copy with forced cast, independent of the layout of ``arr``
        auto c = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arr);
        if (!c)
            throw std::invalid_argument("EigenBlockBackend::Block: could not convert array");
        if (!data_.empty())
            std::memcpy(data_.data(), c.data(), data_.size());
    });
}
#endif // LASYTEN_HAS_PYTHON

int64
EigenBlockBackend::Block::numel() const
{
    return prod(shape_);
}

BlockBackend*
EigenBlockBackend::Block::get_backend() const
{
    return EigenBlockBackend::from_factory("cpu");
}

const std::string&
EigenBlockBackend::Block::device() const
{
    static const std::string device = "cpu";
    return device;
}

#if LASYTEN_HAS_PYTHON
py::array
EigenBlockBackend::Block::to_numpy() const
{
    std::vector<py::ssize_t> shp(shape_.begin(), shape_.end());
    py::array res;
    visit_dtype(dtype_, [&](auto tag) {
        using T = decltype(tag);
        py::array_t<T> arr(shp);
        if (!data_.empty())
            std::memcpy(arr.mutable_data(), data_.data(), data_.size());
        res = std::move(arr);
    });
    return res;
}

py::array
EigenBlockBackend::Block::to_numpy(Dtype dtype) const
{
    return cast_copy(*this, dtype)->to_numpy();
}
#endif // LASYTEN_HAS_PYTHON

BlockCPtr
EigenBlockBackend::Block::get_item(std::span<const BlockIndex> key) const
{
    IndexPlan plan = make_index_plan(shape_, key);
    BPtr res = make(dtype_, plan.result_shape);
    visit_dtype(dtype_, [&](auto tag) {
        using T = decltype(tag);
        const T* src = data<T>();
        T* dst = res->data<T>();
        for (size_t i = 0; i < plan.offsets.size(); ++i)
            dst[i] = src[plan.offsets[i]];
    });
    return res;
}

BlockPtr
EigenBlockBackend::Block::get_item(std::span<const BlockIndex> key)
{
    return std::const_pointer_cast<BlockBackend::Block>(
      static_cast<const Block*>(this)->get_item(key));
}

#if LASYTEN_HAS_PYTHON
namespace {

/// Pointer to the C++ object behind a bound lasyten Python object (nanobind).
/// Raises if \p obj is not an instance of T.
template<class T>
T*
bound_ptr(const py::object& obj)
{
    return ::nanobind::cast<T*>(::nanobind::handle(obj.ptr()));
}

/// Shared owner of the C++ object behind a bound lasyten Python object (nanobind).
template<class T>
std::shared_ptr<T>
bound_shared(const py::object& obj)
{
    return ::nanobind::cast<std::shared_ptr<T>>(::nanobind::handle(obj.ptr()));
}

/// Replace Block entries of a (tuple) key by numpy arrays; for the numpy escape hatch.
py::object
key_blocks_to_numpy(py::object key)
{
    if (py::isinstance<BlockBackend::Block>(key))
        return (*bound_ptr<BlockBackend::Block>(key)).to_numpy();
    if (py::isinstance<py::tuple>(key)) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(key);
        py::tuple out(t.size());
        for (py::ssize_t i = 0; i < t.size(); ++i)
            out[i] = key_blocks_to_numpy(py::reinterpret_borrow<py::object>(t[i]));
        return std::move(out);
    }
    return key;
}

BPtr
block_from_py_array_like(py::object obj)
{
    py::array arr =
      py::reinterpret_borrow<py::array>(py::module_::import("numpy").attr("asarray")(obj));
    return std::make_shared<Blk>(arr);
}

} // namespace

BlockCPtr
EigenBlockBackend::Block::get_item(py::object key) const
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    // Escape hatch for Ellipsis / newaxis / exotic keys: index a numpy copy.
    py::object result = to_numpy().attr("__getitem__")(key_blocks_to_numpy(key));
    return block_from_py_array_like(result);
}

BlockPtr
EigenBlockBackend::Block::get_item(py::object key)
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    py::object result = to_numpy().attr("__getitem__")(key_blocks_to_numpy(key));
    return block_from_py_array_like(result);
}
#endif // LASYTEN_HAS_PYTHON

void
EigenBlockBackend::Block::set_item(std::span<const BlockIndex> key,
                                   const BlockBackend::Block& value)
{
    IndexPlan plan = make_index_plan(shape_, key);
    CBPtr v = coerce(value);
    CBPtr vc = as_dtype(*v, dtype_);
    CBPtr vb = broadcast_to(*vc, plan.result_shape);
    visit_dtype(dtype_, [&](auto tag) {
        using T = decltype(tag);
        T* dst = data<T>();
        const T* src = vb->data<T>();
        for (size_t i = 0; i < plan.offsets.size(); ++i)
            dst[plan.offsets[i]] = src[i];
    });
}

#if LASYTEN_HAS_PYTHON
void
EigenBlockBackend::Block::set_item(py::object key, py::object value)
{
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key)) {
        if (py::isinstance<BlockBackend::Block>(value)) {
            set_item(std::span<const BlockIndex>(*idcs), (*bound_ptr<BlockBackend::Block>(value)));
            return;
        }
        if (py::isinstance<BlockBackend::Scalar>(value)) {
            set_item(std::span<const BlockIndex>(*idcs),
                     (*bound_ptr<BlockBackend::Scalar>(value)));
            return;
        }
        BPtr tmp = block_from_py_array_like(value);
        set_item(std::span<const BlockIndex>(*idcs), *tmp);
        return;
    }
    // Escape hatch: assign into a numpy copy, then copy back.
    py::array arr = to_numpy();
    py::object np_value = value;
    if (py::isinstance<BlockBackend::Block>(value))
        np_value = (*bound_ptr<BlockBackend::Block>(value)).to_numpy();
    else if (py::isinstance<BlockBackend::Scalar>(value))
        np_value = (*bound_ptr<BlockBackend::Scalar>(value)).to_numpy();
    arr.attr("__setitem__")(key_blocks_to_numpy(key), np_value);
    Blk updated(arr);
    data_ = cast_copy(updated, dtype_)->bytes();
}
#endif // LASYTEN_HAS_PYTHON

void
EigenBlockBackend::Block::set_item(const std::vector<int64>& key, const Scalar& value)
{
    std::vector<BlockIndex> idx;
    idx.reserve(key.size());
    for (int64 k : key)
        idx.emplace_back(k);
    set_item(std::span<const BlockIndex>(idx), *value._block());
}

void
EigenBlockBackend::Block::set_item(int64 idx, const Scalar& value)
{
    if (ndim() != 1)
        throw std::invalid_argument(
          "EigenBlockBackend::Block::set_item(int64): block must be 1-dimensional");
    set_item(std::vector<int64>{ idx }, value);
}

complex128
EigenBlockBackend::Block::_item_as_complex128() const
{
    if (numel() != 1)
        throw std::invalid_argument("can only convert a block with exactly one entry to a scalar");
    complex128 res;
    visit_dtype(dtype_, [&](auto tag) {
        using T = decltype(tag);
        res = conv<complex128>(*data<T>());
    });
    return res;
}

int64
EigenBlockBackend::Block::_item_as_int64() const
{
    if (numel() != 1)
        throw std::invalid_argument("can only convert a block with exactly one entry to a scalar");
    if (dtype_ == Dtype::Int64)
        return *data<int64>();
    if (dtype_ == Dtype::Bool)
        return *data<bool>() ? 1 : 0;
    throw std::invalid_argument("block item is not an integer dtype");
}

BlockPtr
EigenBlockBackend::Block::operator+(const BlockBackend::Block& other) const
{
    return arith(*this, *coerce(other), BinOp::Add);
}
BlockPtr
EigenBlockBackend::Block::operator-(const BlockBackend::Block& other) const
{
    return arith(*this, *coerce(other), BinOp::Sub);
}
BlockPtr
EigenBlockBackend::Block::operator*(const BlockBackend::Block& other) const
{
    return arith(*this, *coerce(other), BinOp::Mul);
}
BlockPtr
EigenBlockBackend::Block::operator/(const BlockBackend::Block& other) const
{
    return arith(*this, *coerce(other), BinOp::Div);
}
BlockPtr
EigenBlockBackend::Block::operator<(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Lt);
}
BlockPtr
EigenBlockBackend::Block::operator<=(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Le);
}
BlockPtr
EigenBlockBackend::Block::operator>(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Gt);
}
BlockPtr
EigenBlockBackend::Block::operator>=(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Ge);
}
BlockPtr
EigenBlockBackend::Block::operator==(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Eq);
}
BlockPtr
EigenBlockBackend::Block::operator!=(const BlockBackend::Block& other) const
{
    return compare(*this, *coerce(other), CmpOp::Ne);
}
BlockPtr
EigenBlockBackend::Block::pow(const BlockBackend::Scalar& exponent) const
{
    return arith(*this, *coerce(*exponent._block()), BinOp::Pow);
}
BlockPtr
EigenBlockBackend::Block::pow(const BlockBackend::Block& exponent) const
{
    return arith(*this, *coerce(exponent), BinOp::Pow);
}

namespace {

/// Block dtype -> HDF5 buffer kind (numpy ``dtype.kind`` letter).
char
hdf5_kind(Dtype dtype)
{
    switch (dtype) {
        case Dtype::Bool:
            return 'b';
        case Dtype::Int64:
            return 'i';
        case Dtype::Float32:
        case Dtype::Float64:
            return 'f';
        case Dtype::Complex64:
        case Dtype::Complex128:
            return 'c';
    }
    throw std::logic_error("EigenBlockBackend: unknown dtype");
}

/// HDF5 buffer (kind, itemsize) -> Block dtype, mirroring the numpy-based conversion of
/// ``Block(py::array)``: other widths are cast to the closest supported dtype by the caller.
Dtype
dtype_from_hdf5_kind(char kind, int itemsize)
{
    switch (kind) {
        case 'b':
            return Dtype::Bool;
        case 'u':
            if (itemsize == 1)
                return Dtype::Bool; // legacy HDF5 writers store bool as uint8
            return Dtype::Int64;
        case 'i':
            return Dtype::Int64;
        case 'f':
            return itemsize == 4 ? Dtype::Float32 : Dtype::Float64;
        case 'c':
            return itemsize == 8 ? Dtype::Complex64 : Dtype::Complex128;
    }
    throw std::invalid_argument(
      std::string("EigenBlockBackend::Block: unsupported HDF5 dtype kind '") + kind + "'");
}

/// Read element \p i of an HDF5 buffer with the given kind / itemsize as ``T``.
template<typename T>
T
buffer_entry(const std::uint8_t* p, char kind, int itemsize)
{
    auto load = [&](auto tag) {
        using S = decltype(tag);
        S v;
        std::memcpy(&v, p, sizeof(S));
        return v;
    };
    if constexpr (is_cplx_v<T>) {
        using R = typename T::value_type;
        if (kind == 'c')
            return itemsize == 8 ? T(R(load(complex64{}).real()), R(load(complex64{}).imag()))
                                 : T(R(load(complex128{}).real()), R(load(complex128{}).imag()));
        return T(R(buffer_entry<float64>(p, kind, itemsize)), R(0));
    } else if constexpr (std::is_same_v<T, bool>) {
        return load(std::uint8_t{}) != 0;
    } else {
        if (kind == 'f')
            return itemsize == 4 ? static_cast<T>(load(float32{}))
                                 : static_cast<T>(load(float64{}));
        if (kind == 'b')
            return static_cast<T>(load(std::uint8_t{}) != 0);
        if (kind == 'u') {
            switch (itemsize) {
                case 1:
                    return static_cast<T>(load(std::uint8_t{}));
                case 2:
                    return static_cast<T>(load(std::uint16_t{}));
                case 4:
                    return static_cast<T>(load(std::uint32_t{}));
                default:
                    return static_cast<T>(load(std::uint64_t{}));
            }
        }
        switch (itemsize) { // 'i'
            case 1:
                return static_cast<T>(load(std::int8_t{}));
            case 2:
                return static_cast<T>(load(std::int16_t{}));
            case 4:
                return static_cast<T>(load(std::int32_t{}));
            default:
                return static_cast<T>(load(std::int64_t{}));
        }
    }
}

} // namespace

void
EigenBlockBackend::Block::save_hdf5(lasyten::hdf5::Saver& saver,
                                    HighFive::Group& /*h5gr*/,
                                    const std::string& subpath)
{
    // Python-free: the owning C-ordered buffer is exactly what HDF5 wants.
    tenpy_hdf5::Hdf5Buffer buf;
    buf.kind = hdf5_kind(dtype_);
    buf.itemsize = static_cast<int>(dtype::itemsize(dtype_));
    buf.shape.assign(shape_.begin(), shape_.end());
    const std::size_t nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    if (nbytes > 0)
        std::memcpy(buf.storage.get(), data_.data(), nbytes);
    saver.save_array(subpath + std::string("arr"), buf);
}

std::shared_ptr<EigenBlockBackend::Block>
EigenBlockBackend::Block::from_hdf5(lasyten::hdf5::Loader& loader,
                                    HighFive::Group& h5gr,
                                    std::string const& subpath)
{
    hid_t id = loader.open(subpath + std::string("arr"));
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    const Dtype dt = dtype_from_hdf5_kind(buf.kind, buf.itemsize);
    std::vector<int64> shape(buf.shape.begin(), buf.shape.end());
    auto obj = std::make_shared<EigenBlockBackend::Block>(dt, std::move(shape));
    const auto* src = static_cast<const std::uint8_t*>(buf.data());
    const std::size_t n = static_cast<std::size_t>(obj->numel());
    if (n > 0 && src) {
        visit_dtype(dt, [&](auto tag) {
            using T = decltype(tag);
            T* dst = obj->template data<T>();
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = buffer_entry<T>(
                  src + i * static_cast<std::size_t>(buf.itemsize), buf.kind, buf.itemsize);
        });
    }
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: factory, scalars, devices
// -----------------------------------------------------------------------------

EigenBlockBackend::EigenBlockBackend()
  : BlockBackend("cpu")
{
}

EigenBlockBackend*
EigenBlockBackend::from_factory(const std::string& device)
{
    return from_factory_shared(device).get();
}

std::shared_ptr<EigenBlockBackend>
EigenBlockBackend::from_factory_shared(const std::string& device)
{
    if (device != "cpu" && device != "eigen")
        throw std::invalid_argument(
          "EigenBlockBackend::from_factory only supports device \"cpu\" (or \"eigen\"), got: " +
          device);
    static std::mutex mutex;
    static std::shared_ptr<EigenBlockBackend> instance;
    std::lock_guard<std::mutex> lock(mutex);
    if (!instance)
        instance = std::shared_ptr<EigenBlockBackend>(new EigenBlockBackend());
    return instance;
}

void
EigenBlockBackend::seed(std::uint64_t seed)
{
    std::lock_guard<std::mutex> lock(rng_mutex());
    rng().seed(seed);
}

std::shared_ptr<EigenBlockBackend>
EigenBlockBackend::from_hdf5(lasyten::hdf5::Loader& loader,
                             HighFive::Group& h5gr,
                             std::string const& subpath)
{
    (void)subpath;
    auto obj = EigenBlockBackend::from_factory_shared("cpu");
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

std::string
EigenBlockBackend::get_backend_name() const
{
    return "EigenBlockBackend";
}

bool
EigenBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<const Blk*>(block.get()) != nullptr;
}

std::string
EigenBlockBackend::as_device(std::optional<std::string> device)
{
    if (!device)
        return default_device;
    if (*device == "cpu" || *device == "eigen")
        return default_device;
    if (std::regex_match(*device, std::regex(R"(cpu:\d+)")))
        return default_device;
    throw std::invalid_argument("EigenBlockBackend does not support device " + *device);
}

BlockBackend::Scalar
EigenBlockBackend::as_scalar(complex128 value, Dtype dtype)
{
    BPtr b = make(dtype, {});
    visit_dtype(dtype, [&](auto tag) {
        using T = decltype(tag);
        *b->data<T>() = conv<T>(value);
    });
    return mk_scalar(std::move(b));
}

#if LASYTEN_HAS_PYTHON
BlockBackend::Scalar
EigenBlockBackend::as_scalar(py::object value, Dtype dtype)
{
    if (py::isinstance<BlockBackend::Scalar>(value)) {
        auto scalar = *bound_ptr<BlockBackend::Scalar>(value);
        if (scalar.dtype() == dtype)
            return scalar;
        return mk_scalar(cast_copy(*coerce(*scalar._block()), dtype));
    }
    py::array arr = py::reinterpret_borrow<py::array>(
      py::module_::import("numpy").attr("asarray")(value, dtype::to_numpy_dtype(dtype)));
    BPtr b = std::make_shared<Blk>(arr);
    if (b->dtype() != dtype)
        b = cast_copy(*b, dtype);
    return mk_scalar(std::move(b));
}
#endif // LASYTEN_HAS_PYTHON

BlockBackend::Scalar
EigenBlockBackend::as_scalar(bool b)
{
    return mk_scalar(scalar_block<bool>(b));
}
BlockBackend::Scalar
EigenBlockBackend::as_scalar(int64 x)
{
    return mk_scalar(scalar_block<int64>(x));
}
BlockBackend::Scalar
EigenBlockBackend::as_scalar(float32 x)
{
    return mk_scalar(scalar_block<float32>(x));
}
BlockBackend::Scalar
EigenBlockBackend::as_scalar(float64 x)
{
    return mk_scalar(scalar_block<float64>(x));
}
BlockBackend::Scalar
EigenBlockBackend::as_scalar(complex64 z)
{
    return mk_scalar(scalar_block<complex64>(z));
}
BlockBackend::Scalar
EigenBlockBackend::as_scalar(complex128 z)
{
    return mk_scalar(scalar_block<complex128>(z));
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: conversions and constructors
// -----------------------------------------------------------------------------

#if LASYTEN_HAS_PYTHON
BlockPtr
EigenBlockBackend::as_block(py::object a,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device)
{
    (void)as_device(device);
    BPtr block;
    if (py::isinstance<Blk>(a)) {
        block = bound_shared<Blk>(a);
        if (dtype_opt && *dtype_opt != block->dtype())
            block = cast_copy(*block, *dtype_opt);
        return block;
    }
    if (py::isinstance<BlockBackend::Block>(a)) {
        // foreign backend block: convert via numpy
        block = std::make_shared<Blk>((*bound_ptr<BlockBackend::Block>(a)).to_numpy());
    } else {
        py::object np_dtype = dtype_opt ? dtype::to_numpy_dtype(*dtype_opt) : py::none();
        py::array arr = py::reinterpret_borrow<py::array>(
          py::module_::import("numpy").attr("asarray")(a, np_dtype));
        block = std::make_shared<Blk>(arr);
    }
    if (dtype_opt && *dtype_opt != block->dtype())
        block = cast_copy(*block, *dtype_opt);
    // integer -> float64, like the other backends
    if (block->dtype() == Dtype::Int64)
        block = cast_copy(*block, Dtype::Float64);
    return block;
}

BlockPtr
EigenBlockBackend::block_from_numpy(const py::array& a,
                                    std::optional<Dtype> dtype_opt,
                                    std::optional<std::string> device)
{
    (void)as_device(device);
    BPtr block = std::make_shared<Blk>(a);
    if (dtype_opt && *dtype_opt != block->dtype())
        block = cast_copy(*block, *dtype_opt);
    return block;
}
#endif // LASYTEN_HAS_PYTHON

BlockPtr
EigenBlockBackend::to_dtype(const BlockCPtr& a, Dtype dtype)
{
    return cast_copy(eb_ref(a), dtype);
}

BlockPtr
EigenBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    (void)as_device(device);
    return copy_of(eb_ref(a));
}

BlockPtr
EigenBlockBackend::zeros(const std::vector<int64>& shape,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    (void)as_device(device);
    return make(dtype, shape);
}

BlockPtr
EigenBlockBackend::ones_block(const std::vector<int64>& shape,
                              Dtype dtype,
                              std::optional<std::string> device)
{
    (void)as_device(device);
    BPtr res = make(dtype, shape);
    const int64 n = res->numel();
    visit_dtype(dtype, [&](auto tag) {
        using T = decltype(tag);
        T* p = res->data<T>();
        for (int64 i = 0; i < n; ++i)
            p[i] = conv<T>(1);
    });
    return res;
}

BlockPtr
EigenBlockBackend::eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device)
{
    (void)as_device(device);
    BPtr res = make(dtype, { dim, dim });
    visit_dtype(dtype, [&](auto tag) {
        using T = decltype(tag);
        T* p = res->data<T>();
        for (int64 i = 0; i < dim; ++i)
            p[i * dim + i] = conv<T>(1);
    });
    return res;
}

BlockPtr
EigenBlockBackend::block_from_diagonal(const BlockCPtr& diag)
{
    const Blk& d = eb_ref(diag);
    require_ndim(d, 1, "block_from_diagonal");
    const int64 n = d.shape()[0];
    BPtr res = make(d.dtype(), { n, n });
    visit_dtype(d.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* src = d.data<T>();
        T* dst = res->data<T>();
        for (int64 i = 0; i < n; ++i)
            dst[i * n + i] = src[i];
    });
    return res;
}

BlockPtr
EigenBlockBackend::block_from_mask(const BlockCPtr& mask, Dtype dtype)
{
    const Blk& m = eb_ref(mask);
    require_ndim(m, 1, "block_from_mask");
    if (m.dtype() != Dtype::Bool)
        throw std::invalid_argument("block_from_mask requires a boolean mask");
    const int64 M = m.shape()[0];
    const bool* pm = m.data<bool>();
    int64 N = 0;
    for (int64 i = 0; i < M; ++i)
        N += pm[i] ? 1 : 0;
    BPtr res = make(dtype, { N, M });
    visit_dtype(dtype, [&](auto tag) {
        using T = decltype(tag);
        T* p = res->data<T>();
        int64 k = 0;
        for (int64 i = 0; i < M; ++i)
            if (pm[i])
                p[(k++) * M + i] = conv<T>(1);
    });
    return res;
}

BlockPtr
EigenBlockBackend::random_normal(const std::vector<int64>& dims,
                                 Dtype dtype,
                                 float64 sigma,
                                 std::optional<std::string> device)
{
    (void)as_device(device);
    if (dtype == Dtype::Bool || dtype == Dtype::Int64)
        throw std::invalid_argument("random_normal requires a float or complex dtype");
    // sigma is the standard deviation of the complex number: split it among real and imag part
    if (dtype::is_complex(dtype))
        sigma /= std::sqrt(2.0);
    BPtr res = make(dtype, dims);
    std::lock_guard<std::mutex> lock(rng_mutex());
    std::normal_distribution<float64> dist(0.0, sigma);
    visit_inexact(dtype, [&](auto tag) {
        using T = decltype(tag);
        T* p = res->data<T>();
        const int64 n = res->numel();
        for (int64 i = 0; i < n; ++i) {
            if constexpr (is_cplx_v<T>) {
                float64 re = dist(rng());
                float64 im = dist(rng());
                p[i] = T(static_cast<real_t<T>>(re), static_cast<real_t<T>>(im));
            } else {
                p[i] = static_cast<T>(dist(rng()));
            }
        }
    });
    return res;
}

BlockPtr
EigenBlockBackend::random_uniform(const std::vector<int64>& dims,
                                  Dtype dtype,
                                  std::optional<std::string> device)
{
    (void)as_device(device);
    if (dtype == Dtype::Bool || dtype == Dtype::Int64)
        throw std::invalid_argument("random_uniform requires a float or complex dtype");
    BPtr res = make(dtype, dims);
    std::lock_guard<std::mutex> lock(rng_mutex());
    std::uniform_real_distribution<float64> dist(-1.0, 1.0);
    visit_inexact(dtype, [&](auto tag) {
        using T = decltype(tag);
        T* p = res->data<T>();
        const int64 n = res->numel();
        for (int64 i = 0; i < n; ++i) {
            if constexpr (is_cplx_v<T>) {
                float64 re = dist(rng());
                float64 im = dist(rng());
                p[i] = T(static_cast<real_t<T>>(re), static_cast<real_t<T>>(im));
            } else {
                p[i] = static_cast<T>(dist(rng()));
            }
        }
    });
    return res;
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: shape manipulation
// -----------------------------------------------------------------------------

BlockPtr
EigenBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    const Blk& b = eb_ref(a);
    Shape s = shape;
    int64 unknown = -1;
    int64 known = 1;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == -1) {
            if (unknown >= 0)
                throw std::invalid_argument("can only specify one unknown dimension");
            unknown = static_cast<int64>(i);
        } else {
            known *= s[i];
        }
    }
    if (unknown >= 0) {
        if (known == 0 || b.numel() % known != 0)
            throw std::invalid_argument("can not reshape block of shape " + shape_str(b.shape()) +
                                        " into shape " + shape_str(shape));
        s[unknown] = b.numel() / known;
    }
    return with_shape(b, std::move(s));
}

BlockPtr
EigenBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    const Blk& b = eb_ref(a);
    return permute_copy(b, validate_permutation(permutation, b.ndim()));
}

BlockPtr
EigenBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    const Blk& b = eb_ref(a);
    const int64 nd = b.ndim();
    if (pos < -(nd + 1) || pos > nd)
        throw std::out_of_range("axis " + std::to_string(pos) + " is out of bounds");
    if (pos < 0)
        pos += nd + 1;
    Shape s = b.shape();
    s.insert(s.begin() + pos, 1);
    return with_shape(b, std::move(s));
}

BlockPtr
EigenBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    const Blk& b = eb_ref(a);
    const int64 nd = b.ndim();
    std::vector<bool> drop(nd, false);
    for (int64 ax : idcs) {
        ax = wrap_axis(ax, nd);
        if (b.shape()[ax] != 1)
            throw std::invalid_argument("can not squeeze axis " + std::to_string(ax) +
                                        " with size != 1");
        drop[ax] = true;
    }
    Shape s;
    for (int64 i = 0; i < nd; ++i)
        if (!drop[i])
            s.push_back(b.shape()[i]);
    return with_shape(b, std::move(s));
}

BlockPtr
EigenBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    const Blk& b = eb_ref(a);
    if (repeats < 0)
        throw std::invalid_argument("repeats must not be negative");
    Shape s = b.shape();
    if (s.empty())
        s = { 1 };
    const int64 n = s.back();
    const int64 outer = n == 0 ? 0 : b.numel() / n;
    Shape out_shape = s;
    out_shape.back() = n * repeats;
    BPtr res = make(b.dtype(), out_shape);
    const size_t isz = dtype::itemsize(b.dtype());
    for (int64 o = 0; o < outer; ++o)
        for (int64 r = 0; r < repeats; ++r)
            if (n > 0)
                std::memcpy(res->bytes().data() + (o * n * repeats + r * n) * isz,
                            b.bytes().data() + o * n * isz,
                            n * isz);
    return res;
}

BlockPtr
EigenBlockBackend::apply_axis_permutations(const BlockCPtr& block,
                                           const std::vector<std::vector<int64>>& perms)
{
    const Blk& b = eb_ref(block);
    if (static_cast<int64>(perms.size()) != b.ndim())
        throw std::invalid_argument("apply_leg_permutations: need one permutation per axis");
    BPtr cur = copy_of(b);
    for (size_t ax = 0; ax < perms.size(); ++ax)
        cur = take_axis(*cur, perms[ax], static_cast<int64>(ax));
    return cur;
}

#if LASYTEN_HAS_PYTHON
BlockPtr
EigenBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                          const std::vector<py::array_t<int64>>& perms)
{
    std::vector<std::vector<int64>> native;
    native.reserve(perms.size());
    for (auto const& perm : perms) {
        auto p = perm.unchecked<1>();
        std::vector<int64> idx(static_cast<size_t>(p.shape(0)));
        for (py::ssize_t i = 0; i < p.shape(0); ++i)
            idx[static_cast<size_t>(i)] = p(i);
        native.push_back(std::move(idx));
    }
    return apply_axis_permutations(block, native);
}
#endif // LASYTEN_HAS_PYTHON

BlockPtr
EigenBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    const Blk& b = eb_ref(block);
    const Blk& m = eb_ref(mask);
    require_ndim(m, 1, "apply_mask");
    if (m.dtype() != Dtype::Bool)
        throw std::invalid_argument("apply_mask requires a boolean mask");
    ax = wrap_axis(ax, b.ndim());
    if (m.shape()[0] != b.shape()[ax])
        throw std::invalid_argument("apply_mask: mask length does not match the axis");
    std::vector<int64> idx;
    const bool* pm = m.data<bool>();
    for (int64 i = 0; i < m.shape()[0]; ++i)
        if (pm[i])
            idx.push_back(i);
    return take_axis(b, idx, ax);
}

BlockPtr
EigenBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    const Blk& b = eb_ref(block);
    const Blk& m = eb_ref(mask);
    require_ndim(m, 1, "enlarge_leg");
    if (m.dtype() != Dtype::Bool)
        throw std::invalid_argument("enlarge_leg requires a boolean mask");
    axis = wrap_axis(axis, b.ndim());
    const int64 L = m.shape()[0];
    std::vector<int64> pos;
    const bool* pm = m.data<bool>();
    for (int64 i = 0; i < L; ++i)
        if (pm[i])
            pos.push_back(i);
    if (static_cast<int64>(pos.size()) != b.shape()[axis])
        throw std::invalid_argument("enlarge_leg: mask has wrong number of True entries");
    Shape s = b.shape();
    const AxisSplit sp = split_axis(s, axis);
    s[axis] = L;
    BPtr res = make(b.dtype(), s);
    const size_t isz = dtype::itemsize(b.dtype());
    for (int64 o = 0; o < sp.outer; ++o)
        for (int64 k = 0; k < sp.n; ++k)
            if (sp.inner > 0)
                std::memcpy(res->bytes().data() + (o * L + pos[k]) * sp.inner * isz,
                            b.bytes().data() + (o * sp.n + k) * sp.inner * isz,
                            sp.inner * isz);
    return res;
}

BlockPtr
EigenBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
{
    const Blk& b = eb_ref(a);
    require_ndim(b, 2, "get_diagonal");
    const int64 M = b.shape()[0], N = b.shape()[1], K = std::min(M, N);
    BPtr res = make(b.dtype(), { K });
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* src = b.data<T>();
        T* dst = res->data<T>();
        for (int64 i = 0; i < K; ++i)
            dst[i] = src[i * N + i];
        if (tol) {
            for (int64 i = 0; i < M; ++i)
                for (int64 j = 0; j < N; ++j)
                    if (i != j && abs_f64(src[i * N + j]) > *tol)
                        throw std::invalid_argument("Not a diagonal block.");
        }
    });
    return res;
}

BlockBackend::Scalar
EigenBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    const Blk& b = eb_ref(a);
    if (static_cast<int64>(idcs.size()) != b.ndim())
        throw std::invalid_argument("get_block_element: need one index per axis");
    int64 off = 0;
    for (size_t i = 0; i < idcs.size(); ++i)
        off = off * b.shape()[i] + to_valid_idx(idcs[i], b.shape()[i]);
    BPtr res = make(b.dtype(), {});
    const size_t isz = dtype::itemsize(b.dtype());
    std::memcpy(res->bytes().data(), b.bytes().data() + off * isz, isz);
    return mk_scalar(std::move(res));
}

BlockBackend::Scalar
EigenBlockBackend::item(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    if (b.numel() != 1)
        throw std::invalid_argument("item() requires a single-element block");
    return mk_scalar(with_shape(b, {}));
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: elementwise
// -----------------------------------------------------------------------------

BlockPtr
EigenBlockBackend::abs(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    BPtr res;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        if constexpr (std::is_same_v<T, bool>) {
            res = copy_of(b);
        } else if constexpr (std::is_same_v<T, int64>) {
            res = map_unary<T, T>(b, b.dtype(), [](T x) { return x < 0 ? -x : x; });
        } else {
            using R = real_t<T>;
            res =
              map_unary<T, R>(b, dtype::of<R>(), [](T x) { return static_cast<R>(std::abs(x)); });
        }
    });
    return res;
}

BlockPtr
EigenBlockBackend::angle(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    Dtype d = float_dtype(b.dtype());
    CBPtr cb = as_dtype(b, d);
    BPtr res;
    visit_inexact(d, [&](auto tag) {
        using T = decltype(tag);
        using R = real_t<T>;
        res = map_unary<T, R>(*cb, dtype::of<R>(), [](T x) {
            if constexpr (is_cplx_v<T>)
                return static_cast<R>(std::atan2(x.imag(), x.real()));
            else
                return static_cast<R>(std::atan2(R(0), x));
        });
    });
    return res;
}

BlockPtr
EigenBlockBackend::conj(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    if (!dtype::is_complex(b.dtype()))
        return copy_of(b);
    BPtr res;
    visit_inexact(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        if constexpr (is_cplx_v<T>)
            res = map_unary<T, T>(b, b.dtype(), [](T x) { return std::conj(x); });
    });
    return res;
}

BlockPtr
EigenBlockBackend::real(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    if (!dtype::is_complex(b.dtype()))
        return copy_of(b);
    BPtr res;
    visit_inexact(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        if constexpr (is_cplx_v<T>)
            res = map_unary<T, real_t<T>>(b, dtype::of<real_t<T>>(), [](T x) { return x.real(); });
    });
    return res;
}

BlockPtr
EigenBlockBackend::imag(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    if (!dtype::is_complex(b.dtype()))
        return make(b.dtype(), b.shape());
    BPtr res;
    visit_inexact(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        if constexpr (is_cplx_v<T>)
            res = map_unary<T, real_t<T>>(b, dtype::of<real_t<T>>(), [](T x) { return x.imag(); });
    });
    return res;
}

BlockPtr
EigenBlockBackend::real_if_close(const BlockCPtr& a, float64 tol)
{
    const Blk& b = eb_ref(a);
    if (!dtype::is_complex(b.dtype()))
        return copy_of(b);
    // like numpy: tol > 1 is in units of machine epsilon
    if (tol > 1)
        tol *= dtype::eps(dtype::to_real(b.dtype()));
    bool close = true;
    visit_inexact(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        if constexpr (is_cplx_v<T>) {
            const T* p = b.data<T>();
            for (int64 i = 0; i < b.numel() && close; ++i)
                close = std::abs(static_cast<float64>(p[i].imag())) < tol;
        }
    });
    if (close)
        return real(a);
    return copy_of(b);
}

BlockPtr
EigenBlockBackend::exp(const BlockCPtr& a)
{
    return float_unary(eb_ref(a), [](auto x) { return std::exp(x); });
}

BlockPtr
EigenBlockBackend::log(const BlockCPtr& a)
{
    return float_unary(eb_ref(a), [](auto x) { return std::log(x); });
}

BlockPtr
EigenBlockBackend::sqrt(const BlockCPtr& a)
{
    return float_unary(eb_ref(a), [](auto x) { return std::sqrt(x); });
}

BlockPtr
EigenBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    return float_unary(eb_ref(a), [cutoff](auto x) {
        using T = decltype(x);
        if (static_cast<float64>(std::abs(x)) < cutoff)
            return T(0);
        return T(T(1) / x);
    });
}

BlockPtr
EigenBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    return float_unary(eb_ref(block), [cutoff](auto x) {
        using T = decltype(x);
        using R = real_t<T>;
        if (less_than(T(static_cast<R>(cutoff)), x))
            return T(std::log(x));
        return T(0);
    });
}

BlockPtr
EigenBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return arith(eb_ref(a), eb_ref(b), BinOp::Mul);
}

BlockPtr
EigenBlockBackend::mul(const Scalar& a, const BlockCPtr& b)
{
    return arith(*coerce(*a._block()), eb_ref(b), BinOp::Mul);
}

BlockPtr
EigenBlockBackend::linear_combination(const Scalar& a_coef,
                                      const BlockCPtr& v,
                                      const Scalar& b_coef,
                                      const BlockCPtr& w)
{
    BPtr av = arith(*coerce(*a_coef._block()), eb_ref(v), BinOp::Mul);
    BPtr bw = arith(*coerce(*b_coef._block()), eb_ref(w), BinOp::Mul);
    return arith(*av, *bw, BinOp::Add);
}

BlockPtr
EigenBlockBackend::scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis)
{
    const Blk& b = eb_ref(block);
    const Blk& f = eb_ref(factors);
    require_ndim(f, 1, "scale_axis factors");
    axis = wrap_axis(axis, b.ndim());
    Shape fs(b.ndim(), 1);
    fs[axis] = f.shape()[0];
    BPtr fv = with_shape(f, fs);
    return arith(b, *fv, BinOp::Mul);
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: reductions
// -----------------------------------------------------------------------------

bool
EigenBlockBackend::all(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    bool res = true;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        for (int64 i = 0; i < b.numel() && res; ++i)
            res = p[i] != T{};
    });
    return res;
}

bool
EigenBlockBackend::any(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    bool res = false;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        for (int64 i = 0; i < b.numel() && !res; ++i)
            res = p[i] != T{};
    });
    return res;
}

bool
EigenBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    const Blk& ba = eb_ref(a);
    const Blk& bb = eb_ref(b);
    const Shape out = broadcast_shapes(ba.shape(), bb.shape());
    const bool is_c = dtype::is_complex(ba.dtype()) || dtype::is_complex(bb.dtype());
    const Dtype P = is_c ? Dtype::Complex128 : Dtype::Float64;
    CBPtr ca = as_dtype(ba, P);
    CBPtr cb = as_dtype(bb, P);
    std::array<Shape, 2> st{ broadcast_strides(ba.shape(), out),
                             broadcast_strides(bb.shape(), out) };
    bool ok = true;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* pa = ca->data<T>();
        const T* pb = cb->data<T>();
        strided_loop<2>(out, st, [&](int64, const std::array<int64, 2>& o) {
            const T x = pa[o[0]];
            const T y = pb[o[1]];
            const float64 ay = static_cast<float64>(std::abs(y));
            if (std::isfinite(ay))
                ok = ok && static_cast<float64>(std::abs(x - y)) <= atol + rtol * ay;
            else
                ok = ok && x == y;
        });
    };
    if (is_c)
        run(complex128{});
    else
        run(float64{});
    return ok;
}

namespace {

template<class T>
struct SumType
{
    using type = T;
};
template<>
struct SumType<bool>
{
    using type = int64;
};

} // namespace

BlockPtr
EigenBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    const Blk& b = eb_ref(a);
    ax = wrap_axis(ax, b.ndim());
    const AxisSplit sp = split_axis(b.shape(), ax);
    Shape out_shape = b.shape();
    out_shape.erase(out_shape.begin() + ax);
    BPtr res;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        using S = typename SumType<T>::type;
        res = make(dtype::of<S>(), out_shape);
        const T* src = b.data<T>();
        S* dst = res->data<S>();
        for (int64 o = 0; o < sp.outer; ++o)
            for (int64 k = 0; k < sp.n; ++k)
                for (int64 i = 0; i < sp.inner; ++i)
                    dst[o * sp.inner + i] += static_cast<S>(src[(o * sp.n + k) * sp.inner + i]);
    });
    return res;
}

BlockBackend::Scalar
EigenBlockBackend::sum_all(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    BPtr res;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        using S = typename SumType<T>::type;
        S acc{};
        const T* p = b.data<T>();
        for (int64 i = 0; i < b.numel(); ++i)
            acc += static_cast<S>(p[i]);
        res = scalar_block<S>(acc);
    });
    return mk_scalar(std::move(res));
}

namespace {

template<bool IsMax>
BlockBackend::Scalar
extremum(const Blk& b)
{
    if (b.numel() == 0)
        throw std::invalid_argument("min/max of an empty block is undefined");
    BPtr res = make(b.dtype(), {});
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        T best = p[0];
        for (int64 i = 1; i < b.numel(); ++i) {
            if constexpr (IsMax) {
                if (less_than(best, p[i]))
                    best = p[i];
            } else {
                if (less_than(p[i], best))
                    best = p[i];
            }
        }
        *res->data<T>() = best;
    });
    return mk_scalar(std::move(res));
}

} // namespace

BlockBackend::Scalar
EigenBlockBackend::max(const BlockCPtr& a)
{
    return extremum<true>(eb_ref(a));
}

BlockBackend::Scalar
EigenBlockBackend::min(const BlockCPtr& a)
{
    return extremum<false>(eb_ref(a));
}

BlockBackend::Scalar
EigenBlockBackend::max_abs(const BlockCPtr& a)
{
    BlockPtr abs_a = abs(a);
    return extremum<true>(eb_ref(abs_a));
}

std::vector<int64>
EigenBlockBackend::abs_argmax(const BlockCPtr& block)
{
    const Blk& b = eb_ref(block);
    if (b.numel() == 0)
        throw std::invalid_argument("argmax of an empty block is undefined");
    int64 best = 0;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        float64 best_val = abs_f64(p[0]);
        for (int64 i = 1; i < b.numel(); ++i) {
            float64 v = abs_f64(p[i]);
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }
    });
    return unravel(best, b.shape());
}

std::vector<int64>
EigenBlockBackend::argmin(const BlockCPtr& block)
{
    const Blk& b = eb_ref(block);
    if (b.numel() == 0)
        throw std::invalid_argument("argmin of an empty block is undefined");
    int64 best = 0;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        for (int64 i = 1; i < b.numel(); ++i)
            if (less_than(p[i], p[best]))
                best = i;
    });
    return unravel(best, b.shape());
}

BlockPtr
EigenBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    const Blk& b = eb_ref(block);
    axis = wrap_axis(axis, b.ndim());
    const AxisSplit sp = split_axis(b.shape(), axis);
    BPtr res = make(Dtype::Int64, b.shape());
    int64* out = res->data<int64>();
    std::vector<int64> line(sp.n);
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        for (int64 o = 0; o < sp.outer; ++o)
            for (int64 i = 0; i < sp.inner; ++i) {
                std::iota(line.begin(), line.end(), int64(0));
                auto at = [&](int64 k) { return p[(o * sp.n + k) * sp.inner + i]; };
                std::stable_sort(line.begin(), line.end(), [&](int64 x, int64 y) {
                    return less_than(at(x), at(y));
                });
                for (int64 k = 0; k < sp.n; ++k)
                    out[(o * sp.n + k) * sp.inner + i] = line[k];
            }
    });
    return res;
}

BlockBackend::Scalar
EigenBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    const Blk& b = eb_ref(a);
    Shape s = b.shape();
    int64 ax = 0;
    AxisSplit sp{ 1, b.numel(), 1 };
    Shape out_shape;
    if (axis) {
        ax = wrap_axis(*axis, b.ndim());
        sp = split_axis(s, ax);
        out_shape = s;
        out_shape.erase(out_shape.begin() + ax);
    }
    const Dtype rd = real_dtype(b.dtype());
    BPtr res = make(rd, out_shape);
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T* p = b.data<T>();
        visit_inexact(rd, [&](auto rtag) {
            using R = decltype(rtag);
            R* out = res->data<R>();
            for (int64 o = 0; o < sp.outer; ++o)
                for (int64 i = 0; i < sp.inner; ++i) {
                    auto at = [&](int64 k) { return abs_f64(p[(o * sp.n + k) * sp.inner + i]); };
                    float64 acc = 0;
                    if (std::isinf(order)) {
                        acc = order > 0 ? 0.0 : std::numeric_limits<float64>::infinity();
                        for (int64 k = 0; k < sp.n; ++k)
                            acc = order > 0 ? std::max(acc, at(k)) : std::min(acc, at(k));
                    } else if (order == 0.0) {
                        for (int64 k = 0; k < sp.n; ++k)
                            acc += at(k) != 0.0 ? 1.0 : 0.0;
                    } else if (order == 1.0) {
                        for (int64 k = 0; k < sp.n; ++k)
                            acc += at(k);
                    } else if (order == 2.0) {
                        for (int64 k = 0; k < sp.n; ++k)
                            acc += at(k) * at(k);
                        acc = std::sqrt(acc);
                    } else {
                        for (int64 k = 0; k < sp.n; ++k)
                            acc += std::pow(at(k), order);
                        acc = std::pow(acc, 1.0 / order);
                    }
                    out[o * sp.inner + i] = static_cast<R>(acc);
                }
        });
    });
    if (res->numel() != 1)
        throw std::invalid_argument(
          "norm with an axis only supports blocks for which the result is a single value");
    return mk_scalar(with_shape(*res, {}));
}

BlockBackend::Scalar
EigenBlockBackend::inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger)
{
    const Blk& ba = eb_ref(a);
    const Blk& bb = eb_ref(b);
    if (ba.ndim() != bb.ndim())
        throw std::invalid_argument("a and b must have the same number of dimensions");
    Dtype P = dtype::common({ ba.dtype(), bb.dtype() });
    if (P == Dtype::Bool)
        P = Dtype::Int64;
    CBPtr ca = as_dtype(ba, P);
    CBPtr cb = as_dtype(bb, P);
    if (!do_dagger) {
        std::vector<int64> rev(ba.ndim());
        for (int64 i = 0; i < ba.ndim(); ++i)
            rev[i] = ba.ndim() - 1 - i;
        cb = permuted(*cb, rev);
    }
    if (ca->shape() != cb->shape())
        throw std::invalid_argument("inner: shape mismatch " + shape_str(ca->shape()) + " vs " +
                                    shape_str(cb->shape()));
    BPtr res;
    visit_dtype(P, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (!std::is_same_v<T, bool>) {
            const T* pa = ca->data<T>();
            const T* pb = cb->data<T>();
            T acc{};
            for (int64 i = 0; i < ca->numel(); ++i) {
                if constexpr (is_cplx_v<T>)
                    acc += (do_dagger ? std::conj(pa[i]) : pa[i]) * pb[i];
                else
                    acc += pa[i] * pb[i];
            }
            res = scalar_block<T>(acc);
        }
    });
    return mk_scalar(std::move(res));
}

BlockBackend::Scalar
EigenBlockBackend::trace_full(const BlockCPtr& a)
{
    const Blk& b = eb_ref(a);
    const int64 nd = b.ndim();
    const int64 num_trace = nd / 2;
    int64 trace_dim = 1;
    for (int64 i = 0; i < num_trace; ++i)
        trace_dim *= b.shape()[i];
    std::vector<int64> perm(nd);
    for (int64 i = 0; i < num_trace; ++i)
        perm[i] = i;
    for (int64 i = 0; i < num_trace; ++i)
        perm[num_trace + i] = 2 * num_trace - 1 - i;
    for (int64 i = 2 * num_trace; i < nd; ++i)
        perm[i] = i;
    CBPtr p = permuted(b, perm);
    BPtr res;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        using S = typename SumType<T>::type;
        const T* src = p->data<T>();
        S acc{};
        for (int64 i = 0; i < trace_dim; ++i)
            acc += static_cast<S>(src[i * trace_dim + i]);
        res = scalar_block<S>(acc);
    });
    return mk_scalar(std::move(res));
}

BlockPtr
EigenBlockBackend::trace_partial(const BlockCPtr& a,
                                 const std::vector<int64>& idcs1,
                                 const std::vector<int64>& idcs2,
                                 const std::vector<int64>& remaining_idcs)
{
    const Blk& b = eb_ref(a);
    const int64 nd = b.ndim();
    std::vector<int64> perm;
    for (int64 i : remaining_idcs)
        perm.push_back(wrap_axis(i, nd));
    int64 trace_dim = 1;
    for (int64 i : idcs1) {
        perm.push_back(wrap_axis(i, nd));
        trace_dim *= b.shape()[perm.back()];
    }
    for (int64 i : idcs2)
        perm.push_back(wrap_axis(i, nd));
    perm = validate_permutation(perm, nd);
    CBPtr p = permuted(b, perm);
    Shape out_shape;
    for (size_t i = 0; i < remaining_idcs.size(); ++i)
        out_shape.push_back(b.shape()[perm[i]]);
    const int64 batch = prod(out_shape);
    BPtr res;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        using S = typename SumType<T>::type;
        res = make(dtype::of<S>(), out_shape);
        const T* src = p->data<T>();
        S* dst = res->data<S>();
        for (int64 o = 0; o < batch; ++o)
            for (int64 i = 0; i < trace_dim; ++i)
                dst[o] += static_cast<S>(src[(o * trace_dim + i) * trace_dim + i]);
    });
    return res;
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: products
// -----------------------------------------------------------------------------

BlockPtr
EigenBlockBackend::tdot(const BlockCPtr& a,
                        const BlockCPtr& b,
                        const std::vector<int64>& idcs_a,
                        const std::vector<int64>& idcs_b)
{
    return tdot_impl(eb_ref(a), eb_ref(b), idcs_a, idcs_b);
}

BlockPtr
EigenBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    const Blk& ba = eb_ref(a);
    const Blk& bb = eb_ref(b);
    if (ba.ndim() == 0 || bb.ndim() == 0)
        return arith(ba, bb, BinOp::Mul);
    // as in numpy.dot: sum over the last axis of a and the second-to-last axis of b
    const int64 ib = bb.ndim() == 1 ? 0 : bb.ndim() - 2;
    return tdot_impl(ba, bb, { ba.ndim() - 1 }, { ib });
}

BlockPtr
EigenBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    return tdot_impl(eb_ref(a), eb_ref(b), {}, {});
}

BlockPtr
EigenBlockBackend::kron(const BlockCPtr& a, const BlockCPtr& b)
{
    const Blk& ba = eb_ref(a);
    const Blk& bb = eb_ref(b);
    const Shape shape_a = ba.shape();
    const Shape shape_b = bb.shape();
    const size_t nd = std::max(shape_a.size(), shape_b.size());
    Shape sa(nd - shape_a.size(), 1), sb(nd - shape_b.size(), 1);
    sa.insert(sa.end(), shape_a.begin(), shape_a.end());
    sb.insert(sb.end(), shape_b.begin(), shape_b.end());
    Shape out_shape(nd);
    for (size_t i = 0; i < nd; ++i)
        out_shape[i] = sa[i] * sb[i];
    const Shape out_st = c_strides(out_shape);
    Shape st_a(nd), st_b(nd);
    for (size_t i = 0; i < nd; ++i) {
        st_a[i] = sb[i] * out_st[i];
        st_b[i] = out_st[i];
    }
    std::vector<int64> off_a(prod(sa)), off_b(prod(sb));
    strided_loop<1>(sa, std::array<Shape, 1>{ st_a }, [&](int64 i, const std::array<int64, 1>& o) {
        off_a[i] = o[0];
    });
    strided_loop<1>(sb, std::array<Shape, 1>{ st_b }, [&](int64 i, const std::array<int64, 1>& o) {
        off_b[i] = o[0];
    });
    Dtype P = dtype::common({ ba.dtype(), bb.dtype() });
    CBPtr ca = as_dtype(ba, P);
    CBPtr cb = as_dtype(bb, P);
    BPtr res = make(P, out_shape);
    visit_dtype(P, [&](auto tag) {
        using T = decltype(tag);
        const T* pa = ca->data<T>();
        const T* pb = cb->data<T>();
        T* po = res->data<T>();
        for (size_t i = 0; i < off_a.size(); ++i)
            for (size_t j = 0; j < off_b.size(); ++j)
                po[off_a[i] + off_b[j]] = T(pa[i] * pb[j]);
    });
    return res;
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: linear algebra
// -----------------------------------------------------------------------------

BlockPtr
EigenBlockBackend::matrix_exp(const BlockCPtr& matrix)
{
    const Blk& b = eb_ref(matrix);
    if (b.ndim() < 2 || b.shape()[b.ndim() - 1] != b.shape()[b.ndim() - 2])
        throw std::invalid_argument("matrix_exp requires square matrices, got shape " +
                                    shape_str(b.shape()));
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    BPtr res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = matrix_exp_typed<T>(*cb);
    });
    return res;
}

std::tuple<BlockPtr, BlockPtr>
EigenBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    const Blk& b = eb_ref(a);
    require_ndim(b, 2, "matrix_qr");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = qr_typed<T>(*cb, full);
    });
    return { std::get<0>(res), std::get<1>(res) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
EigenBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
{
    const std::string algo = algorithm ? *algorithm : "gesdd";
    const auto& possible = possible_svd_algorithms();
    if (std::find(possible.begin(), possible.end(), algo) == possible.end())
        throw std::invalid_argument("SVD algorithm not supported: " + algo);
    const Blk& b = eb_ref(a);
    require_ndim(b, 2, "matrix_svd");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = svd_typed<T>(*cb, algo);
    });
    return { std::get<0>(res), std::get<1>(res), std::get<2>(res) };
}

const std::vector<std::string>&
EigenBlockBackend::possible_svd_algorithms() const
{
    // LAPACKE: gesdd (divide and conquer, default), gesvd (QR iteration), and fallbacks
    static const std::vector<std::string> algorithms = {
        "gesdd", "gesvd", "robust", "robust_silent"
    };
    return algorithms;
}

std::tuple<BlockPtr, BlockPtr>
EigenBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
{
    const Blk& b = eb_ref(block);
    require_square(b, "eigh");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = eigh_typed<T>(*cb, true);
    });
    BPtr w = std::get<0>(res);
    BPtr v = std::get<1>(res);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        std::vector<int64> p = int64_vector(eb_ref(perm));
        w = take_axis(*w, p, 0);
        v = take_axis(*v, p, 1);
    }
    return { w, v };
}

BlockPtr
EigenBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    const Blk& b = eb_ref(block);
    require_square(b, "eigvalsh");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = eigh_typed<T>(*cb, false);
    });
    BPtr w = std::get<0>(res);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        w = take_axis(*w, int64_vector(eb_ref(perm)), 0);
    }
    return w;
}

std::tuple<BlockPtr, BlockPtr>
EigenBlockBackend::eig(const BlockCPtr& block, std::optional<std::string> sort)
{
    const Blk& b = eb_ref(block);
    require_square(b, "eig");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = eig_typed<T>(*cb, true);
    });
    BPtr w = std::get<0>(res);
    BPtr v = std::get<1>(res);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        std::vector<int64> p = int64_vector(eb_ref(perm));
        w = take_axis(*w, p, 0);
        v = take_axis(*v, p, 1);
    }
    return { w, v };
}

BlockPtr
EigenBlockBackend::eigvals(const BlockCPtr& block, std::optional<std::string> sort)
{
    const Blk& b = eb_ref(block);
    require_square(b, "eigvals");
    CBPtr cb = as_dtype(b, linalg_dtype(b.dtype()));
    std::tuple<BPtr, BPtr> res;
    visit_inexact(cb->dtype(), [&](auto tag) {
        using T = decltype(tag);
        res = eig_typed<T>(*cb, false);
    });
    BPtr w = std::get<0>(res);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        w = take_axis(*w, int64_vector(eb_ref(perm)), 0);
    }
    return w;
}

// -----------------------------------------------------------------------------
// EigenBlockBackend: repr
// -----------------------------------------------------------------------------

#if !LASYTEN_HAS_PYTHON
namespace {

/// Minimal NumPy-like nested-bracket formatting (used when numpy is not available).
std::string
native_entry_to_string(const Blk& b, int64 flat)
{
    std::ostringstream os;
    visit_dtype(b.dtype(), [&](auto tag) {
        using T = decltype(tag);
        const T v = b.data<T>()[flat];
        if constexpr (std::is_same_v<T, bool>)
            os << (v ? "True" : "False");
        else if constexpr (is_cplx_v<T>)
            os << '(' << std::setprecision(8) << v.real() << (v.imag() < 0 ? "-" : "+")
               << std::setprecision(8) << std::abs(v.imag()) << "j)";
        else
            os << std::setprecision(8) << v;
    });
    return os.str();
}

void
native_array_rec(const Blk& b, size_t axis, int64& flat, std::string& out)
{
    const auto shape = b.shape();
    if (axis == shape.size()) {
        out += native_entry_to_string(b, flat++);
        return;
    }
    out += '[';
    for (int64 i = 0; i < shape[axis]; ++i) {
        if (i > 0) {
            out += (axis + 1 == shape.size()) ? ", " : ",\n" + std::string(axis + 1, ' ');
        }
        native_array_rec(b, axis + 1, flat, out);
    }
    out += ']';
}

std::string
native_array_to_string(const Blk& b)
{
    std::string out;
    int64 flat = 0;
    native_array_rec(b, 0, flat, out);
    return out;
}

} // namespace
#endif // !LASYTEN_HAS_PYTHON

std::vector<std::string>
EigenBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                     const std::string& indent,
                                     int64 max_width,
                                     int64 max_lines)
{
    std::string text;
#if LASYTEN_HAS_PYTHON
    py::module_ np = py::module_::import("numpy");
    py::str s = np.attr("array2string")(eb_ref(a).to_numpy(),
                                        py::arg("max_line_width") =
                                          max_width - static_cast<int64>(indent.size()));
    text = s;
#else
    (void)max_width;
    text = native_array_to_string(eb_ref(a));
#endif
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (true) {
        std::size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(indent + text.substr(start));
            break;
        }
        lines.push_back(indent + text.substr(start, end - start));
        start = end + 1;
    }
    if (static_cast<int64>(lines.size()) > max_lines) {
        int64 first = (max_lines - 1) / 2;
        int64 last = max_lines - 1 - first;
        std::vector<std::string> trimmed(lines.begin(), lines.begin() + first);
        trimmed.push_back(indent + "...");
        trimmed.insert(trimmed.end(), lines.end() - last, lines.end());
        return trimmed;
    }
    return lines;
}

} // namespace lasyten
