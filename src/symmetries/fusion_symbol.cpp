#include <lasyten/symmetries/fusion_symbol.h>

#include <lasyten/block_backend/eigen.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <numeric>

namespace lasyten {

namespace {

FusionSymbol::Shape
pad_shape(std::uint8_t rank, FusionSymbol::Shape shape)
{
    for (std::uint8_t i = rank; i < 4; ++i) {
        shape[i] = 1;
    }
    return shape;
}

} // namespace

Dtype
FusionSymbol::check_symbol_dtype(Dtype dtype)
{
    if (dtype != Dtype::Float64 && dtype != Dtype::Complex128) {
        throw std::invalid_argument(std::format(
          "FusionSymbol dtype must be Float64 or Complex128, got {}", dtype::repr(dtype)));
    }
    return dtype;
}

FusionSymbol::FusionSymbol(std::uint8_t rank, Shape shape, Dtype dtype)
  : dtype_(check_symbol_dtype(dtype))
  , rank_(rank)
  , shape_(pad_shape(rank, shape))
{
    ensure_size();
    validate();
}

std::size_t
FusionSymbol::product(Shape const& shape)
{
    return shape[0] * shape[1] * shape[2] * shape[3];
}

std::size_t
FusionSymbol::extent(std::uint8_t axis) const
{
    if (axis >= rank_) {
        throw std::out_of_range("FusionSymbol::extent: axis out of range");
    }
    return shape_[axis];
}

std::size_t
FusionSymbol::size() const noexcept
{
    return product(shape_);
}

std::vector<int64>
FusionSymbol::shape_as_int64() const
{
    std::vector<int64> out;
    out.reserve(rank_);
    for (std::uint8_t i = 0; i < rank_; ++i) {
        out.push_back(static_cast<int64>(shape_[i]));
    }
    return out;
}

void
FusionSymbol::ensure_size()
{
    auto const n = size();
    if (dtype_ == Dtype::Float64) {
        data_ = std::vector<float64>(n, 0.0);
    } else {
        data_ = std::vector<complex128>(n, complex128{ 0.0, 0.0 });
    }
}

void
FusionSymbol::validate() const
{
    if (rank_ < 1 || rank_ > 4) {
        throw std::invalid_argument("FusionSymbol rank must be in 1..4");
    }
    for (std::uint8_t i = rank_; i < 4; ++i) {
        if (shape_[i] != 1) {
            throw std::invalid_argument("FusionSymbol unused axes must have extent 1");
        }
    }
    auto const n = product(shape_);
    if (dtype_ == Dtype::Float64) {
        if (!std::holds_alternative<std::vector<float64>>(data_) ||
            std::get<std::vector<float64>>(data_).size() != n) {
            throw std::logic_error("FusionSymbol float data size mismatch");
        }
    } else if (!std::holds_alternative<std::vector<complex128>>(data_) ||
               std::get<std::vector<complex128>>(data_).size() != n) {
        throw std::logic_error("FusionSymbol complex data size mismatch");
    }
}

std::size_t
FusionSymbol::offset(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) const
{
    if (i0 >= shape_[0] || i1 >= shape_[1] || i2 >= shape_[2] || i3 >= shape_[3]) {
        throw std::out_of_range("FusionSymbol index out of range");
    }
    return ((i0 * shape_[1] + i1) * shape_[2] + i2) * shape_[3] + i3;
}

complex128
FusionSymbol::get_complex(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) const
{
    auto const off = offset(i0, i1, i2, i3);
    if (dtype_ == Dtype::Float64) {
        return complex128{ std::get<std::vector<float64>>(data_)[off], 0.0 };
    }
    return std::get<std::vector<complex128>>(data_)[off];
}

void
FusionSymbol::set(std::size_t i0, complex128 value)
{
    set(i0, 0, 0, 0, value);
}

void
FusionSymbol::set(std::size_t i0, std::size_t i1, complex128 value)
{
    set(i0, i1, 0, 0, value);
}

void
FusionSymbol::set(std::size_t i0, std::size_t i1, std::size_t i2, complex128 value)
{
    set(i0, i1, i2, 0, value);
}

void
FusionSymbol::set(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, complex128 value)
{
    auto const off = offset(i0, i1, i2, i3);
    if (dtype_ == Dtype::Float64) {
        if (value.imag() != 0.0) {
            throw std::invalid_argument("Cannot store complex value in Float64 FusionSymbol");
        }
        std::get<std::vector<float64>>(data_)[off] = value.real();
    } else {
        std::get<std::vector<complex128>>(data_)[off] = value;
    }
}

std::span<float64>
FusionSymbol::as_float64()
{
    if (dtype_ != Dtype::Float64) {
        throw std::invalid_argument("FusionSymbol::as_float64: dtype is not Float64");
    }
    auto& v = std::get<std::vector<float64>>(data_);
    return { v.data(), v.size() };
}

std::span<float64 const>
FusionSymbol::as_float64() const
{
    if (dtype_ != Dtype::Float64) {
        throw std::invalid_argument("FusionSymbol::as_float64: dtype is not Float64");
    }
    auto const& v = std::get<std::vector<float64>>(data_);
    return { v.data(), v.size() };
}

std::span<complex128>
FusionSymbol::as_complex128()
{
    if (dtype_ != Dtype::Complex128) {
        throw std::invalid_argument("FusionSymbol::as_complex128: dtype is not Complex128");
    }
    auto& v = std::get<std::vector<complex128>>(data_);
    return { v.data(), v.size() };
}

std::span<complex128 const>
FusionSymbol::as_complex128() const
{
    if (dtype_ != Dtype::Complex128) {
        throw std::invalid_argument("FusionSymbol::as_complex128: dtype is not Complex128");
    }
    auto const& v = std::get<std::vector<complex128>>(data_);
    return { v.data(), v.size() };
}

FusionSymbol&
FusionSymbol::reshape(std::uint8_t new_rank, Shape new_shape)
{
    new_shape = pad_shape(new_rank, new_shape);
    if (product(new_shape) != size()) {
        throw std::invalid_argument("FusionSymbol::reshape: size mismatch");
    }
    rank_ = new_rank;
    shape_ = new_shape;
    validate();
    return *this;
}

FusionSymbol
FusionSymbol::reshaped(std::uint8_t new_rank, Shape new_shape) const
{
    FusionSymbol out = *this;
    out.reshape(new_rank, new_shape);
    return out;
}

FusionSymbol
FusionSymbol::conj() const
{
    if (dtype_ == Dtype::Float64) {
        return *this;
    }
    FusionSymbol out(rank_, shape_, Dtype::Complex128);
    auto const& src = std::get<std::vector<complex128>>(data_);
    auto& dst = std::get<std::vector<complex128>>(out.data_);
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = std::conj(src[i]);
    }
    return out;
}

FusionSymbol
FusionSymbol::as_complex() const
{
    if (dtype_ == Dtype::Complex128) {
        return *this;
    }
    FusionSymbol out(rank_, shape_, Dtype::Complex128);
    auto const& src = std::get<std::vector<float64>>(data_);
    auto& dst = std::get<std::vector<complex128>>(out.data_);
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = complex128{ src[i], 0.0 };
    }
    return out;
}

FusionSymbol
FusionSymbol::as_dtype(Dtype target) const
{
    target = check_symbol_dtype(target);
    if (target == dtype_) {
        return *this;
    }
    if (target == Dtype::Complex128) {
        return as_complex();
    }
    // Complex → Float64: require vanishing imag
    FusionSymbol out(rank_, shape_, Dtype::Float64);
    auto const& src = std::get<std::vector<complex128>>(data_);
    auto& dst = std::get<std::vector<float64>>(out.data_);
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (src[i].imag() != 0.0) {
            throw std::invalid_argument(
              "Cannot cast complex FusionSymbol with imag != 0 to Float64");
        }
        dst[i] = src[i].real();
    }
    return out;
}

complex128
FusionSymbol::sum() const
{
    complex128 acc{ 0.0, 0.0 };
    if (dtype_ == Dtype::Float64) {
        for (auto v : std::get<std::vector<float64>>(data_)) {
            acc += v;
        }
    } else {
        for (auto v : std::get<std::vector<complex128>>(data_)) {
            acc += v;
        }
    }
    return acc;
}

FusionSymbol
FusionSymbol::operator*(float64 scale) const
{
    FusionSymbol out = *this;
    if (dtype_ == Dtype::Float64) {
        for (auto& v : std::get<std::vector<float64>>(out.data_)) {
            v *= scale;
        }
    } else {
        for (auto& v : std::get<std::vector<complex128>>(out.data_)) {
            v *= scale;
        }
    }
    return out;
}

FusionSymbol
FusionSymbol::operator*(complex128 scale) const
{
    if (scale.imag() == 0.0) {
        return (*this) * scale.real();
    }
    FusionSymbol out = as_complex();
    for (auto& v : std::get<std::vector<complex128>>(out.data_)) {
        v *= scale;
    }
    return out;
}

FusionSymbol
FusionSymbol::multiply(FusionSymbol const& other) const
{
    if (rank_ != other.rank_) {
        throw std::invalid_argument("FusionSymbol::multiply: rank mismatch");
    }
    Shape out_shape{};
    for (std::uint8_t i = 0; i < 4; ++i) {
        std::size_t a = shape_[i];
        std::size_t b = other.shape_[i];
        if (a == b) {
            out_shape[i] = a;
        } else if (a == 1) {
            out_shape[i] = b;
        } else if (b == 1) {
            out_shape[i] = a;
        } else {
            throw std::invalid_argument("FusionSymbol::multiply: cannot broadcast shapes");
        }
    }
    Dtype out_dtype = (dtype_ == Dtype::Complex128 || other.dtype_ == Dtype::Complex128)
                        ? Dtype::Complex128
                        : Dtype::Float64;
    FusionSymbol out(rank_, out_shape, out_dtype);
    for (std::size_t i0 = 0; i0 < out_shape[0]; ++i0) {
        for (std::size_t i1 = 0; i1 < out_shape[1]; ++i1) {
            for (std::size_t i2 = 0; i2 < out_shape[2]; ++i2) {
                for (std::size_t i3 = 0; i3 < out_shape[3]; ++i3) {
                    auto const av = get_complex(shape_[0] == 1 ? 0 : i0,
                                                shape_[1] == 1 ? 0 : i1,
                                                shape_[2] == 1 ? 0 : i2,
                                                shape_[3] == 1 ? 0 : i3);
                    auto const bv = other.get_complex(other.shape_[0] == 1 ? 0 : i0,
                                                      other.shape_[1] == 1 ? 0 : i1,
                                                      other.shape_[2] == 1 ? 0 : i2,
                                                      other.shape_[3] == 1 ? 0 : i3);
                    out.set(i0, i1, i2, i3, av * bv);
                }
            }
        }
    }
    return out;
}

FusionSymbol
FusionSymbol::slice2d(std::size_t i0, std::size_t i1) const
{
    if (rank_ != 4) {
        throw std::invalid_argument("FusionSymbol::slice2d requires rank 4");
    }
    Shape sh{ { shape_[2], shape_[3], 1, 1 } };
    FusionSymbol out(2, sh, dtype_);
    for (std::size_t i2 = 0; i2 < shape_[2]; ++i2) {
        for (std::size_t i3 = 0; i3 < shape_[3]; ++i3) {
            out.set(i2, i3, get_complex(i0, i1, i2, i3));
        }
    }
    return out;
}

FusionSymbol
FusionSymbol::slice2d_trailing(std::size_t i2, std::size_t i3) const
{
    if (rank_ != 4) {
        throw std::invalid_argument("FusionSymbol::slice2d_trailing requires rank 4");
    }
    Shape sh{ { shape_[0], shape_[1], 1, 1 } };
    FusionSymbol out(2, sh, dtype_);
    for (std::size_t i0 = 0; i0 < shape_[0]; ++i0) {
        for (std::size_t i1 = 0; i1 < shape_[1]; ++i1) {
            out.set(i0, i1, get_complex(i0, i1, i2, i3));
        }
    }
    return out;
}

FusionSymbol
FusionSymbol::transpose(std::array<std::uint8_t, 4> const& axes) const
{
    Shape new_shape{};
    for (std::uint8_t i = 0; i < rank_; ++i) {
        if (axes[i] >= rank_) {
            throw std::invalid_argument("FusionSymbol::transpose: bad axis");
        }
        new_shape[i] = shape_[axes[i]];
    }
    for (std::uint8_t i = rank_; i < 4; ++i) {
        new_shape[i] = 1;
    }
    FusionSymbol out(rank_, new_shape, dtype_);
    std::array<std::size_t, 4> idx{};
    std::function<void(std::uint8_t)> rec = [&](std::uint8_t axis) {
        if (axis == rank_) {
            std::array<std::size_t, 4> src{};
            for (std::uint8_t i = 0; i < rank_; ++i) {
                src[axes[i]] = idx[i];
            }
            out.set(idx[0], idx[1], idx[2], idx[3], get_complex(src[0], src[1], src[2], src[3]));
            return;
        }
        for (std::size_t i = 0; i < new_shape[axis]; ++i) {
            idx[axis] = i;
            rec(static_cast<std::uint8_t>(axis + 1));
        }
    };
    rec(0);
    return out;
}

FusionSymbol
FusionSymbol::take_leading_matrix() const
{
    if (rank_ == 2) {
        return *this;
    }
    if (rank_ != 4) {
        throw std::invalid_argument("take_leading_matrix requires rank 2 or 4");
    }
    return slice2d(0, 0);
}

void
FusionSymbol::fill(complex128 value)
{
    auto const n = size();
    if (dtype_ == Dtype::Float64) {
        if (value.imag() != 0.0) {
            throw std::invalid_argument("Cannot fill Float64 FusionSymbol with complex value");
        }
        auto& v = std::get<std::vector<float64>>(data_);
        std::fill(v.begin(), v.end(), value.real());
    } else {
        auto& v = std::get<std::vector<complex128>>(data_);
        std::fill(v.begin(), v.end(), value);
    }
    (void)n;
}

void
FusionSymbol::for_each2d(std::function<void(std::size_t, std::size_t, complex128)> const& fn) const
{
    if (rank_ != 2) {
        throw std::invalid_argument("for_each2d requires rank 2");
    }
    for (std::size_t i = 0; i < shape_[0]; ++i) {
        for (std::size_t j = 0; j < shape_[1]; ++j) {
            fn(i, j, get_complex(i, j));
        }
    }
}

FusionSymbol
FusionSymbol::zeros(std::uint8_t rank, Shape shape, Dtype dtype)
{
    return FusionSymbol(rank, shape, dtype);
}

FusionSymbol
FusionSymbol::ones(std::uint8_t rank, Shape shape, Dtype dtype)
{
    FusionSymbol out(rank, shape, dtype);
    out.fill(complex128{ 1.0, 0.0 });
    return out;
}

FusionSymbol
FusionSymbol::full(std::uint8_t rank, Shape shape, complex128 value, Dtype dtype)
{
    FusionSymbol out(rank, shape, dtype);
    out.fill(value);
    return out;
}

FusionSymbol
FusionSymbol::from_float64(std::uint8_t rank, Shape shape, std::vector<float64> data)
{
    shape = pad_shape(rank, shape);
    if (data.size() != product(shape)) {
        throw std::invalid_argument("from_float64: size mismatch");
    }
    FusionSymbol out;
    out.dtype_ = Dtype::Float64;
    out.rank_ = rank;
    out.shape_ = shape;
    out.data_ = std::move(data);
    out.validate();
    return out;
}

FusionSymbol
FusionSymbol::from_complex128(std::uint8_t rank, Shape shape, std::vector<complex128> data)
{
    shape = pad_shape(rank, shape);
    if (data.size() != product(shape)) {
        throw std::invalid_argument("from_complex128: size mismatch");
    }
    FusionSymbol out;
    out.dtype_ = Dtype::Complex128;
    out.rank_ = rank;
    out.shape_ = shape;
    out.data_ = std::move(data);
    out.validate();
    return out;
}

FusionSymbol
FusionSymbol::scalar1d(complex128 value, Dtype dtype)
{
    dtype = check_symbol_dtype(dtype);
    FusionSymbol out(1, Shape{ { 1, 1, 1, 1 } }, dtype);
    out.set(0, value);
    return out;
}

FusionSymbol
FusionSymbol::scalar1d(float64 value, Dtype dtype)
{
    return scalar1d(complex128{ value, 0.0 }, dtype);
}

FusionSymbol
FusionSymbol::one_1D()
{
    return ones(1, Shape{ { 1, 1, 1, 1 } }, Dtype::Float64);
}

FusionSymbol
FusionSymbol::one_2D()
{
    return ones(2, Shape{ { 1, 1, 1, 1 } }, Dtype::Float64);
}

FusionSymbol
FusionSymbol::one_4D()
{
    return ones(4, Shape{ { 1, 1, 1, 1 } }, Dtype::Float64);
}

FusionSymbol
kron(FusionSymbol const& a, FusionSymbol const& b)
{
    if (a.rank() != b.rank()) {
        throw std::invalid_argument("kron: rank mismatch");
    }
    auto const r = a.rank();
    FusionSymbol::Shape out_shape{};
    for (std::uint8_t i = 0; i < 4; ++i) {
        out_shape[i] = a.shape()[i] * b.shape()[i];
    }
    Dtype out_dtype = (a.dtype() == Dtype::Complex128 || b.dtype() == Dtype::Complex128)
                        ? Dtype::Complex128
                        : Dtype::Float64;
    FusionSymbol out(r, out_shape, out_dtype);

    // NumPy kron for ND: for each axis, output index = ia * b.extent + ib
    std::array<std::size_t, 4> ia{}, ib{}, io{};
    std::function<void(std::uint8_t)> rec = [&](std::uint8_t axis) {
        if (axis == 4) {
            out.set(io[0],
                    io[1],
                    io[2],
                    io[3],
                    a.get_complex(ia[0], ia[1], ia[2], ia[3]) *
                      b.get_complex(ib[0], ib[1], ib[2], ib[3]));
            return;
        }
        for (std::size_t i = 0; i < a.shape()[axis]; ++i) {
            ia[axis] = i;
            for (std::size_t j = 0; j < b.shape()[axis]; ++j) {
                ib[axis] = j;
                io[axis] = i * b.shape()[axis] + j;
                rec(static_cast<std::uint8_t>(axis + 1));
            }
        }
    };
    rec(0);
    return out;
}

#if LASYTEN_HAS_PYTHON
py::array
fusion_symbol_to_numpy(FusionSymbol const& src)
{
    auto shape = src.shape_as_int64();
    std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    if (src.dtype() == Dtype::Float64) {
        py::array_t<float64> arr(py_shape);
        auto buf = arr.request();
        auto* ptr = static_cast<float64*>(buf.ptr);
        auto span = src.as_float64();
        std::memcpy(ptr, span.data(), span.size() * sizeof(float64));
        return arr;
    }
    py::array_t<complex128> arr(py_shape);
    auto buf = arr.request();
    auto* ptr = static_cast<complex128*>(buf.ptr);
    auto span = src.as_complex128();
    std::memcpy(ptr, span.data(), span.size() * sizeof(complex128));
    return arr;
}

FusionSymbol
fusion_symbol_from_numpy(py::array arr)
{
    arr = py::array::ensure(arr);
    if (!arr) {
        throw std::invalid_argument("fusion_symbol_from_numpy: expected array");
    }
    auto info = arr.request();
    if (info.ndim < 1 || info.ndim > 4) {
        throw std::invalid_argument("fusion_symbol_from_numpy: rank must be 1..4");
    }
    auto const rank = static_cast<std::uint8_t>(info.ndim);
    FusionSymbol::Shape shape{ { 1, 1, 1, 1 } };
    for (std::uint8_t i = 0; i < rank; ++i) {
        shape[i] = static_cast<std::size_t>(info.shape[i]);
    }

    // Promote ints / float32 etc. used by historical ones arrays, then forcecast into a
    // contiguous native buffer (no numpy.asarray).
    Dtype dt = dtype::from_numpy_dtype(arr.attr("dtype"));
    if (dt != Dtype::Float64 && dt != Dtype::Complex128) {
        if (dtype::is_real(dt) || dt == Dtype::Int64 || dt == Dtype::Bool) {
            dt = Dtype::Float64;
        } else {
            dt = Dtype::Complex128;
        }
    }

    auto const n = FusionSymbol::product(shape);
    if (dt == Dtype::Float64) {
        py::array_t<float64, py::array::c_style | py::array::forcecast> casted(arr);
        auto cinfo = casted.request();
        auto const* ptr = static_cast<float64 const*>(cinfo.ptr);
        return FusionSymbol::from_float64(rank, shape, std::vector<float64>(ptr, ptr + n));
    }
    py::array_t<complex128, py::array::c_style | py::array::forcecast> casted(arr);
    auto cinfo = casted.request();
    auto const* ptr = static_cast<complex128 const*>(cinfo.ptr);
    return FusionSymbol::from_complex128(rank, shape, std::vector<complex128>(ptr, ptr + n));
}
#endif // LASYTEN_HAS_PYTHON

BlockBackend::BlockPtr
block_from_fusion_symbol(BlockBackend& backend,
                         FusionSymbol const& arr,
                         std::optional<Dtype> dtype,
                         std::optional<std::string> device)
{
    if (auto* eigen = dynamic_cast<EigenBlockBackend*>(&backend)) {
        // Python-free path: copy the symbol's (C-ordered) data into an Eigen block.
        (void)eigen->as_device(device);
        auto const shape = arr.shape_as_int64();
        std::vector<std::byte> bytes;
        if (arr.dtype() == Dtype::Float64) {
            auto const span = arr.as_float64();
            bytes.resize(span.size() * sizeof(float64));
            std::memcpy(bytes.data(), span.data(), bytes.size());
        } else {
            auto const span = arr.as_complex128();
            bytes.resize(span.size() * sizeof(complex128));
            std::memcpy(bytes.data(), span.data(), bytes.size());
        }
        BlockBackend::BlockPtr block =
          std::make_shared<EigenBlockBackend::Block>(arr.dtype(), shape, std::move(bytes));
        if (dtype && *dtype != block->dtype()) {
            block = eigen->to_dtype(block, *dtype);
        }
        return block;
    }
#if LASYTEN_HAS_PYTHON
    return backend.block_from_numpy(fusion_symbol_to_numpy(arr), dtype, device);
#else
    throw NotImplemented("block_from_fusion_symbol for non-Eigen block backends needs Python");
#endif
}

FusionSymbol
fusion_symbol_from_block(BlockBackend::BlockCPtr const& block)
{
    if (!block) {
        throw std::invalid_argument("fusion_symbol_from_block: null block");
    }
    auto shape = block->shape();
    if (shape.size() < 1 || shape.size() > 4) {
        throw std::invalid_argument("fusion_symbol_from_block: block ndim must be 1..4");
    }
    if (auto const* eb = dynamic_cast<EigenBlockBackend::Block const*>(block.get())) {
        // Python-free path. Promote ints / bools / float32 like the numpy path does.
        auto const rank = static_cast<std::uint8_t>(shape.size());
        FusionSymbol::Shape sym_shape{ { 1, 1, 1, 1 } };
        for (std::uint8_t i = 0; i < rank; ++i) {
            sym_shape[i] = static_cast<std::size_t>(shape[i]);
        }
        auto const n = FusionSymbol::product(sym_shape);
        auto* backend = eb->get_backend();
        if (dtype::is_complex(eb->dtype())) {
            auto const conv = backend->to_dtype(block, Dtype::Complex128);
            auto const* p = static_cast<EigenBlockBackend::Block const&>(*conv).data<complex128>();
            return FusionSymbol::from_complex128(
              rank, sym_shape, std::vector<complex128>(p, p + n));
        }
        auto const conv = backend->to_dtype(block, Dtype::Float64);
        auto const* p = static_cast<EigenBlockBackend::Block const&>(*conv).data<float64>();
        return FusionSymbol::from_float64(rank, sym_shape, std::vector<float64>(p, p + n));
    }
#if LASYTEN_HAS_PYTHON
    return fusion_symbol_from_numpy(block->to_numpy());
#else
    throw NotImplemented("fusion_symbol_from_block for non-Eigen blocks needs Python");
#endif
}

} // namespace lasyten
