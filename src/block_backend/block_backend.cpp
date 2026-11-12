#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/block_backend/numpy.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cstring>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace lasyten {

namespace {

bool
is_py_integer_scalar_key(const py::handle& obj)
{
    if (obj.is_none())
        return false;
    if (py::isinstance<py::array>(obj))
        return false;
    if (py::isinstance<py::slice>(obj))
        return false;
    try {
        (void)obj.cast<int64>();
        return true;
    } catch (const py::cast_error&) {
        return false;
    }
}

std::optional<BlockBackend::BlockIndex>
try_py_handle_to_block_index(py::handle key)
{
    if (py::isinstance<BlockBackend::Block>(key)) {
        return BlockBackend::BlockCPtr(key.cast<BlockPtr>());
    }
    if (py::isinstance<py::array>(key)) {
        py::array arr = py::reinterpret_borrow<py::array>(key);
        // Bool masks must stay boolean; wrap as a numpy Block so backends can index natively.
        try {
            if (dtype::from_numpy_dtype(arr.dtype()) == Dtype::Bool) {
                return BlockBackend::BlockCPtr(std::make_shared<NumpyBlockBackend::Block>(arr));
            }
        } catch (std::invalid_argument const&) {
            // Unsupported dtype for lasyten — leave for the py::object escape hatch.
            return std::nullopt;
        }
        py::array_t<int64, py::array::c_style | py::array::forcecast> iarr(arr);
        auto buf = iarr.request();
        auto* ptr = static_cast<int64*>(buf.ptr);
        return std::vector<int64>(ptr, ptr + buf.size);
    }
    // Plain Python list/tuple of integers → host index array (not a multi-axis key).
    if (py::isinstance<py::list>(key) || py::isinstance<py::tuple>(key)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(key);
        std::vector<int64> vec;
        vec.reserve(static_cast<size_t>(seq.size()));
        for (py::ssize_t i = 0; i < seq.size(); ++i) {
            if (!is_py_integer_scalar_key(seq[i]))
                return std::nullopt;
            vec.push_back(seq[i].cast<int64>());
        }
        return vec;
    }
    if (py::isinstance<py::slice>(key)) {
        py::slice sl = py::reinterpret_borrow<py::slice>(key);
        auto as_opt_int = [](py::object o) -> std::optional<int64> {
            if (o.is_none())
                return std::nullopt;
            return o.cast<int64>();
        };
        return BlockBackend::AxisSlice{ as_opt_int(sl.attr("start")),
                                        as_opt_int(sl.attr("stop")),
                                        as_opt_int(sl.attr("step")) };
    }
    if (is_py_integer_scalar_key(key))
        return key.cast<int64>();
    return std::nullopt;
}

} // namespace

py::object
BlockBackend::block_index_to_py(const BlockIndex& idx)
{
    return std::visit(
      [](auto const& v) -> py::object {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, int64>) {
              return py::int_(v);
          } else if constexpr (std::is_same_v<T, AxisSlice>) {
              return py::slice(v.start.has_value() ? py::object(py::int_(*v.start)) : py::none(),
                               v.stop.has_value() ? py::object(py::int_(*v.stop)) : py::none(),
                               v.step.has_value() ? py::object(py::int_(*v.step)) : py::none());
          } else if constexpr (std::is_same_v<T, std::vector<int64>>) {
              py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
              if (!v.empty())
                  std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int64));
              return std::move(arr);
          } else {
              static_assert(std::is_same_v<T, BlockCPtr>);
              return py::object(v->to_numpy());
          }
      },
      idx);
}

py::object
BlockBackend::block_indices_to_py(std::span<const BlockIndex> key)
{
    if (key.size() == 1)
        return block_index_to_py(key[0]);
    py::tuple t(static_cast<py::ssize_t>(key.size()));
    for (size_t i = 0; i < key.size(); ++i)
        t[static_cast<py::ssize_t>(i)] = block_index_to_py(key[i]);
    return t;
}

std::optional<std::vector<BlockBackend::BlockIndex>>
BlockBackend::try_py_key_to_block_indices(py::object key)
{
    if (py::isinstance<py::tuple>(key)) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(key);
        std::vector<BlockIndex> out;
        out.reserve(static_cast<size_t>(t.size()));
        for (auto item : t) {
            auto idx = try_py_handle_to_block_index(item);
            if (!idx)
                return std::nullopt;
            out.push_back(std::move(*idx));
        }
        return out;
    }
    auto single = try_py_handle_to_block_index(key);
    if (!single)
        return std::nullopt;
    return std::vector<BlockIndex>{ std::move(*single) };
}

// -----------------------------------------------------------------------------
// Block class
// -----------------------------------------------------------------------------

BlockPtr
BlockBackend::Block::operator*(const Scalar& s) const
{
    return get_backend()->mul(s, shared_from_this());
}

BlockPtr
BlockBackend::Block::operator/(const Scalar& s) const
{
    return get_backend()->mul(s.inverse(), shared_from_this());
}

BlockPtr
BlockBackend::Block::abs() const
{
    return get_backend()->abs(shared_from_this());
}

std::shared_ptr<BlockBackend::Block>
BlockBackend::Block::operator[](py::object key)
{
    return get_item(key);
}
std::shared_ptr<const BlockBackend::Block>
BlockBackend::Block::operator[](py::object key) const
{
    return get_item(key);
}

std::shared_ptr<BlockBackend::Block>
BlockBackend::Block::operator[](std::span<const BlockIndex> key)
{
    return get_item(key);
}

std::shared_ptr<const BlockBackend::Block>
BlockBackend::Block::operator[](std::span<const BlockIndex> key) const
{
    return get_item(key);
}

std::shared_ptr<BlockBackend::Block>
BlockBackend::Block::operator[](std::initializer_list<BlockIndex> key)
{
    return get_item(std::span<const BlockIndex>(key.begin(), key.size()));
}

std::shared_ptr<const BlockBackend::Block>
BlockBackend::Block::operator[](std::initializer_list<BlockIndex> key) const
{
    return get_item(std::span<const BlockIndex>(key.begin(), key.size()));
}

BlockBackend::Scalar
BlockBackend::Block::get_item(const std::vector<int64>& key) const
{
    return get_backend()->get_block_element(shared_from_this(), key);
}

BlockBackend::Scalar
BlockBackend::Block::get_item(int64 idx) const
{
    if (ndim() != 1)
        throw std::invalid_argument("Block::get_item(int64): block must be 1-dimensional");
    return get_item(std::vector<int64>{ idx });
}

void
BlockBackend::Block::set_item(std::span<const BlockIndex> key, const Scalar& value)
{
    set_item(key, *value._block());
}

void
BlockBackend::Block::set_item(const std::vector<int64>& key, const Scalar& value)
{
    std::vector<BlockIndex> idcs;
    idcs.reserve(key.size());
    for (int64 k : key)
        idcs.emplace_back(k);
    set_item(std::span<const BlockIndex>(idcs), value);
}

void
BlockBackend::Block::set_item(int64 idx, const Scalar& value)
{
    if (ndim() != 1)
        throw std::invalid_argument("Block::set_item(int64): block must be 1-dimensional");
    set_item(std::vector<int64>{ idx }, value);
}

BlockBackend::Block&
BlockBackend::Block::operator=(py::object rhs)
{
    BlockIndex all = AxisSlice::all();
    if (py::isinstance<Block>(rhs)) {
        set_item(std::span<const BlockIndex>(&all, 1), rhs.cast<Block&>());
    } else if (py::isinstance<Scalar>(rhs)) {
        set_item(std::span<const BlockIndex>(&all, 1), rhs.cast<Scalar&>());
    } else {
        // Fall back: assign via Python key path with a converted full-slice key.
        set_item(block_indices_to_py(std::span<const BlockIndex>(&all, 1)), rhs);
    }
    return *this;
}

py::array
BlockBackend::Block::to_numpy(Dtype dtype) const
{
    py::array arr = to_numpy();
    py::module_ np = py::module_::import("numpy");
    return py::array(np.attr("asarray")(arr, dtype::to_numpy_dtype(dtype)));
}

int64
BlockBackend::Block::ndim() const
{
    return static_cast<int64>(shape().size());
}

// -----------------------------------------------------------------------------
// Scalar class
// -----------------------------------------------------------------------------

BlockBackend::Scalar::Scalar(std::shared_ptr<Block> block)
{
    if (!block || block->ndim() != 0)
        throw std::invalid_argument(
          "Scalar: block must be non-null and have ndim() == 0 (trivial empty shape).");
    block_ = std::move(block);
}

float64
BlockBackend::Scalar::as_float64() const
{
    if (block_->dtype() == Dtype::Bool)
        throw std::runtime_error("Scalar::as_float64: dtype is Bool");
    if (!dtype::is_real(block_->dtype()))
        throw std::runtime_error("Scalar::as_float64: dtype is complex");
    return block_->_item_as_complex128().real();
}

float32
BlockBackend::Scalar::as_float32() const
{
    if (block_->dtype() != Dtype::Float32)
        throw std::runtime_error("Scalar::as_float32: dtype is not Float32");
    return static_cast<float32>(as_float64());
}

complex64
BlockBackend::Scalar::as_complex64() const
{
    if (block_->dtype() != Dtype::Complex64)
        throw std::runtime_error("Scalar::as_complex64: dtype is not Complex64");
    return static_cast<complex64>(block_->_item_as_complex128());
}
complex128
BlockBackend::Scalar::as_complex128() const
{
    return block_->_item_as_complex128();
}

int64
BlockBackend::Scalar::as_int64() const
{
    if (block_->dtype() != Dtype::Int64)
        throw std::runtime_error("Scalar::as_int64: dtype is not Int64");
    return block_->_item_as_int64();
}

bool
BlockBackend::Scalar::as_bool() const
{
    if (block_->dtype() != Dtype::Bool)
        throw std::runtime_error("Scalar::as_bool: dtype is not Bool");
    complex128 z = block_->_item_as_complex128();
    return z.real() != float64(0) || z.imag() != float64(0);
}

py::object
BlockBackend::Scalar::to_numpy() const
{
    py::object val;
    switch (block_->dtype()) {
        case Dtype::Bool:
            val = py::cast(as_bool());
            break;
        case Dtype::Int64:
            val = py::cast(as_int64());
            break;
        case Dtype::Float32:
        case Dtype::Float64:
            val = py::cast(as_float64());
            break;
        case Dtype::Complex64:
        case Dtype::Complex128:
            val = py::cast(as_complex128());
            break;
        default:
            throw std::runtime_error("Scalar::to_numpy: unknown dtype");
    }
    return dtype::to_numpy_dtype(block_->dtype())(val);
}

BlockBackend::Scalar
BlockBackend::Scalar::operator+(const Scalar& other) const
{
    return Scalar(block_->get_backend()->linear_combination(block_->get_backend()->as_scalar(1.0),
                                                            block_,
                                                            block_->get_backend()->as_scalar(1.0),
                                                            other.block_));
}

BlockBackend::Scalar
BlockBackend::Scalar::operator-() const
{
    return block_->get_backend()->mul(block_->get_backend()->as_scalar(-1.0), block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::operator-(const Scalar& other) const
{
    return Scalar(block_->get_backend()->linear_combination(block_->get_backend()->as_scalar(1.0),
                                                            block_,
                                                            block_->get_backend()->as_scalar(-1.0),
                                                            other.block_));
}

BlockBackend::Scalar
BlockBackend::Scalar::operator*(const Scalar& other) const
{
    return block_->get_backend()->mul(*this, other.block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::operator/(const Scalar& other) const
{
    return *this * other.inverse();
}

BlockBackend::Scalar
BlockBackend::Scalar::inverse() const
{
    if (as_complex128() == complex128(0.0, 0.0)) {
        throw std::runtime_error("Division by zero");
    }
    // TODO: this should actually use block operation instead of converting to C++ types!
    if (dtype::is_complex(block_->dtype())) {
        return block_->get_backend()->as_scalar(1.0 / as_complex128());
    }
    return block_->get_backend()->as_scalar(1.0 / as_float64());
}

BlockCPtr
BlockBackend::Scalar::_block() const
{
    return block_;
}

void
BlockBackend::Scalar::save_hdf5(lasyten::hdf5::Saver& saver,
                                HighFive::Group& /*h5gr*/,
                                const std::string& subpath)
{
    hdf5_export::save_block(saver, subpath + std::string("_block"), block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::from_hdf5(lasyten::hdf5::Loader& loader,
                                HighFive::Group& h5gr,
                                std::string const& subpath)
{
    auto block = hdf5_export::load_block(loader, subpath + std::string("_block"));
    Scalar obj(block);
    // Scalar is not a shared_ptr; memorize the group id with a no-op owning marker via block.
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(block));
    return obj;
}

std::shared_ptr<BlockBackend::Block>
BlockBackend::Block::from_hdf5(lasyten::hdf5::Loader& /*loader*/,
                               HighFive::Group& /*h5gr*/,
                               const std::string& /*subpath*/)
{
    throw NotImplemented(
      "Needs to be implemented in Subclass, since we don't know the subclass type here!");
}

BlockBackend::Scalar
BlockBackend::Scalar::operator<(const Scalar& other) const
{
    return *block_ < *other.block_;
};
BlockBackend::Scalar
BlockBackend::Scalar::operator>(const Scalar& other) const
{
    return *block_ > *other.block_;
};
BlockBackend::Scalar
BlockBackend::Scalar::operator<=(const Scalar& other) const
{
    return *block_ <= *other.block_;
};
BlockBackend::Scalar
BlockBackend::Scalar::operator>=(const Scalar& other) const
{
    return *block_ >= *other.block_;
};

BlockBackend::Scalar
BlockBackend::Scalar::operator+(float64 right) const
{
    return *this + block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator-(float64 right) const
{
    return *this - block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator*(float64 right) const
{
    return *this * block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator/(float64 right) const
{
    return *this / block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator+(complex128 right) const
{
    return *this + block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator-(complex128 right) const
{
    return *this - block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator*(complex128 right) const
{
    return *this * block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator/(complex128 right) const
{
    return *this / block_->get_backend()->as_scalar(right);
};

BlockBackend::Scalar
operator+(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) + right;
};
BlockBackend::Scalar
operator-(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) - right;
};
BlockBackend::Scalar
operator*(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) * right;
};
BlockBackend::Scalar
operator/(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) / right;
};
BlockBackend::Scalar
operator+(complex128 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) + right;
};
BlockBackend::Scalar
operator-(complex128 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) - right;
};
BlockBackend::Scalar
operator*(complex128 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) * right;
};
BlockBackend::Scalar
operator/(complex128 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) / right;
};

BlockBackend::Scalar
BlockBackend::Scalar::operator<(float64 right) const
{
    return *this < block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator>(float64 right) const
{
    return *this > block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator<=(float64 right) const
{
    return *this <= block_->get_backend()->as_scalar(right);
};
BlockBackend::Scalar
BlockBackend::Scalar::operator>=(float64 right) const
{
    return *this >= block_->get_backend()->as_scalar(right);
};

BlockBackend::Scalar
operator<(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) < right;
};
BlockBackend::Scalar
operator>(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) > right;
};
BlockBackend::Scalar
operator<=(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) <= right;
};
BlockBackend::Scalar
operator>=(float64 left, const BlockBackend::Scalar& right)
{
    return right._block()->get_backend()->as_scalar(left) >= right;
};

BlockBackend::Scalar
BlockBackend::Scalar::real() const
{
    return block_->get_backend()->real(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::imag() const
{
    return block_->get_backend()->imag(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::conj() const
{
    return block_->get_backend()->conj(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::abs() const
{
    return block_->get_backend()->abs(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::sqrt() const
{
    return block_->get_backend()->sqrt(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::real_if_close(float64 tol) const
{
    return block_->get_backend()->real_if_close(block_, tol);
}

BlockBackend::Scalar
BlockBackend::Scalar::exp() const
{
    return block_->get_backend()->exp(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::log() const
{
    return block_->get_backend()->log(block_);
}

BlockBackend::Scalar
BlockBackend::Scalar::pow(const Scalar& exponent) const
{
    return block_->pow(exponent);
}

// -----------------------------------------------------------------------------
// BlockBackend class
// -----------------------------------------------------------------------------

BlockBackend::BlockBackend(std::string default_device)
  : default_device(std::move(default_device))
{
}

BlockBackend*
BlockBackend::from_factory(std::string /* device */)
{
    throw NotImplemented(
      "from_factory needs to be called on a subclass (e.g. NumpyBlockBackend::from_factory)");
}

std::string
BlockBackend::get_backend_name() const
{
    return "BlockBackend";
}

bool
BlockBackend::operator==(BlockBackend const& other) const
{
    if (this == &other)
        return true;
    return get_backend_name() == other.get_backend_name() &&
           default_device == other.default_device;
}

std::vector<int64>
BlockBackend::get_shape(const BlockCPtr& a)
{
    return a->shape();
}

Dtype
BlockBackend::get_dtype(const BlockCPtr& a)
{
    return a->dtype();
}

const std::string&
BlockBackend::get_device(const BlockCPtr& a)
{
    return a->device();
}

BlockBackend::Scalar
BlockBackend::as_scalar(const Scalar& value)
{
    return value;
}

BlockPtr
BlockBackend::linear_combination(const Scalar& a_coef,
                                 const BlockCPtr& v,
                                 const Scalar& b_coef,
                                 const BlockCPtr& w)
{
    BlockPtr av = (*v) * a_coef;
    BlockPtr bw = (*w) * b_coef;
    return (*av) + (*bw);
}

BlockPtr
BlockBackend::mul(const Scalar& a, const BlockCPtr& b)
{
    return (*b) * a;
}

BlockPtr
BlockBackend::mul(float64 a, const BlockCPtr& b)
{
    return mul(as_scalar(a), b);
}

BlockPtr
BlockBackend::mul(complex128 a, const BlockCPtr& b)
{
    return mul(as_scalar(a), b);
}

py::object
BlockBackend::to_numpy(const BlockCPtr& a, std::optional<py::object> numpy_dtype)
{
    if (numpy_dtype)
        return py::object(a->to_numpy(dtype::from_numpy_dtype(*numpy_dtype)));
    return py::object(a->to_numpy());
}

BlockPtr
BlockBackend::apply_basis_perm(const BlockCPtr& block, const std::vector<LegCPtr>& legs, bool inv)
{
    std::vector<py::array_t<int64>> perms;
    perms.reserve(legs.size());
    for (LegCPtr const& leg : legs) {
        if (!leg) {
            throw std::invalid_argument("apply_basis_perm: leg must not be None");
        }
        auto const& perm_vec = inv ? leg.inverse_basis_perm() : leg.basis_perm();
        py::array_t<int64> arr(static_cast<py::ssize_t>(perm_vec.size()));
        auto buf = arr.mutable_unchecked<1>();
        for (std::size_t i = 0; i < perm_vec.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = perm_vec[i];
        }
        perms.push_back(std::move(arr));
    }
    return apply_leg_permutations(block, perms);
}

BlockBackend::Scalar
BlockBackend::get_block_mask_element(const BlockCPtr& a,
                                     int64 large_leg_idx,
                                     int64 small_leg_idx,
                                     int64 sum_block)
{
    if (a->dtype() != Dtype::Bool)
        throw std::invalid_argument("a must be a boolean block");
    int64 dim0 = get_shape(a).at(0);
    int64 offset = (large_leg_idx / dim0) * sum_block;
    large_leg_idx %= dim0;
    // if this does not work, need to override.
    if (!a->get_item(large_leg_idx).as_bool())
        // if the block has a False entry, the matrix has only False in that column
        return as_scalar(false);
    // otherwise, there is exactly one True in that column, at index sum(a[:large_leg_idx])
    int64 running = sum_all((*a)[{ AxisSlice{ 0, large_leg_idx, std::nullopt } }]).as_int64();
    return as_scalar(small_leg_idx == offset + running);
}

BlockPtr
BlockBackend::argsort(const BlockCPtr& block, std::optional<std::string> sort, int64 axis)
{
    BlockCPtr work = block;
    if (sort) {
        if (*sort == "m<" || *sort == "SM") {
            work = abs(block);
        } else if (*sort == "m>" || *sort == "LM") {
            work = mul(as_scalar(float64(-1.0)), abs(block));
        } else if (*sort == "<" || *sort == "SR" || *sort == "SA") {
            work = real(block);
        } else if (*sort == ">" || *sort == "LR" || *sort == "LA") {
            work = mul(as_scalar(float64(-1.0)), real(block));
        } else if (*sort == "SI") {
            work = imag(block);
        } else if (*sort == "LI") {
            work = mul(as_scalar(float64(-1.0)), imag(block));
        } else {
            throw std::invalid_argument("Unknown sort option: '" + *sort + "'");
        }
    }
    return _argsort(work, axis);
}

BlockPtr
BlockBackend::combine_legs(const BlockCPtr& a,
                           const std::vector<std::vector<int64>>& leg_idcs_combine,
                           const std::vector<bool>& cstyles_in)
{
    std::vector<bool> cstyles = cstyles_in;
    if (cstyles.size() == 1u)
        cstyles.resize(leg_idcs_combine.size(), cstyles[0]);

    std::vector<int64> const old_shape = get_shape(a);
    size_t const ndim = old_shape.size();
    std::vector<int64> axes_perm(ndim);
    std::iota(axes_perm.begin(), axes_perm.end(), int64(0));

    std::vector<int64> new_shape;
    size_t last_stop = 0;

    for (size_t g = 0; g < leg_idcs_combine.size(); ++g) {
        const std::vector<int64>& group = leg_idcs_combine[g];
        int64 const start = group.front();
        int64 const stop = group.back() + 1;

        if (start < static_cast<int64>(last_stop))
            throw std::invalid_argument("The groups in leg_idcs_combine must not overlap");
        for (size_t i = 0; i < group.size(); ++i)
            if (group[i] != static_cast<int64>(start + i))
                throw std::invalid_argument(
                  "Each group in leg_idcs_combine must be contiguous and ascending");

        for (size_t i = last_stop; i < static_cast<size_t>(start); ++i)
            new_shape.push_back(old_shape[i]);

        int64 combined = 1;
        for (int64 i = start; i < stop; ++i)
            combined *= old_shape[i];
        new_shape.push_back(combined);

        if (!cstyles[g])
            std::reverse(axes_perm.begin() + start, axes_perm.begin() + stop);

        last_stop = static_cast<size_t>(stop);
    }
    for (size_t i = last_stop; i < ndim; ++i)
        new_shape.push_back(old_shape[i]);

    return reshape(permute_axes(a, axes_perm), new_shape);
}

BlockPtr
BlockBackend::combine_legs(const BlockCPtr& a,
                           const std::vector<std::vector<int64>>& leg_idcs_combine,
                           bool cstyles)
{
    return combine_legs(a, leg_idcs_combine, std::vector<bool>(1, cstyles));
}

BlockPtr
BlockBackend::dagger(const BlockCPtr& a)
{
    std::vector<int64> const sh = get_shape(a);
    int64 const num_legs = static_cast<int64>(sh.size());
    std::vector<int64> rev(num_legs);
    for (int64 i = 0; i < num_legs; ++i)
        rev[i] = num_legs - 1 - i;
    return conj(permute_axes(a, rev));
}

bool
BlockBackend::is_real(const BlockCPtr& a)
{
    return dtype::is_real(get_dtype(a));
}

BlockPtr
BlockBackend::permute_combined_matrix(const BlockCPtr& block,
                                      const std::vector<int64>& dims1,
                                      const std::vector<int64>& idcs1,
                                      const std::vector<int64>& dims2,
                                      const std::vector<int64>& idcs2)
{
    std::vector<int64> shape = dims1;
    shape.insert(shape.end(), dims2.begin(), dims2.end());
    BlockPtr b = reshape(block, shape);

    // idcs1 and idcs2 are absolute indices into [0..ndim-1] (same as Python)
    std::vector<int64> perm;
    perm.reserve(idcs1.size() + idcs2.size());
    for (int64 i : idcs1)
        perm.push_back(i);
    for (int64 i : idcs2)
        perm.push_back(i);
    b = permute_axes(b, perm);

    std::vector<int64> const sh = get_shape(b);
    size_t const n1 = idcs1.size();
    int64 M_new = 1;
    for (size_t i = 0; i < n1; ++i)
        M_new *= sh[i];
    int64 N_new = 1;
    for (size_t i = n1; i < sh.size(); ++i)
        N_new *= sh[i];
    return reshape(b, { M_new, N_new });
}

BlockPtr
BlockBackend::permute_combined_idx(const BlockCPtr& block,
                                   int64 axis,
                                   const std::vector<int64>& dims,
                                   const std::vector<int64>& idcs)
{
    std::vector<int64> const sh = get_shape(block);
    if (sh.size() != 2)
        throw std::runtime_error("permute_combined_idx: block must be 2D");
    int64 const M = sh[0];
    int64 const N = sh[1];

    if (axis == -2 || axis == 0) {
        std::vector<int64> new_shape = dims;
        new_shape.push_back(N);
        BlockPtr b = reshape(block, new_shape);
        std::vector<int64> perm;
        for (int64 i : idcs)
            perm.push_back(i);
        perm.push_back(static_cast<int64>(idcs.size()));
        b = permute_axes(b, perm);
        return reshape(b, { M, N });
    }
    if (axis == -1 || axis == 1) {
        std::vector<int64> new_shape = { M };
        new_shape.insert(new_shape.end(), dims.begin(), dims.end());
        BlockPtr b = reshape(block, new_shape);
        std::vector<int64> perm = { 0 };
        for (int64 i : idcs)
            perm.push_back(1 + i);
        b = permute_axes(b, perm);
        return reshape(b, { M, N });
    }
    throw std::invalid_argument("Invalid axis.");
}

BlockPtr
BlockBackend::split_legs(const BlockCPtr& a,
                         const std::vector<int64>& idcs,
                         const std::vector<std::vector<int64>>& dims,
                         const std::vector<bool>& cstyles)
{
    if (idcs.size() != dims.size() || idcs.size() != cstyles.size())
        throw std::invalid_argument("idcs, dims, and cstyles must have the same length");
    std::vector<int64> const old_shape = get_shape(a);
    size_t new_ndim = old_shape.size() - idcs.size();
    for (auto& dim : dims)
        new_ndim += dim.size();
    std::vector<int64> new_shape;
    std::vector<int64> axes_perm;
    new_shape.reserve(new_ndim);
    axes_perm.reserve(new_ndim);
    for (int64 i : idcs)
        if (i < 0 || i >= old_shape.size())
            throw std::invalid_argument("idcs must be within the range of the block shape");
    for (size_t i = 1; i < idcs.size(); ++i)
        if (idcs[i - 1] >= idcs[i])
            throw std::invalid_argument("idcs must be in ascending order and unique");

    size_t start_old_shape = 0;
    for (size_t g = 0; g < idcs.size(); ++g) {
        size_t stop_old_shape = static_cast<size_t>(idcs[g]);
        for (size_t k = start_old_shape; k < stop_old_shape; ++k)
            new_shape.push_back(old_shape[k]);
        for (size_t j = axes_perm.size(); j < new_shape.size(); ++j)
            axes_perm.push_back(static_cast<int64>(j));
        // now insert the split dimensions instead of the old dimensions
        size_t replace_dims = 1;
        for (int64 d : dims[g]) {
            new_shape.push_back(d);
            replace_dims *= d;
        }
        if (old_shape[stop_old_shape] != replace_dims)
            throw std::invalid_argument(
              "The dimensions of the split legs do not match the old dimensions");

        size_t n_axes_now = axes_perm.size();
        size_t n_axes_after = new_shape.size();
        if (cstyles[g]) {
            for (size_t j = n_axes_now; j < n_axes_after; ++j)
                axes_perm.push_back(static_cast<int64>(j));
        } else {
            // reverse the order of the axes
            for (size_t j = n_axes_after; j > n_axes_now; --j)
                axes_perm.push_back(static_cast<int64>(j - 1));
        }
        start_old_shape = stop_old_shape + 1;
    }
    size_t stop_old_shape = old_shape.size();
    for (size_t k = start_old_shape; k < stop_old_shape; ++k)
        new_shape.push_back(old_shape[k]);
    for (size_t j = axes_perm.size(); j < new_shape.size(); ++j)
        axes_perm.push_back(static_cast<int64>(j));

    return permute_axes(reshape(a, new_shape), axes_perm);
}

BlockPtr
BlockBackend::split_legs(const BlockCPtr& a,
                         const std::vector<int64>& idcs,
                         const std::vector<std::vector<int64>>& dims,
                         bool cstyles)
{
    return split_legs(a, idcs, dims, std::vector<bool>(idcs.size(), cstyles));
}

BlockPtr
BlockBackend::tensor_outer(const BlockCPtr& a, const BlockCPtr& b, int64 K)
{
    BlockPtr res = outer(a, b);
    std::vector<int64> const sh_a = get_shape(a);
    std::vector<int64> const sh_b = get_shape(b);
    int64 const N = static_cast<int64>(sh_a.size());
    int64 const M = static_cast<int64>(sh_b.size());

    std::vector<int64> perm;
    for (int64 i = 0; i < K; ++i)
        perm.push_back(i);
    for (int64 i = 0; i < M; ++i)
        perm.push_back(N + i);
    for (int64 i = K; i < N; ++i)
        perm.push_back(i);
    return permute_axes(res, perm);
}

BlockPtr
BlockBackend::eye_block(const std::vector<int64>& legs,
                        Dtype dtype,
                        std::optional<std::string> device)
{
    int64 dim = 1;
    for (int64 d : legs)
        dim *= d;
    BlockPtr eye = eye_matrix(static_cast<int64>(dim), dtype, device);
    std::vector<int64> shape = legs;
    shape.insert(shape.end(), legs.begin(), legs.end());
    eye = reshape(eye, shape);
    int64 const J = static_cast<int64>(legs.size());
    std::vector<int64> perm;
    for (int64 i = 0; i < J; ++i)
        perm.push_back(i);
    for (int64 i = J - 1; i >= 0; --i)
        perm.push_back(J + i);
    return permute_axes(eye, perm);
}

std::tuple<BlockPtr, BlockPtr>
BlockBackend::matrix_lq(const BlockCPtr& a, bool full)
{
    std::vector<int64> perm = { 1, 0 };
    BlockPtr at = permute_axes(a, perm);
    auto [q, r] = matrix_qr(at, full);
    return { permute_axes(r, perm), permute_axes(q, perm) };
}

void
BlockBackend::synchronize()
{
}

void
BlockBackend::test_block_sanity(const BlockCPtr& block,
                                std::optional<std::vector<int64>> expect_shape,
                                std::optional<Dtype> expect_dtype,
                                std::optional<std::string> expect_device)
{
    if (!is_correct_block_type(block)) {
        throw std::runtime_error("wrong block type");
    }
    if (expect_shape) {
        std::vector<int64> const got = get_shape(block);
        if (got != *expect_shape) {
            std::ostringstream msg;
            msg << "wrong block shape ";
            msg << "(";
            for (size_t i = 0; i < got.size(); ++i)
                msg << (i ? ", " : "") << got[i];
            msg << ") != (";
            for (size_t i = 0; i < expect_shape->size(); ++i)
                msg << (i ? ", " : "") << (*expect_shape)[i];
            msg << ")";
            throw std::runtime_error(msg.str());
        }
    }
    if (expect_dtype) {
        if (get_dtype(block) != *expect_dtype) {
            throw std::runtime_error("wrong block dtype");
        }
    }
    if (expect_device) {
        if (get_device(block) != *expect_device) {
            throw std::runtime_error("wrong block device");
        }
    }
}

BlockBackend::Scalar
BlockBackend::inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger)
{
    BlockCPtr ac;
    if (do_dagger) {
        ac = conj(a);
    } else {
        std::vector<int64> const sh = get_shape(a);
        std::vector<int64> rev(sh.size());
        for (size_t i = 0; i < sh.size(); ++i)
            rev[i] = static_cast<int64>(sh.size() - 1 - i);
        ac = permute_axes(a, rev);
    }
    return sum_all(multiply_blocks(ac, b));
}

void
BlockBackend::save_hdf5(lasyten::hdf5::Saver& saver,
                        HighFive::Group& /*h5gr*/,
                        const std::string& subpath)
{
    saver.save_string(subpath + std::string("default_device"), default_device);
}

std::shared_ptr<BlockBackend>
BlockBackend::from_hdf5(lasyten::hdf5::Loader& loader,
                        HighFive::Group& h5gr,
                        std::string const& subpath)
{
    throw NotImplemented(
      "Needs to be implemented in Subclass, since we don't know the subclass type here!");
}

BlockPtr
operator*(const BlockBackend::Scalar& left, const BlockBackend::Block& right)
{
    return right.get_backend()->mul(left, right.shared_from_this());
}

BlockPtr
operator<(const BlockBackend::Block& left, const BlockBackend::Scalar& right)
{
    return left < *right._block();
}
BlockPtr
operator>(const BlockBackend::Block& left, const BlockBackend::Scalar& right)
{
    return left > *right._block();
}
BlockPtr
operator<=(const BlockBackend::Block& left, const BlockBackend::Scalar& right)
{
    return left <= *right._block();
}
BlockPtr
operator>=(const BlockBackend::Block& left, const BlockBackend::Scalar& right)
{
    return left >= *right._block();
}
BlockPtr
operator<(const BlockBackend::Scalar& left, const BlockBackend::Block& right)
{
    return *left._block() < right;
}
BlockPtr
operator>(const BlockBackend::Scalar& left, const BlockBackend::Block& right)
{
    return *left._block() > right;
}
BlockPtr
operator<=(const BlockBackend::Scalar& left, const BlockBackend::Block& right)
{
    return *left._block() <= right;
}
BlockPtr
operator>=(const BlockBackend::Scalar& left, const BlockBackend::Block& right)
{
    return *left._block() >= right;
}
BlockPtr
operator<(const BlockBackend::Block& left, float64 right)
{
    return left < left.get_backend()->as_scalar(right);
}
BlockPtr
operator>(const BlockBackend::Block& left, float64 right)
{
    return left > left.get_backend()->as_scalar(right);
}
BlockPtr
operator<=(const BlockBackend::Block& left, float64 right)
{
    return left <= left.get_backend()->as_scalar(right);
}
BlockPtr
operator>=(const BlockBackend::Block& left, float64 right)
{
    return left >= left.get_backend()->as_scalar(right);
}
BlockPtr
operator<(float64 left, const BlockBackend::Block& right)
{
    return right.get_backend()->as_scalar(left) < right;
}
BlockPtr
operator>(float64 left, const BlockBackend::Block& right)
{
    return right.get_backend()->as_scalar(left) > right;
}
BlockPtr
operator<=(float64 left, const BlockBackend::Block& right)
{
    return right.get_backend()->as_scalar(left) <= right;
}
BlockPtr
operator>=(float64 left, const BlockBackend::Block& right)
{
    return right.get_backend()->as_scalar(left) >= right;
}

std::ostream&
operator<<(std::ostream& os, const BlockBackend::Block& block)
{
    // Metadata header, then the dense array contents (via numpy stringification).
    os << "Block(shape=(";
    const std::vector<int64> shape = block.shape();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            os << ", ";
        os << shape[i];
    }
    if (shape.size() == 1)
        os << ',';
    os << "), dtype=" << dtype::repr(block.dtype()) << ", device=" << block.device() << ")\n";
    os << py::str(block.to_numpy()).cast<std::string>();
    return os;
}

std::ostream&
operator<<(std::ostream& os, const BlockBackend::Scalar& scalar)
{
    os << "Scalar(" << py::str(scalar.to_numpy()).cast<std::string>()
       << ", dtype=" << dtype::repr(scalar.dtype()) << ')';
    return os;
}

} // namespace lasyten
