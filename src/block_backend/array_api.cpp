#include <lasyten/block_backend/array_api.h>
#include <lasyten/detail/python_api.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <span>
#include <stdexcept>
#include <utility>

namespace lasyten {

namespace {

py::object
numpy_module()
{
    return py::module_::import("numpy");
}

py::tuple
to_py_tuple(const std::vector<int64>& v)
{
    py::tuple t(static_cast<py::ssize_t>(v.size()));
    for (size_t i = 0; i < v.size(); ++i)
        t[i] = py::int_(v[i]);
    return t;
}

py::list
to_py_list(const std::vector<int64>& v)
{
    py::list l;
    for (int64 x : v)
        l.append(py::int_(x));
    return l;
}

py::object
py_mul(py::object a, py::object b)
{
    py::object res = a.attr("__mul__")(b);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        res = b.attr("__rmul__")(a);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        throw py::type_error("unsupported operand type(s) for *");
    return res;
}

py::object
py_add(py::object a, py::object b)
{
    py::object res = a.attr("__add__")(b);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        res = b.attr("__radd__")(a);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        throw py::type_error("unsupported operand type(s) for +");
    return res;
}

py::object
py_sub(py::object a, py::object b)
{
    py::object res = a.attr("__sub__")(b);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        res = b.attr("__rsub__")(a);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        throw py::type_error("unsupported operand type(s) for -");
    return res;
}

py::object
py_truediv(py::object a, py::object b)
{
    py::object res = a.attr("__truediv__")(b);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        res = b.attr("__rtruediv__")(a);
    if (res.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)))
        throw py::type_error("unsupported operand type(s) for /");
    return res;
}

py::object
py_lt(py::object a, py::object b)
{
    return a.attr("__lt__")(b);
}

py::object
py_le(py::object a, py::object b)
{
    return a.attr("__le__")(b);
}

py::object
py_gt(py::object a, py::object b)
{
    return a.attr("__gt__")(b);
}

} // namespace

// -----------------------------------------------------------------------------
// ArrayApiBlockBackend::Block
// -----------------------------------------------------------------------------

ArrayApiBlockBackend::Block::Block(py::object arr, ArrayApiBlockBackend* backend)
  : arr_(std::move(arr))
  , backend_(backend)
{
    if (!backend_)
        throw std::invalid_argument("ArrayApiBlockBackend::Block requires a non-null backend");
    // Cache device string without calling as_device() (avoids recursion via ones probes).
    if (py::hasattr(arr_, "device"))
        device_ = py::cast<std::string>(py::str(arr_.attr("device")));
    else
        device_ = backend_->default_device;
}

BlockBackend*
ArrayApiBlockBackend::Block::get_backend() const
{
    return backend_;
}

std::vector<int64>
ArrayApiBlockBackend::Block::shape() const
{
    py::tuple shape = py::to_tuple(arr_.attr("shape"));
    std::vector<int64> out;
    out.reserve(static_cast<size_t>(shape.size()));
    for (auto item : shape) {
        if (item.is_none())
            throw std::runtime_error(
              "Inconsistent block. Unknown dimensions with None in shape not allowed.");
        out.push_back(py::cast<int64>(item));
    }
    return out;
}

Dtype
ArrayApiBlockBackend::Block::dtype() const
{
    return backend_->dtype_from_api(arr_.attr("dtype"));
}

const std::string&
ArrayApiBlockBackend::Block::device() const
{
    return device_;
}

py::array
ArrayApiBlockBackend::Block::to_numpy() const
{
    return py::array(numpy_module().attr("asarray")(arr_));
}

py::array
ArrayApiBlockBackend::Block::to_numpy(Dtype dt) const
{
    return py::array(to_numpy().attr("astype")(dtype::to_numpy_dtype(dt)));
}

BlockCPtr
ArrayApiBlockBackend::Block::get_item(std::span<const BlockIndex> key) const
{
    return backend_->wrap(arr_.attr("__getitem__")(BlockBackend::block_indices_to_py(key)));
}

BlockPtr
ArrayApiBlockBackend::Block::get_item(std::span<const BlockIndex> key)
{
    return backend_->wrap(arr_.attr("__getitem__")(BlockBackend::block_indices_to_py(key)));
}

BlockCPtr
ArrayApiBlockBackend::Block::get_item(py::object key) const
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    return backend_->wrap(arr_.attr("__getitem__")(key));
}

BlockPtr
ArrayApiBlockBackend::Block::get_item(py::object key)
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    return backend_->wrap(arr_.attr("__getitem__")(key));
}

void
ArrayApiBlockBackend::Block::set_item(std::span<const BlockIndex> key,
                                      const BlockBackend::Block& value)
{
    py::object py_key = BlockBackend::block_indices_to_py(key);
    if (auto const* ab = dynamic_cast<ArrayApiBlockBackend::Block const*>(&value)) {
        arr_.attr("__setitem__")(py_key, ab->obj());
        return;
    }
    arr_.attr("__setitem__")(py_key, backend_->api().attr("asarray")(value.to_numpy()));
}

void
ArrayApiBlockBackend::Block::set_item(py::object key, py::object value)
{
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key)) {
        if (py::isinstance<BlockBackend::Block>(value)) {
            set_item(std::span<const BlockIndex>(*idcs), py::cast<BlockBackend::Block&>(value));
            return;
        }
        if (py::isinstance<BlockBackend::Scalar>(value)) {
            set_item(std::span<const BlockIndex>(*idcs), py::cast<BlockBackend::Scalar&>(value));
            return;
        }
        arr_.attr("__setitem__")(BlockBackend::block_indices_to_py(*idcs), value);
        return;
    }
    if (py::isinstance<ArrayApiBlockBackend::Block>(value)) {
        auto* block = py::cast<ArrayApiBlockBackend::Block*>(value);
        arr_.attr("__setitem__")(key, block->obj());
        return;
    }
    if (py::isinstance<BlockBackend::Block>(value)) {
        auto* block = py::cast<BlockBackend::Block*>(value);
        arr_.attr("__setitem__")(key, backend_->api().attr("asarray")(block->to_numpy()));
        return;
    }
    arr_.attr("__setitem__")(key, value);
}

void
ArrayApiBlockBackend::Block::set_item(const std::vector<int64>& key, const Scalar& value)
{
    set_item(to_py_tuple(key), value.to_numpy());
}

void
ArrayApiBlockBackend::Block::set_item(int64 idx, const Scalar& value)
{
    if (ndim() != 1)
        throw std::invalid_argument(
          "ArrayApiBlockBackend::Block::set_item(int64): block must be 1-dimensional");
    set_item(py::int_(idx), value.to_numpy());
}

complex128
ArrayApiBlockBackend::Block::_item_as_complex128() const
{
    py::object val = arr_.attr("item")();
    try {
        return py::cast<complex128>(val);
    } catch (py::cast_error const&) {
        return static_cast<complex128>(py::cast<float64>(val));
    }
}

int64
ArrayApiBlockBackend::Block::_item_as_int64() const
{
    return py::cast<int64>(arr_.attr("item")());
}

namespace {

py::object
other_as_api_obj(ArrayApiBlockBackend* backend, const BlockBackend::Block& other)
{
    if (auto const* o = dynamic_cast<ArrayApiBlockBackend::Block const*>(&other))
        return o->obj();
    return backend->api().attr("asarray")(other.to_numpy());
}

} // namespace

BlockPtr
ArrayApiBlockBackend::Block::operator+(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__add__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator-(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__sub__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator*(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__mul__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator/(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__truediv__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator<(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__lt__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator<=(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__le__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator>(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__gt__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator>=(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__ge__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator==(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__eq__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::operator!=(const BlockBackend::Block& other) const
{
    return backend_->wrap(arr_.attr("__ne__")(other_as_api_obj(backend_, other)));
}

BlockPtr
ArrayApiBlockBackend::Block::pow(const BlockBackend::Scalar& exponent) const
{
    return backend_->wrap(
      arr_.attr("__pow__")(backend_->api().attr("asarray")(exponent.to_numpy())));
}

BlockPtr
ArrayApiBlockBackend::Block::pow(const BlockBackend::Block& exponent) const
{
    return backend_->wrap(arr_.attr("__pow__")(other_as_api_obj(backend_, exponent)));
}

void
ArrayApiBlockBackend::Block::save_hdf5(lasyten::hdf5::Saver& saver,
                                       HighFive::Group& /*h5gr*/,
                                       const std::string& subpath)
{
    saver.save_array(subpath + std::string("arr"), hdf5_export::buffer_from_numpy(to_numpy()));
}

std::shared_ptr<ArrayApiBlockBackend::Block>
ArrayApiBlockBackend::Block::from_hdf5(lasyten::hdf5::Loader& /*loader*/,
                                       HighFive::Group& /*h5gr*/,
                                       const std::string& /*subpath*/)
{
    throw NotImplemented("ArrayApiBlockBackend::Block::from_hdf5 needs the Array API namespace; "
                         "load via ArrayApiBlockBackend::block_from_numpy instead.");
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

ArrayApiBlockBackend::Block const*
ArrayApiBlockBackend::ptr(const BlockCPtr& b)
{
    auto* p = dynamic_cast<ArrayApiBlockBackend::Block const*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not an ArrayApiBlock");
    return p;
}

py::object
ArrayApiBlockBackend::obj(const BlockCPtr& b)
{
    return ptr(b)->obj();
}

BlockPtr
ArrayApiBlockBackend::wrap(py::object arr)
{
    return std::make_shared<Block>(std::move(arr), this);
}

bool
ArrayApiBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<ArrayApiBlockBackend::Block const*>(block.get()) != nullptr;
}

Dtype
ArrayApiBlockBackend::dtype_from_api(py::object api_dtype) const
{
    if (api_dtype.is_none())
        throw std::invalid_argument("ArrayApiBlockBackend: dtype is None");
    auto key = reinterpret_cast<std::uintptr_t>(api_dtype.ptr());
    auto it = lasyten_dtype_map_.find(key);
    if (it != lasyten_dtype_map_.end())
        return it->second;
    // Fall back via numpy dtype conversion (covers aliases / dtype instances).
    return dtype::from_numpy_dtype(numpy_module().attr("dtype")(api_dtype));
}

py::object
ArrayApiBlockBackend::dtype_to_api(Dtype dt) const
{
    auto it = backend_dtype_map_.find(dt);
    if (it == backend_dtype_map_.end())
        throw std::invalid_argument("ArrayApiBlockBackend: unsupported dtype " + dtype::repr(dt));
    return it->second;
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(py::object value)
{
    return Scalar(wrap(std::move(value)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(complex128 value, Dtype dt)
{
    if (dtype::is_real(dt)) {
        return as_scalar(
          api_.attr("asarray")(py::cast(value.real()), py::arg("dtype") = dtype_to_api(dt)));
    }
    return as_scalar(api_.attr("asarray")(py::cast(value), py::arg("dtype") = dtype_to_api(dt)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(py::object value, Dtype dt)
{
    if (py::isinstance<BlockBackend::Scalar>(value)) {
        auto scalar = py::cast<BlockBackend::Scalar>(value);
        if (scalar.dtype() == dt)
            return scalar;
        BlockPtr converted =
          to_dtype(std::const_pointer_cast<BlockBackend::Block>(scalar._block()), dt);
        return Scalar(std::move(converted));
    }
    return as_scalar(api_.attr("asarray")(value, py::arg("dtype") = dtype_to_api(dt)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(bool b)
{
    return as_scalar(
      api_.attr("asarray")(py::cast(b), py::arg("dtype") = dtype_to_api(Dtype::Bool)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(int64 x)
{
    if (backend_dtype_map_.contains(Dtype::Int64))
        return as_scalar(
          api_.attr("asarray")(py::cast(x), py::arg("dtype") = dtype_to_api(Dtype::Int64)));
    // Fall back through numpy int64 then asarray.
    return as_scalar(numpy_module().attr("asarray")(py::cast(x), py::arg("dtype") = "int64"));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(float32 x)
{
    return as_scalar(
      api_.attr("asarray")(py::cast(x), py::arg("dtype") = dtype_to_api(Dtype::Float32)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(float64 x)
{
    return as_scalar(
      api_.attr("asarray")(py::cast(x), py::arg("dtype") = dtype_to_api(Dtype::Float64)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(complex64 z)
{
    return as_scalar(
      api_.attr("asarray")(py::cast(z), py::arg("dtype") = dtype_to_api(Dtype::Complex64)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::as_scalar(complex128 z)
{
    return as_scalar(
      api_.attr("asarray")(py::cast(z), py::arg("dtype") = dtype_to_api(Dtype::Complex128)));
}

// -----------------------------------------------------------------------------
// ArrayApiBlockBackend
// -----------------------------------------------------------------------------

ArrayApiBlockBackend::ArrayApiBlockBackend(py::object api_namespace,
                                           const std::string& default_device)
  : BlockBackend(default_device)
  , api_(std::move(api_namespace))
{
    backend_dtype_map_ = {
        { Dtype::Float32, api_.attr("float32") },
        { Dtype::Float64, api_.attr("float64") },
        { Dtype::Complex64, api_.attr("complex64") },
        { Dtype::Complex128, api_.attr("complex128") },
        { Dtype::Bool, api_.attr("bool") },
    };
    // Int64 is used for index arrays; map if the API exposes it, else fall back to float64 cast
    // sites that need indices via numpy.
    if (py::hasattr(api_, "int64"))
        backend_dtype_map_.emplace(Dtype::Int64, api_.attr("int64"));

    for (auto const& [dt, api_dt] : backend_dtype_map_) {
        lasyten_dtype_map_[reinterpret_cast<std::uintptr_t>(api_dt.ptr())] = dt;
    }
}

std::shared_ptr<ArrayApiBlockBackend>
ArrayApiBlockBackend::from_hdf5(lasyten::hdf5::Loader& /*loader*/,
                                HighFive::Group& /*h5gr*/,
                                const std::string& /*subpath*/)
{
    throw NotImplemented(
      "ArrayApiBlockBackend::from_hdf5 cannot restore the Array API namespace from HDF5 alone.");
}

std::string
ArrayApiBlockBackend::get_backend_name() const
{
    return "ArrayApiBlockBackend";
}

bool
ArrayApiBlockBackend::operator==(BlockBackend const& other) const
{
    if (!BlockBackend::operator==(other))
        return false;
    auto const* o = dynamic_cast<ArrayApiBlockBackend const*>(&other);
    if (o == nullptr)
        return false;
    return api_.is(o->api_);
}

BlockPtr
ArrayApiBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                             const std::vector<py::array_t<int64>>& perms)
{
    // Array API has no np.ix_; fall back via numpy advanced indexing then convert back.
    py::object np = numpy_module();
    py::list ix_parts;
    for (auto const& p : perms)
        ix_parts.append(p);
    py::object indexed = np.attr("asarray")(obj(block))[np.attr("ix_")(*ix_parts)];
    return block_from_numpy(py::array(indexed), get_dtype(block), get_device(block));
}

BlockPtr
ArrayApiBlockBackend::as_block(py::object a,
                               std::optional<Dtype> dtype_opt,
                               std::optional<std::string> device)
{
    if (!device && !py::hasattr(a, "device"))
        device = default_device;

    if (py::isinstance<ArrayApiBlockBackend::Block>(a)) {
        auto* block = py::cast<ArrayApiBlockBackend::Block*>(a);
        BlockPtr out = block->shared_from_this();
        if (dtype_opt && get_dtype(out) != *dtype_opt)
            out = to_dtype(out, *dtype_opt);
        if (device && get_device(out) != as_device(device))
            out = copy_block(out, device);
        return out;
    }

    py::object dtype_arg = py::none();
    if (dtype_opt)
        dtype_arg = dtype_to_api(*dtype_opt);

    py::object block =
      api_.attr("asarray")(a,
                           py::arg("dtype") = dtype_arg,
                           py::arg("device") = device ? py::cast(*device) : py::none());
    if (!dtype_opt || *dtype_opt != Dtype::Bool)
        block = py_mul(py::float_(1.0), block);
    return wrap(std::move(block));
}

std::string
ArrayApiBlockBackend::as_device(std::optional<std::string> device)
{
    std::string dev = device ? *device : default_device;
    // Validate without constructing a Block (Block ctor calls as_device).
    py::object probe = api_.attr("ones")(py::make_tuple(1),
                                         py::arg("dtype") = dtype_to_api(Dtype::Float64),
                                         py::arg("device") = py::cast(dev));
    if (py::hasattr(probe, "device"))
        return py::cast<std::string>(py::str(probe.attr("device")));
    return dev;
}

std::vector<int64>
ArrayApiBlockBackend::abs_argmax(const BlockCPtr& block)
{
    py::object flat_idx = api_.attr("argmax")(api_.attr("abs")(obj(block)));
    // May be 0-d array; convert to Python int.
    int64 idx = py::cast<int64>(py::int_(flat_idx));
    auto shape = get_shape(block);
    std::vector<int64> idcs;
    idcs.reserve(shape.size());
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        idcs.push_back(idx % *it);
        idx /= *it;
    }
    std::reverse(idcs.begin(), idcs.end());
    return idcs;
}

std::vector<int64>
ArrayApiBlockBackend::argmin(const BlockCPtr& block)
{
    py::object flat_idx = api_.attr("argmin")(obj(block));
    // May be 0-d array; convert to Python int.
    int64 idx = py::cast<int64>(py::int_(flat_idx));
    auto shape = get_shape(block);
    std::vector<int64> idcs;
    idcs.reserve(shape.size());
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        idcs.push_back(idx % *it);
        idx /= *it;
    }
    std::reverse(idcs.begin(), idcs.end());
    return idcs;
}

BlockPtr
ArrayApiBlockBackend::abs(const BlockCPtr& a)
{
    return wrap(api_.attr("abs")(obj(a)));
}

BlockPtr
ArrayApiBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    return wrap(api_.attr("expand_dims")(obj(a), py::arg("axis") = pos));
}

bool
ArrayApiBlockBackend::all(const BlockCPtr& a)
{
    return item(wrap(api_.attr("all")(obj(a)))).as_bool();
}

bool
ArrayApiBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    py::object aa = obj(a);
    py::object bb = obj(b);
    py::object res = api_.attr("all")(
      py_le(api_.attr("abs")(py_sub(aa, bb)),
            py_add(py::float_(atol), py_mul(py::float_(rtol), api_.attr("abs")(bb)))));
    return item(wrap(std::move(res))).as_bool();
}

BlockPtr
ArrayApiBlockBackend::angle(const BlockCPtr& /*a*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support angle.");
}

bool
ArrayApiBlockBackend::any(const BlockCPtr& a)
{
    return item(wrap(api_.attr("any")(obj(a)))).as_bool();
}

BlockPtr
ArrayApiBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    py::tuple idx(static_cast<py::ssize_t>(get_shape(block).size()));
    for (py::ssize_t i = 0; i < idx.size(); ++i) {
        if (i == ax)
            idx[i] = obj(mask);
        else
            idx[i] = py::slice(py::none(), py::none(), py::none());
    }
    return wrap(obj(block).attr("__getitem__")(idx));
}

BlockPtr
ArrayApiBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    return wrap(api_.attr("argsort")(obj(block), py::arg("axis") = axis));
}

BlockPtr
ArrayApiBlockBackend::conj(const BlockCPtr& a)
{
    return wrap(api_.attr("conj")(obj(a)));
}

BlockPtr
ArrayApiBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    return wrap(api_.attr("asarray")(obj(a),
                                     py::arg("copy") = true,
                                     py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockPtr
ArrayApiBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    py::object arr = obj(a);
    py::object denom = api_.attr("where")(
      py_lt(api_.attr("abs")(arr), py::float_(cutoff)), api_.attr("asarray")(INFINITY), arr);
    return wrap(py_truediv(py::float_(1.0), denom));
}

std::tuple<BlockPtr, BlockPtr>
ArrayApiBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
{
    py::tuple pair = api_.attr("linalg").attr("eigh")(obj(block));
    BlockPtr w = wrap(pair[0]);
    BlockPtr v = wrap(pair[1]);
    if (sort) {
        BlockPtr perm = argsort(w, sort, /*axis=*/0);
        w = wrap(obj(w).attr("__getitem__")(obj(perm)));
        py::tuple col_idx =
          py::make_tuple(py::slice(py::none(), py::none(), py::none()), obj(perm));
        v = wrap(obj(v).attr("__getitem__")(col_idx));
    }
    return { std::move(w), std::move(v) };
}

std::tuple<BlockPtr, BlockPtr>
ArrayApiBlockBackend::eig(const BlockCPtr& block, std::optional<std::string> sort)
{
    py::tuple pair = api_.attr("linalg").attr("eig")(obj(block));
    Dtype cdtype = dtype::to_complex(get_dtype(block));
    BlockPtr w = to_dtype(wrap(pair[0]), cdtype);
    BlockPtr v = to_dtype(wrap(pair[1]), cdtype);
    if (sort) {
        BlockPtr perm = argsort(w, sort, /*axis=*/0);
        w = wrap(obj(w).attr("__getitem__")(obj(perm)));
        py::tuple col_idx =
          py::make_tuple(py::slice(py::none(), py::none(), py::none()), obj(perm));
        v = wrap(obj(v).attr("__getitem__")(col_idx));
    }
    return { std::move(w), std::move(v) };
}

BlockPtr
ArrayApiBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    BlockPtr w = wrap(api_.attr("linalg").attr("eigvalsh")(obj(block)));
    if (sort) {
        BlockPtr perm = argsort(w, sort, /*axis=*/0);
        w = wrap(obj(w).attr("__getitem__")(obj(perm)));
    }
    return w;
}

BlockPtr
ArrayApiBlockBackend::eigvals(const BlockCPtr& block, std::optional<std::string> sort)
{
    Dtype cdtype = dtype::to_complex(get_dtype(block));
    BlockPtr w = to_dtype(wrap(api_.attr("linalg").attr("eigvals")(obj(block))), cdtype);
    if (sort) {
        BlockPtr perm = argsort(w, sort, /*axis=*/0);
        w = wrap(obj(w).attr("__getitem__")(obj(perm)));
    }
    return w;
}

BlockPtr
ArrayApiBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    auto shape = get_shape(block);
    shape[static_cast<size_t>(axis)] = get_shape(mask)[0];
    BlockPtr res = zeros(shape, get_dtype(block), get_device(block));
    py::list idcs;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64>(i) == axis)
            idcs.append(obj(mask));
        else
            idcs.append(py::slice(py::none(), py::none(), py::none()));
    }
    obj(res).attr("__setitem__")(py::tuple(idcs), obj(copy_block(block, std::nullopt)));
    return res;
}

BlockPtr
ArrayApiBlockBackend::exp(const BlockCPtr& a)
{
    return wrap(api_.attr("exp")(obj(a)));
}

BlockPtr
ArrayApiBlockBackend::block_from_diagonal(const BlockCPtr& /*diag*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support block_from_diagonal.");
}

BlockPtr
ArrayApiBlockBackend::block_from_mask(const BlockCPtr& /*mask*/, Dtype /*dtype*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support block_from_mask.");
}

BlockPtr
ArrayApiBlockBackend::block_from_numpy(const py::array& a,
                                       std::optional<Dtype> dtype_opt,
                                       std::optional<std::string> device)
{
    py::object dtype_arg = py::none();
    if (dtype_opt)
        dtype_arg = dtype_to_api(*dtype_opt);
    return wrap(api_.attr("asarray")(a,
                                     py::arg("dtype") = dtype_arg,
                                     py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockPtr
ArrayApiBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
{
    if (get_shape(a).size() != 2)
        throw std::invalid_argument("get_diagonal expects a 2D block");
    BlockPtr res = wrap(api_.attr("diagonal")(obj(a)));
    if (tol) {
        if (!allclose(a, block_from_diagonal(res), /*rtol=*/0.0, /*atol=*/*tol))
            throw std::invalid_argument("Not a diagonal block.");
    }
    return res;
}

BlockPtr
ArrayApiBlockBackend::imag(const BlockCPtr& a)
{
    return wrap(api_.attr("imag")(obj(a)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::item(const BlockCPtr& a)
{
    py::object arr = obj(a);
    if (py::hasattr(arr, "item"))
        return as_scalar(arr.attr("item")(), get_dtype(a));
    // 0-d Array API arrays: convert via Python float/complex.
    if (is_real(a))
        return as_scalar(py::cast<float64>(py::float_(arr)));
    return as_scalar(py::cast<complex128>(py::module_::import("builtins").attr("complex")(arr)));
}

BlockPtr
ArrayApiBlockBackend::kron(const BlockCPtr& /*a*/, const BlockCPtr& /*b*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support kron.");
}

BlockPtr
ArrayApiBlockBackend::log(const BlockCPtr& a)
{
    return wrap(api_.attr("log")(obj(a)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::max(const BlockCPtr& a)
{
    return item(wrap(api_.attr("max")(obj(a))));
}

BlockBackend::Scalar
ArrayApiBlockBackend::max_abs(const BlockCPtr& a)
{
    return item(wrap(api_.attr("max")(api_.attr("abs")(obj(a)))));
}

BlockBackend::Scalar
ArrayApiBlockBackend::min(const BlockCPtr& a)
{
    return item(wrap(api_.attr("min")(obj(a))));
}

BlockBackend::Scalar
ArrayApiBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    py::object res;
    if (axis)
        res = api_.attr("linalg").attr("vector_norm")(
          obj(a), py::arg("axis") = *axis, py::arg("ord") = order);
    else
        res = api_.attr("linalg").attr("vector_norm")(
          obj(a), py::arg("axis") = py::none(), py::arg("ord") = order);
    return item(wrap(std::move(res)));
}

BlockPtr
ArrayApiBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    // tensordot with 0 contracted axes; for bool-friendly APIs prefer broadcast mul if needed.
    return wrap(api_.attr("tensordot")(obj(a), obj(b), 0));
}

BlockPtr
ArrayApiBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    return wrap(api_.attr("permute_dims")(obj(a), to_py_list(permutation)));
}

BlockPtr
ArrayApiBlockBackend::random_normal(const std::vector<int64>& dims,
                                    Dtype dt,
                                    float64 sigma,
                                    std::optional<std::string> device)
{
    py::object np = numpy_module();
    py::object res = np.attr("random").attr("normal")(
      py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = to_py_list(dims));
    if (!dtype::is_real(dt)) {
        py::object imag = np.attr("random").attr("normal")(
          py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = to_py_list(dims));
        res = py_add(res, py_mul(py::cast(complex128{ 0.0, 1.0 }), imag));
    }
    return wrap(
      api_.attr("asarray")(res, py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockPtr
ArrayApiBlockBackend::random_uniform(const std::vector<int64>& dims,
                                     Dtype dt,
                                     std::optional<std::string> device)
{
    py::object np = numpy_module();
    py::object res = np.attr("random").attr("uniform")(-1, 1, py::arg("size") = to_py_list(dims));
    if (!dtype::is_real(dt)) {
        py::object imag =
          np.attr("random").attr("uniform")(-1, 1, py::arg("size") = to_py_list(dims));
        res = py_add(res, py_mul(py::cast(complex128{ 0.0, 1.0 }), imag));
    }
    return wrap(
      api_.attr("asarray")(res, py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockPtr
ArrayApiBlockBackend::real(const BlockCPtr& a)
{
    return wrap(api_.attr("real")(obj(a)));
}

BlockPtr
ArrayApiBlockBackend::real_if_close(const BlockCPtr& /*a*/, float64 /*tol*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support real_if_close.");
}

BlockPtr
ArrayApiBlockBackend::scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis)
{
    auto shape = get_shape(block);
    py::list idx;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64>(i) == axis)
            idx.append(py::slice(py::none(), py::none(), py::none()));
        else
            idx.append(py::none());
    }
    return wrap(py_mul(obj(block), obj(factors).attr("__getitem__")(py::tuple(idx))));
}

BlockPtr
ArrayApiBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    // Not in Array API; implement 1D repeat via concatenate.
    if (get_shape(a).size() != 1)
        throw NotImplemented("ArrayApiBlockBackend::tile only supports 1D blocks.");
    py::list parts;
    for (int64 i = 0; i < repeats; ++i)
        parts.append(obj(a));
    return wrap(api_.attr("concat")(parts, py::arg("axis") = 0));
}

std::vector<std::string>
ArrayApiBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                        const std::string& indent,
                                        int64 /*max_width*/,
                                        int64 max_lines)
{
    std::string s = py::cast<std::string>(py::str(obj(a)));
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= s.size()) {
        auto pos = s.find('\n', start);
        std::string line =
          (pos == std::string::npos) ? s.substr(start) : s.substr(start, pos - start);
        lines.push_back(indent + line);
        if (pos == std::string::npos)
            break;
        start = pos + 1;
    }
    if (static_cast<int64>(lines.size()) > max_lines) {
        int64 first = (max_lines - 1) / 2;
        int64 last = max_lines - 1 - first;
        std::vector<std::string> truncated;
        truncated.insert(truncated.end(), lines.begin(), lines.begin() + first);
        truncated.push_back(indent + "...");
        truncated.insert(truncated.end(), lines.end() - last, lines.end());
        return truncated;
    }
    return lines;
}

BlockPtr
ArrayApiBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    return wrap(api_.attr("reshape")(obj(a), to_py_list(shape)));
}

BlockPtr
ArrayApiBlockBackend::sqrt(const BlockCPtr& /*a*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support sqrt.");
}

BlockPtr
ArrayApiBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    return wrap(api_.attr("squeeze")(obj(a), to_py_tuple(idcs)));
}

BlockPtr
ArrayApiBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    py::object arr = obj(block);
    return wrap(api_.attr("where")(py_gt(arr, py::float_(cutoff)), api_.attr("log")(arr), 0.0));
}

BlockPtr
ArrayApiBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    return wrap(api_.attr("sum")(obj(a), py::arg("axis") = ax));
}

BlockBackend::Scalar
ArrayApiBlockBackend::sum_all(const BlockCPtr& a)
{
    return item(wrap(api_.attr("sum")(obj(a))));
}

BlockPtr
ArrayApiBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(py_mul(obj(a), obj(b)));
}

BlockPtr
ArrayApiBlockBackend::tdot(const BlockCPtr& a,
                           const BlockCPtr& b,
                           const std::vector<int64>& idcs_a,
                           const std::vector<int64>& idcs_b)
{
    return wrap(api_.attr("tensordot")(
      obj(a), obj(b), py::make_tuple(to_py_list(idcs_a), to_py_list(idcs_b))));
}

BlockPtr
ArrayApiBlockBackend::to_dtype(const BlockCPtr& a, Dtype dt)
{
    return wrap(api_.attr("astype")(obj(a), dtype_to_api(dt)));
}

BlockBackend::Scalar
ArrayApiBlockBackend::trace_full(const BlockCPtr& a)
{
    auto shape = get_shape(a);
    int64 num_trace = static_cast<int64>(shape.size()) / 2;
    int64 trace_dim = 1;
    for (int64 i = 0; i < num_trace; ++i)
        trace_dim *= shape[static_cast<size_t>(i)];
    std::vector<int64> perm;
    perm.reserve(shape.size());
    for (int64 i = 0; i < num_trace; ++i)
        perm.push_back(i);
    for (int64 i = 2 * num_trace - 1; i >= num_trace; --i)
        perm.push_back(i);
    BlockPtr reshaped = reshape(permute_axes(a, perm), { trace_dim, trace_dim });
    return item(wrap(api_.attr("linalg").attr("trace")(obj(reshaped))));
}

BlockPtr
ArrayApiBlockBackend::trace_partial(const BlockCPtr& a,
                                    const std::vector<int64>& idcs1,
                                    const std::vector<int64>& idcs2,
                                    const std::vector<int64>& remaining_idcs)
{
    std::vector<int64> perm = remaining_idcs;
    perm.insert(perm.end(), idcs1.begin(), idcs1.end());
    perm.insert(perm.end(), idcs2.begin(), idcs2.end());
    BlockPtr permuted = permute_axes(a, perm);
    auto shape = get_shape(permuted);
    int64 trace_dim = 1;
    for (size_t i = 0; i < idcs1.size(); ++i)
        trace_dim *= shape[remaining_idcs.size() + i];
    std::vector<int64> new_shape = { -1, trace_dim, trace_dim };
    return wrap(api_.attr("linalg").attr("trace")(obj(reshape(permuted, new_shape))));
}

BlockPtr
ArrayApiBlockBackend::eye_matrix(int64 dim, Dtype dt, std::optional<std::string> device)
{
    return wrap(api_.attr("eye")(dim,
                                 py::arg("dtype") = dtype_to_api(dt),
                                 py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockBackend::Scalar
ArrayApiBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    return item(wrap(obj(a).attr("__getitem__")(to_py_tuple(idcs))));
}

BlockPtr
ArrayApiBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(api_.attr("matmul")(obj(a), obj(b)));
}

BlockPtr
ArrayApiBlockBackend::matrix_exp(const BlockCPtr& /*matrix*/)
{
    throw NotImplemented("ArrayApiBlockBackend does not support matrix_exp.");
}

std::tuple<BlockPtr, BlockPtr>
ArrayApiBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    py::tuple pair =
      api_.attr("linalg").attr("qr")(obj(a), py::arg("mode") = full ? "complete" : "reduced");
    return { wrap(pair[0]), wrap(pair[1]) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
ArrayApiBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
{
    std::string algo = algorithm ? *algorithm : "default";
    if (algo != "default")
        throw std::invalid_argument("SVD algorithm not supported: " + algo);
    py::tuple triple = api_.attr("linalg").attr("svd")(obj(a), py::arg("full_matrices") = false);
    return { wrap(triple[0]), wrap(triple[1]), wrap(triple[2]) };
}

const std::vector<std::string>&
ArrayApiBlockBackend::possible_svd_algorithms() const
{
    static const std::vector<std::string> algos{ "default" };
    return algos;
}

BlockPtr
ArrayApiBlockBackend::ones_block(const std::vector<int64>& shape,
                                 Dtype dt,
                                 std::optional<std::string> device)
{
    return wrap(api_.attr("ones")(to_py_list(shape),
                                  py::arg("dtype") = dtype_to_api(dt),
                                  py::arg("device") = device ? py::cast(*device) : py::none()));
}

BlockPtr
ArrayApiBlockBackend::zeros(const std::vector<int64>& shape,
                            Dtype dt,
                            std::optional<std::string> device)
{
    return wrap(api_.attr("zeros")(to_py_list(shape),
                                   py::arg("dtype") = dtype_to_api(dt),
                                   py::arg("device") = device ? py::cast(*device) : py::none()));
}

} // namespace lasyten
