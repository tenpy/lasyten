#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/numpy.h>
#include <lasyten/detail/python_api.h>
#include <lasyten/tools.h>

#include <lasyten/tools/hdf5.h>
#include <lasyten/tools/hdf5_export.h>
#include <map>
#include <mutex>
#include <regex>
#include <span>
#include <stdexcept>

namespace lasyten {

using std::complex_literals::operator""i;

namespace {
py::module_
numpy_module()
{
    return py::module_::import("numpy");
}

py::object
np_attr(char const* name)
{
    return numpy_module().attr(name);
}
} // namespace

// -----------------------------------------------------------------------------
// NumpyBlock
// -----------------------------------------------------------------------------

NumpyBlockBackend::Block::Block(py::array arr)
  : arr_(std::move(arr))
{
    try {
        dtype::from_numpy_dtype(arr_.dtype());
    } catch (std::invalid_argument& e) {
        std::string dtype_str = py::str(arr_.dtype()).cast<std::string>();
        throw std::invalid_argument("NumpyBlockBackend::Block: invalid numpy dtype: " + dtype_str);
    }
}

std::vector<int64>
NumpyBlockBackend::Block::shape() const
{
    std::vector<int64> s;
    s.reserve(static_cast<size_t>(arr_.ndim()));
    for (py::ssize_t i = 0; i < arr_.ndim(); ++i)
        s.push_back(static_cast<int64>(arr_.shape(i)));
    return s;
}

Dtype
NumpyBlockBackend::Block::dtype() const
{
    return dtype::from_numpy_dtype(arr_.dtype());
}

const std::string&
NumpyBlockBackend::Block::device() const
{
    static std::string device = "cpu";
    return device;
}

py::array
NumpyBlockBackend::Block::to_numpy(Dtype dtype) const
{
    return py::array(arr_.attr("astype")(dtype::to_numpy_dtype(dtype)));
}

BlockBackend*
NumpyBlockBackend::Block::get_backend() const
{
    return NumpyBlockBackend::from_factory(device());
}

BlockCPtr
NumpyBlockBackend::Block::get_item(std::span<const BlockIndex> key) const
{
    py::object py_key = BlockBackend::block_indices_to_py(key);
    py::array result = arr_.attr("__getitem__")(py_key);
    return std::make_shared<const NumpyBlockBackend::Block>(std::move(result));
}

BlockPtr
NumpyBlockBackend::Block::get_item(std::span<const BlockIndex> key)
{
    py::object py_key = BlockBackend::block_indices_to_py(key);
    py::array result = arr_.attr("__getitem__")(py_key);
    return std::make_shared<NumpyBlockBackend::Block>(std::move(result));
}

BlockCPtr
NumpyBlockBackend::Block::get_item(py::object key) const
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    py::array result = arr_.attr("__getitem__")(key_copy);
    return std::make_shared<const NumpyBlockBackend::Block>(std::move(result));
}

py::object
NumpyBlockBackend::Block::_item_key_cast_Blocks_to_numpy(py::object key) const
{
    if (py_isinstance<NumpyBlockBackend::Block>(key)) {
        NumpyBlockBackend::Block* block = key.cast<NumpyBlockBackend::Block*>();
        return block->to_numpy();
    } else if (py::isinstance<py::tuple>(key)) {
        // cast any Block objects in the key to numpy arrays and make a new tuple
        py::tuple key_tuple = py::reinterpret_borrow<py::tuple>(key);
        py::tuple new_tuple = py::tuple(key_tuple.size());
        for (py::ssize_t i = 0; i < key_tuple.size(); ++i) {
            if (py_isinstance<NumpyBlockBackend::Block>(key_tuple[i])) {
                NumpyBlockBackend::Block* block = key_tuple[i].cast<NumpyBlockBackend::Block*>();
                new_tuple[i] = block->to_numpy();
            } else {
                new_tuple[i] = key_tuple[i];
            }
        }
        return std::move(new_tuple);
    }
    return key;
}

BlockPtr
NumpyBlockBackend::Block::get_item(py::object key)
{
    if (key.is_none())
        return shared_from_this();
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
        return get_item(std::span<const BlockIndex>(*idcs));
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    py::array result = arr_.attr("__getitem__")(key_copy);
    return std::make_shared<NumpyBlockBackend::Block>(std::move(result));
}

void
NumpyBlockBackend::Block::set_item(std::span<const BlockIndex> key,
                                   const BlockBackend::Block& value)
{
    py::object py_key = BlockBackend::block_indices_to_py(key);
    arr_.attr("__setitem__")(py_key, value.to_numpy());
}

void
NumpyBlockBackend::Block::set_item(py::object key, py::object value)
{
    if (auto idcs = BlockBackend::try_py_key_to_block_indices(key)) {
        if (py_isinstance<BlockBackend::Block>(value)) {
            set_item(std::span<const BlockIndex>(*idcs), value.cast<BlockBackend::Block&>());
            return;
        }
        if (py_isinstance<BlockBackend::Scalar>(value)) {
            set_item(std::span<const BlockIndex>(*idcs), value.cast<BlockBackend::Scalar&>());
            return;
        }
        NumpyBlockBackend::Block tmp(
          py::module_::import("numpy").attr("asarray")(value).cast<py::array>());
        set_item(std::span<const BlockIndex>(*idcs), tmp);
        return;
    }
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    if (py_isinstance<NumpyBlockBackend::Block>(value)) {
        NumpyBlockBackend::Block* block = value.cast<NumpyBlockBackend::Block*>();
        value = block->to_numpy();
    }
    arr_.attr("__setitem__")(key_copy, value);
}

void
NumpyBlockBackend::Block::set_item(const std::vector<int64>& key, const Scalar& value)
{
    // Tuple key so numpy does multi-index assignment, not fancy indexing.
    arr_.attr("__setitem__")(py::tuple(py::cast(key)), value.to_numpy());
}

void
NumpyBlockBackend::Block::set_item(int64 idx, const Scalar& value)
{
    if (ndim() != 1)
        throw std::invalid_argument(
          "NumpyBlockBackend::Block::set_item(int64): block must be 1-dimensional");
    set_item(std::vector<int64>{ idx }, value);
}

complex128
NumpyBlockBackend::Block::_item_as_complex128() const
{
    return (arr_.attr("item")()).cast<complex128>();
}

int64
NumpyBlockBackend::Block::_item_as_int64() const
{
    return (arr_.attr("item")()).cast<int64>();
}

BlockPtr
NumpyBlockBackend::Block::operator+(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__add__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator-(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__sub__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator*(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__mul__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator/(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__truediv__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator<(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__lt__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator<=(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__le__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator>(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__gt__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator>=(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__ge__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator==(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__eq__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::operator!=(const BlockBackend::Block& other) const
{
    return wrap(arr_.attr("__ne__")(other.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::pow(const BlockBackend::Scalar& exponent) const
{
    return wrap(arr_.attr("__pow__")(exponent.to_numpy()));
}

BlockPtr
NumpyBlockBackend::Block::pow(const BlockBackend::Block& exponent) const
{
    return wrap(arr_.attr("__pow__")(exponent.to_numpy()));
}

void
NumpyBlockBackend::Block::save_hdf5(lasyten::hdf5::Saver& saver,
                                    HighFive::Group& /*h5gr*/,
                                    const std::string& subpath)
{
    saver.save_array(subpath + std::string("arr"), hdf5_export::buffer_from_numpy(arr_));
}

std::shared_ptr<NumpyBlockBackend::Block>
NumpyBlockBackend::Block::from_hdf5(lasyten::hdf5::Loader& loader,
                                    HighFive::Group& h5gr,
                                    std::string const& subpath)
{
    hid_t id = loader.open(subpath + std::string("arr"));
    auto buf = loader.load_array(id);
    H5Idec_ref(id);
    py::array arr = hdf5_export::numpy_from_buffer(buf);
    // Older/raw HDF5 writers store bool as uint8; uint8 is not a valid block dtype.
    auto np = py::module_::import("numpy");
    if (arr.dtype().kind() == 'u' && arr.dtype().itemsize() == 1) {
        arr = py::reinterpret_steal<py::array>(arr.attr("astype")(np.attr("bool_")).release());
    }
    auto obj = std::make_shared<NumpyBlockBackend::Block>(arr);
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend helpers
// -----------------------------------------------------------------------------

NumpyBlockBackend::Block const*
NumpyBlockBackend::ptr(const BlockCPtr& b)
{
    auto* p = dynamic_cast<NumpyBlockBackend::Block const*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not a NumpyBlock");
    return p;
}

py::object
NumpyBlockBackend::obj(const BlockCPtr& b)
{
    return ptr(b)->to_numpy();
}

BlockPtr
NumpyBlockBackend::wrap(py::object arr)
{
    return std::make_shared<NumpyBlockBackend::Block>(std::move(arr));
}

bool
NumpyBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<NumpyBlockBackend::Block const*>(block.get()) != nullptr;
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend::as_scalar
// -----------------------------------------------------------------------------
BlockBackend::Scalar
NumpyBlockBackend::as_scalar(py::array value)
{
    BlockPtr block = wrap(value);
    return BlockBackend::Scalar(std::move(block));
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(complex128 value, Dtype dtype)
{
    // np.array(complex, dtype=float) raises TypeError even for 0j; take the real part
    // when the target dtype is real (mapping amplitudes are stored as complex128).
    if (dtype::is_real(dtype)) {
        py::array arr = np_attr("array")(py::cast(value.real()), dtype::to_numpy_dtype(dtype));
        return as_scalar(arr);
    }
    py::array arr = np_attr("array")(py::cast(value), dtype::to_numpy_dtype(dtype));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(py::object value, Dtype dtype)
{
    if (py_isinstance<BlockBackend::Scalar>(value)) {
        auto scalar = py::cast<BlockBackend::Scalar>(value);
        if (scalar.dtype() == dtype)
            return scalar;
        BlockPtr converted =
          to_dtype(std::const_pointer_cast<BlockBackend::Block>(scalar._block()), dtype);
        return BlockBackend::Scalar(std::move(converted));
    }
    py::array arr = np_attr("array")(value, dtype::to_numpy_dtype(dtype));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(int64 x)
{
    py::array arr = np_attr("int64")(py::cast(x));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(bool b)
{
    py::array arr = np_attr("bool_")(py::cast(b));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(float32 x)
{
    py::array arr = np_attr("float32")(py::cast(x));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(float64 x)
{
    py::array arr = np_attr("float64")(py::cast(x));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(complex64 z)
{
    py::array arr = np_attr("complex64")(py::cast(z));
    return as_scalar(arr);
}

BlockBackend::Scalar
NumpyBlockBackend::as_scalar(complex128 z)
{
    py::array arr = np_attr("complex128")(py::cast(z));
    return as_scalar(arr);
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend
// -----------------------------------------------------------------------------

NumpyBlockBackend*
NumpyBlockBackend::from_factory(const std::string& device)
{
    return from_factory_shared(device).get();
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::from_factory_shared(const std::string& device)
{
    if (device != "cpu")
        throw std::invalid_argument(
          "NumpyBlockBackend::from_factory only supports device \"cpu\", got: " + device);
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<NumpyBlockBackend>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device);
    if (it == cache.end()) {
        it =
          cache.emplace(device, std::shared_ptr<NumpyBlockBackend>(new NumpyBlockBackend())).first;
    }
    return it->second;
}

NumpyBlockBackend::NumpyBlockBackend()
  : BlockBackend("cpu")
{
}

std::string
NumpyBlockBackend::get_backend_name() const
{
    return "NumpyBlockBackend";
}

// the following was generated by ../pybind11_codegen/pybind11_codegen.py
// gen_cpp_definition --py-name NumpyBlockBackend --header-file
// include/lasyten/block_backend/numpy.h
// --src-file src/block_backend/numpy_FROM_SCRIPT.cpp

BlockPtr
NumpyBlockBackend::abs(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.abs(a)
     */
    return wrap(np_attr("abs")(obj(a)));
}

BlockPtr
NumpyBlockBackend::as_block(py::object a,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * block = np.asarray(a, dtype=self.backend_dtype_map[dtype])
     * if np.issubdtype(block.dtype, np.integer):
     *             block = block.astype(np.float64, copy=False)
     * return block
     */
    (void)as_device(device);
    if (py_isinstance<NumpyBlockBackend::Block>(a)) {
        // just use the block directly, no need to convert to numpy and back
        BlockPtr block = a.cast<BlockPtr>();
        if (dtype_opt) {
            block = block->get_backend()->to_dtype(block, *dtype_opt);
        }
        return block;
    }
    if (py_isinstance<BlockBackend::Block>(a)) {
        // Foreign backend Block (e.g. Torch): convert via numpy.
        BlockPtr block = a.cast<BlockPtr>();
        py::array arr = block->to_numpy();
        if (dtype_opt)
            arr = py::array(arr.attr("astype")(dtype::to_numpy_dtype(*dtype_opt)));
        return wrap(arr);
    }
    py::module_ np = numpy_module();
    py::object dt = dtype_opt ? dtype::to_numpy_dtype(*dtype_opt) : py::none();
    py::object arr = np.attr("asarray")(a, dt);
    // integer -> float64 like Python
    if (py::isinstance<py::array>(arr)) {
        py::array arr_arr = py::reinterpret_borrow<py::array>(arr);
        py::object d = arr_arr.dtype();
        if (np.attr("issubdtype")(d, np.attr("integer")).cast<bool>())
            arr = arr.attr("astype")(np.attr("float64"), py::arg("copy") = false);
    }
    return wrap(arr);
}

std::string
NumpyBlockBackend::as_device(std::optional<std::string> device)
{
    /* converted from following python code:
     * if device is None:
     *             return self.default_device
     * if re.match(rf'{self.default_device}:\d+', str(device)):
     *             return self.default_device
     * if device != self.default_device:
     *             msg = f'{self.__class__.__name__} does not support device {device}.'
     *             raise ValueError(msg)
     * return device
     */
    if (!device)
        return default_device;
    // Accept aliases like "cpu:0" for default_device "cpu" (torch-style device strings).
    if (std::regex_match(*device, std::regex(default_device + R"(:\d+)")))
        return default_device;
    if (*device != default_device)
        throw std::invalid_argument("NumpyBlockBackend does not support device " + *device);
    return *device;
}

BlockPtr
NumpyBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    /* converted from following python code:
     * return np.expand_dims(a, pos)
     */
    return wrap(np_attr("expand_dims")(obj(a), pos));
}

std::vector<int64>
NumpyBlockBackend::abs_argmax(const BlockCPtr& block)
{
    /* converted from following python code:
     * return np.unravel_index(np.argmax(np.abs(block)), block.shape)
     */
    py::array arr = py::reinterpret_borrow<py::array>(obj(block));
    py::module_ np = numpy_module();
    py::tuple shape_tuple(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i)
        shape_tuple[i] = py::int_(arr.shape(i));
    py::object idx = np.attr("unravel_index")(np.attr("argmax")(np.attr("abs")(arr)), shape_tuple);
    py::tuple t = py::cast<py::tuple>(idx);
    std::vector<int64> out;
    for (auto const& item : t)
        out.push_back(item.cast<int64>());
    return out;
}

std::vector<int64>
NumpyBlockBackend::argmin(const BlockCPtr& block)
{
    py::array arr = py::reinterpret_borrow<py::array>(obj(block));
    py::module_ np = numpy_module();
    py::tuple shape_tuple(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i)
        shape_tuple[i] = py::int_(arr.shape(i));
    py::object idx = np.attr("unravel_index")(np.attr("argmin")(arr), shape_tuple);
    py::tuple t = py::cast<py::tuple>(idx);
    std::vector<int64> out;
    for (auto const& item : t)
        out.push_back(item.cast<int64>());
    return out;
}

bool
NumpyBlockBackend::all(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.all(a)
     */
    return np_attr("all")(obj(a)).cast<bool>();
}

bool
NumpyBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    /* converted from following python code:
     * return np.allclose(a, b, rtol=rtol, atol=atol)
     */
    return np_attr("allclose")(obj(a), obj(b), py::arg("rtol") = rtol, py::arg("atol") = atol)
      .cast<bool>();
}

BlockPtr
NumpyBlockBackend::angle(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.angle(a)
     */
    return wrap(np_attr("angle")(obj(a)));
}

bool
NumpyBlockBackend::any(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.any(a)
     */
    return np_attr("any")(obj(a)).cast<bool>();
}

BlockPtr
NumpyBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    /* converted from following python code:
     * return np.compress(mask, block, ax)
     */
    return wrap(np_attr("compress")(obj(mask), obj(block), ax));
}

BlockPtr
NumpyBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    /* converted from following python code:
     * return np.argsort(block, axis=axis)
     */
    return wrap(np_attr("argsort")(obj(block), py::arg("axis") = axis));
}

BlockPtr
NumpyBlockBackend::conj(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.conj(a)
     */
    return wrap(np_attr("conj")(obj(a)));
}

BlockPtr
NumpyBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.copy(a)
     */
    (void)as_device(device);
    return wrap(np_attr("copy")(obj(a)));
}

/// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
BlockPtr
NumpyBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    /* converted from following python code:
     * return 1 / np.where(np.abs(a) < cutoff, np.inf, a)
     */
    py::object arr = obj(a);
    py::module_ np = numpy_module();
    py::object denom = np.attr("where")(
      np.attr("less")(np.attr("abs")(arr), py::float_(cutoff)), np.attr("inf"), arr);
    return wrap(py::float_(1.0) / denom);
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
{
    /* converted from following python code:
     * w, v = np.linalg.eigh(block)
     * if sort is not None:
     *             perm = self.argsort(w, sort)
     *             w = np.take(w, perm)
     *             v = np.take(v, perm, axis=1)
     * return w, v
     */
    py::object a = obj(block);
    py::tuple res = py::cast<py::tuple>(np_attr("linalg").attr("eigh")(a));
    py::object w = res[0];
    py::object v = res[1];
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, 0);
        py::object perm_arr = obj(perm);
        w = np_attr("take")(w, perm_arr);
        v = np_attr("take")(v, perm_arr, py::arg("axis") = 1);
    }
    return { wrap(w), wrap(v) };
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::eig(const BlockCPtr& block, std::optional<std::string> sort)
{
    py::object a = obj(block);
    py::tuple res = py::cast<py::tuple>(np_attr("linalg").attr("eig")(a));
    Dtype cdtype = dtype::to_complex(get_dtype(block));
    BlockPtr w = to_dtype(wrap(res[0]), cdtype);
    BlockPtr v = to_dtype(wrap(res[1]), cdtype);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        py::object perm_arr = obj(perm);
        w = wrap(np_attr("take")(obj(w), perm_arr));
        v = wrap(np_attr("take")(obj(v), perm_arr, py::arg("axis") = 1));
    }
    return { std::move(w), std::move(v) };
}

BlockPtr
NumpyBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    /* converted from following python code:
     * w = np.linalg.eigvalsh(block)
     * if sort is not None:
     *             perm = self.argsort(w, sort)
     *             w = np.take(w, perm)
     * return w
     */
    py::object w = np_attr("linalg").attr("eigvalsh")(obj(block));
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, 0);
        w = np_attr("take")(w, obj(perm));
    }
    return wrap(w);
}

BlockPtr
NumpyBlockBackend::eigvals(const BlockCPtr& block, std::optional<std::string> sort)
{
    Dtype cdtype = dtype::to_complex(get_dtype(block));
    BlockPtr w = to_dtype(wrap(np_attr("linalg").attr("eigvals")(obj(block))), cdtype);
    if (sort) {
        BlockPtr perm = argsort(w, sort, 0);
        w = wrap(np_attr("take")(obj(w), obj(perm)));
    }
    return w;
}

BlockPtr
NumpyBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    /* converted from following python code:
     * # OPTIMIZE is there a numpy builtin function that does this? or at least part of this?
     * shape = list(block.shape)
     * shape[axis] = len(mask)
     * res = np.zeros(shape, dtype=block.dtype)
     * idcs = [slice(None, None, None)] * len(shape)
     * idcs[axis] = mask
     * res[tuple(idcs)] = block.copy()
     * # OPTIMIZE is the copy needed
     * return res
     */
    py::array a_arr = py::reinterpret_borrow<py::array>(obj(block));
    py::object m = obj(mask);
    py::module_ np = numpy_module();
    py::list shape_list;
    for (py::ssize_t i = 0; i < a_arr.ndim(); ++i)
        shape_list.append(py::int_(a_arr.shape(i)));
    shape_list[axis] = py::len(m);
    py::object shape = py::tuple(shape_list);
    py::object res = np.attr("zeros")(shape, py::arg("dtype") = a_arr.dtype());
    py::list idcs;
    for (int64 i = 0; i < static_cast<int64>(a_arr.ndim()); ++i)
        idcs.append(i == axis ? m : py::object(py::slice(py::none(), py::none(), py::none())));
    res[py::tuple(idcs)] = a_arr;
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::exp(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.exp(a)
     */
    return wrap(np_attr("exp")(obj(a)));
}

BlockPtr
NumpyBlockBackend::block_from_diagonal(const BlockCPtr& diag)
{
    /* converted from following python code:
     * return np.diag(diag)
     */
    return wrap(np_attr("diag")(obj(diag)));
}

BlockPtr
NumpyBlockBackend::block_from_mask(const BlockCPtr& mask, Dtype dtype)
{
    /* converted from following python code:
     * (M,) = mask.shape
     * N = np.sum(mask)
     * res = np.zeros((N, M), dtype=self.backend_dtype_map[dtype])
     * res[np.arange(N), mask] = 1
     * return res
     */
    py::array m_arr = py::reinterpret_borrow<py::array>(obj(mask));
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    int64 M = static_cast<int64>(m_arr.shape(0));
    int64 N = np.attr("sum")(m_arr).cast<int64>();
    py::object res = np.attr("zeros")(py::make_tuple(N, M), py::arg("dtype") = dt);
    res[py::make_tuple(np.attr("arange")(N), m_arr)] = 1;
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::block_from_numpy(const py::array& a,
                                    std::optional<Dtype> dtype_opt,
                                    std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * if dtype is None:
     *             return a
     * return np.asarray(a, self.backend_dtype_map[dtype])
     */
    (void)as_device(device);
    if (!dtype_opt)
        return wrap(py::object(a));
    return wrap(np_attr("asarray")(a, dtype::to_numpy_dtype(*dtype_opt)));
}

BlockPtr
NumpyBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
{
    /* converted from following python code:
     * res = np.diagonal(a)
     * if tol is not None:
     *             if not np.allclose(a, np.diag(res), atol=tol):
     *                 raise ValueError('Not a diagonal block.')
     * return res
     */
    py::object arr = obj(a);
    py::object res = np_attr("diagonal")(arr);
    if (tol) {
        py::object diag_mat = np_attr("diag")(res);
        if (!np_attr("allclose")(arr, diag_mat, py::arg("atol") = *tol).cast<bool>())
            throw std::invalid_argument("Not a diagonal block.");
    }
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::imag(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.imag(a)
     */
    return wrap(np_attr("imag")(obj(a)));
}

BlockBackend::Scalar
NumpyBlockBackend::inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger)
{
    /* converted from following python code:
     * # OPTIMIZE use np.sum(a * b) instead?
     * if do_dagger:
     *             return np.tensordot(np.conj(a), b, a.ndim).item()
     * return np.tensordot(a, b, [list(range(a.ndim)), list(reversed(range(a.ndim)))]).item()
     */
    // OPTIMIZE use np.sum(a * b) instead?
    if (a->ndim() != b->ndim()) {
        throw std::invalid_argument("a and b must have the same number of dimensions");
    }
    py::object contr;
    if (do_dagger) {
        contr = np_attr("tensordot")(obj(conj(a)), obj(b), a->ndim());
    } else {
        std::vector<int64> range(a->ndim());
        std::vector<int64> rev_range(a->ndim());
        for (int64 i = 0; i < a->ndim(); ++i) {
            range[i] = i;
            rev_range[i] = a->ndim() - 1 - i;
        }
        contr = np_attr("tensordot")(
          obj(a), obj(b), py::make_tuple(py::cast(range), py::cast(rev_range)));
    }
    return item(wrap(contr.attr("item")()));
}

BlockBackend::Scalar
NumpyBlockBackend::item(const BlockCPtr& a)
{
    /* converted from following python code:
     * return a.item()
     */
    return Scalar(wrap(obj(a).attr("item")()));
}

BlockPtr
NumpyBlockBackend::kron(const BlockCPtr& a, const BlockCPtr& b)
{
    /* converted from following python code:
     * return np.kron(a, b)
     */
    return wrap(np_attr("kron")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::log(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.log(a)
     */
    return wrap(np_attr("log")(obj(a)));
}

BlockBackend::Scalar
NumpyBlockBackend::max(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.max(a).item()
     */
    return Scalar(item(wrap(np_attr("max")(obj(a)))));
}

BlockBackend::Scalar
NumpyBlockBackend::max_abs(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.max(np.abs(a)).item()
     */
    return Scalar(item(wrap(np_attr("max")(np_attr("abs")(obj(a))))));
}

BlockBackend::Scalar
NumpyBlockBackend::min(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.min(a).item()
     */
    return Scalar(item(wrap(np_attr("min")(obj(a)))));
}

BlockBackend::Scalar
NumpyBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    /* converted from following python code:
     * if axis is None:
     *             return np.linalg.norm(a.ravel(), ord=order).item()
     * return np.linalg.norm(a, ord=order, axis=axis)
     */
    py::object arr = obj(a);
    if (!axis) {
        return item(
          wrap(np_attr("linalg").attr("norm")(arr.attr("ravel")(), py::arg("ord") = order)));
    }
    return item(
      wrap(np_attr("linalg").attr("norm")(arr, py::arg("ord") = order, py::arg("axis") = *axis)));
}

BlockPtr
NumpyBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    /* converted from following python code:
     * return np.tensordot(a, b, ((), ()))
     */
    return wrap(np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::tuple(), py::tuple())));
}

BlockPtr
NumpyBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    /* converted from following python code:
     * return np.transpose(a, permutation)
     */
    return wrap(np_attr("transpose")(obj(a), py::cast(permutation)));
}

BlockPtr
NumpyBlockBackend::random_normal(const std::vector<int64>& dims,
                                 Dtype dtype,
                                 float64 sigma,
                                 std::optional<std::string> device)
{
    /* converted from following python code:
     * # if sigma is standard deviation for complex numbers, need to divide by sqrt(2)
     * # to get standard deviation in real and imag parts
     * if not dtype.is_real:
     *             sigma /= np.sqrt(2)
     * _ = self.as_device(device)
     * # for input check only
     * res = np.random.normal(loc=0, scale=sigma, size=dims)
     * if not dtype.is_real:
     *             res = res + 1.0j * np.random.normal(loc=0, scale=sigma, size=dims)
     * return res
     */
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    if (!dtype::is_real(dtype))
        sigma /= std::sqrt(2.0);
    py::object res = np.attr("random").attr("normal")(
      py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = dims);
    if (!dtype::is_real(dtype))
        res = res + py::cast(complex128(0, 1)) *
                      np.attr("random").attr("normal")(
                        py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::random_uniform(const std::vector<int64>& dims,
                                  Dtype dtype,
                                  std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * res = np.random.uniform(-1, 1, size=dims)
     * if not dtype.is_real:
     *             res = res + 1.0j * np.random.uniform(-1, 1, size=dims)
     * return res
     */
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    py::object res = np.attr("random").attr("uniform")(
      py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    if (!dtype::is_real(dtype))
        res = res + py::cast(complex128(0, 1)) *
                      np.attr("random").attr("uniform")(
                        py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::real(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.real(a)
     */
    return wrap(np_attr("real")(obj(a)));
}

BlockPtr
NumpyBlockBackend::real_if_close(const BlockCPtr& a, float64 tol)
{
    /* converted from following python code:
     * return np.real_if_close(a, tol=tol)
     */
    return wrap(np_attr("real_if_close")(obj(a), py::arg("tol") = tol));
}

BlockPtr
NumpyBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    /* converted from following python code:
     * return np.tile(a, repeats)
     */
    return wrap(np_attr("tile")(obj(a), repeats));
}

std::vector<std::string>
NumpyBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                     const std::string& indent,
                                     int64 max_width,
                                     int64 max_lines)
{
    /* converted from following python code:
     * with np.printoptions(linewidth=max_width - len(indent)):
     *             lines = [f'{indent}{line}' for line in str(a).split('\n')]
     * if len(lines) > max_lines:
     *             first = (max_lines - 1) // 2
     *             last = max_lines - 1 - first
     *             lines = lines[:first] + [f'{indent}...'] + lines[-last:]
     * return lines
     */
    py::module_ np = numpy_module();
    py::object arr = obj(a);
    int64 old_linewidth = np_attr("get_printoptions")()["linewidth"].cast<int64>();
    np_attr("set_printoptions")(py::arg("linewidth") =
                                  max_width - static_cast<int64>(indent.size()));
    py::str s = py::str(arr);
    np_attr("set_printoptions")(py::arg("linewidth") = old_linewidth);
    py::list lines = s.attr("split")("\n");
    std::vector<std::string> out;
    int64 n = py::len(lines);
    if (n <= max_lines) {
        for (int64 i = 0; i < n; ++i)
            out.push_back(indent + lines[i].cast<std::string>());
    } else {
        int64 first = (max_lines - 1) / 2;
        int64 last = max_lines - 1 - first;
        for (int64 i = 0; i < first; ++i)
            out.push_back(indent + lines[i].cast<std::string>());
        out.push_back(indent + "...");
        for (int64 i = n - last; i < n; ++i)
            out.push_back(indent + lines[i].cast<std::string>());
    }
    return out;
}

BlockPtr
NumpyBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    /* converted from following python code:
     * return np.reshape(a, shape)
     */
    return wrap(np_attr("reshape")(obj(a), py::cast(shape)));
}

BlockPtr
NumpyBlockBackend::sqrt(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.sqrt(a)
     */
    return wrap(np_attr("sqrt")(obj(a)));
}

BlockPtr
NumpyBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    /* converted from following python code:
     * return np.squeeze(a, tuple(idcs))
     */
    // squeeze explicitly needs tuple, list is not ok
    py::tuple t(idcs.size());
    for (size_t i = 0; i < idcs.size(); ++i)
        t[i] = py::int_(idcs[i]);
    return wrap(np_attr("squeeze")(obj(a), t));
}

BlockPtr
NumpyBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    /* converted from following python code:
     * return np.where(block > cutoff, np.log(block), 0.0)
     */
    py::object arr = obj(block);
    py::module_ np = numpy_module();
    return wrap(
      np.attr("where")(np.attr("greater")(arr, py::float_(cutoff)), np_attr("log")(arr), 0.0));
}

BlockPtr
NumpyBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    /* converted from following python code:
     * return np.sum(a, axis=ax)
     */
    return wrap(np_attr("sum")(obj(a), py::arg("axis") = ax));
}

BlockBackend::Scalar
NumpyBlockBackend::sum_all(const BlockCPtr& a)
{
    /* converted from following python code:
     * return np.sum(a).item()
     */
    return Scalar(wrap(np_attr("sum")(obj(a))));
}

BlockPtr
NumpyBlockBackend::tdot(const BlockCPtr& a,
                        const BlockCPtr& b,
                        const std::vector<int64>& idcs_a,
                        const std::vector<int64>& idcs_b)
{
    /* converted from following python code:
     * return np.tensordot(a, b, (idcs_a, idcs_b))
     */
    return wrap(
      np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::cast(idcs_a), py::cast(idcs_b))));
}

BlockPtr
NumpyBlockBackend::to_dtype(const BlockCPtr& a, Dtype dtype)
{
    /* converted from following python code:
     * return np.asarray(a, dtype=self.backend_dtype_map[dtype])
     */
    return wrap(np_attr("asarray")(obj(a), dtype::to_numpy_dtype(dtype)));
}

BlockBackend::Scalar
NumpyBlockBackend::trace_full(const BlockCPtr& a)
{
    /* converted from following python code:
     * num_trace = a.ndim // 2
     * trace_dim = np.prod(a.shape[:num_trace])
     * perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
     * a = np.reshape(np.transpose(a, perm), (trace_dim, trace_dim))
     * return np.trace(a, axis1=0, axis2=1).item()
     */
    py::object arr = obj(a);
    std::vector<int64> sh = get_shape(a);
    size_t num_trace = sh.size() / 2;
    int64 trace_dim = 1;
    for (size_t i = 0; i < num_trace; ++i)
        trace_dim *= sh[i];
    std::vector<int64> perm(sh.size());
    for (size_t i = 0; i < num_trace; ++i)
        perm[i] = static_cast<int>(i);
    for (size_t i = 0; i < num_trace; ++i)
        perm[num_trace + i] = static_cast<int>(2 * num_trace - 1 - i);
    arr = np_attr("transpose")(arr, py::cast(perm));
    arr = np_attr("reshape")(arr, py::make_tuple(trace_dim, trace_dim));
    return Scalar(item(wrap(np_attr("trace")(arr, py::arg("axis1") = 0, py::arg("axis2") = 1))));
}

BlockPtr
NumpyBlockBackend::trace_partial(const BlockCPtr& a,
                                 const std::vector<int64>& idcs1,
                                 const std::vector<int64>& idcs2,
                                 const std::vector<int64>& remaining_idcs)
{
    /* converted from following python code:
     * a = np.transpose(a, remaining + idcs1 + idcs2)
     * trace_dim = np.prod(a.shape[len(remaining) : len(remaining) + len(idcs1)], dtype=int)
     * a = np.reshape(a, a.shape[: len(remaining)] + (trace_dim, trace_dim))
     * return np.trace(a, axis1=-2, axis2=-1)
     */
    py::object arr = obj(a);
    std::vector<int64> perm = remaining_idcs;
    perm.insert(perm.end(), idcs1.begin(), idcs1.end());
    perm.insert(perm.end(), idcs2.begin(), idcs2.end());
    arr = np_attr("transpose")(arr, py::cast(perm));
    std::vector<int64> sh = get_shape(a);
    int64 trace_dim = 1;
    for (int64 i : idcs1)
        trace_dim *= sh[to_valid_idx(i, static_cast<int64>(sh.size()))];
    py::array arr_arr = py::reinterpret_borrow<py::array>(arr);
    py::list new_shape;
    for (size_t i = 0; i < remaining_idcs.size(); ++i)
        new_shape.append(py::int_(arr_arr.shape(static_cast<py::ssize_t>(i))));
    new_shape.append(trace_dim);
    new_shape.append(trace_dim);
    arr = np_attr("reshape")(arr, py::tuple(new_shape));
    return wrap(np_attr("trace")(arr, py::arg("axis1") = -2, py::arg("axis2") = -1));
}

BlockPtr
NumpyBlockBackend::eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.eye(dim, dtype=self.backend_dtype_map[dtype])
     */
    (void)as_device(device);
    return wrap(np_attr("eye")(dim, py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

BlockBackend::Scalar
NumpyBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    /* converted from following python code:
     * return a[tuple(idcs)].item()
     */
    return Scalar(wrap(obj(a).attr("__getitem__")(py::tuple(py::cast(idcs)))));
}

BlockPtr
NumpyBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    /* converted from following python code:
     * return np.dot(a, b)
     */
    return wrap(np_attr("dot")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::matrix_exp(const BlockCPtr& matrix)
{
    /* converted from following python code:
     * return scipy.linalg.expm(matrix)
     */
    return wrap(py::module_::import("scipy.linalg").attr("expm")(obj(matrix)));
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    /* converted from following python code:
     * return scipy.linalg.qr(a, mode='full' if full else 'economic')
     */
    py::tuple qr = py::module_::import("scipy.linalg")
                     .attr("qr")(obj(a), py::arg("mode") = (full ? "full" : "economic"));
    return { wrap(qr[0]), wrap(qr[1]) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
{
    /* converted from following python code:
     * if algorithm is None:
     *             algorithm = 'gesdd'
     * if algorithm == 'gesdd':
     *             return scipy.linalg.svd(a, full_matrices=False)
     *         elif algorithm in ['robust', 'robust_silent']:
     *             silent = algorithm == 'robust_silent'
     *             try:
     *                 return scipy.linalg.svd(a, full_matrices=False)
     *             except np.linalg.LinAlgError:
     *                 if not silent:
     *                     raise NotImplementedError  # log warning
     *             return _svd_gesvd(a)
     *         elif algorithm == 'gesvd':
     *             return _svd_gesvd(a)
     *         else:
     *             raise ValueError(...)
     */
    std::string algo = algorithm ? *algorithm : "gesdd";
    if (algo == "gesdd") {
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a), py::arg("full_matrices") = false);
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    if (algo == "gesvd") {
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a),
                                       py::arg("full_matrices") = false,
                                       py::arg("lapack_driver") = "gesvd");
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    if (algo == "robust" || algo == "robust_silent") {
        try {
            py::tuple uvt = py::module_::import("scipy.linalg")
                              .attr("svd")(obj(a), py::arg("full_matrices") = false);
            return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
        } catch (const py::error_already_set&) {
            if (algo != "robust_silent")
                throw;
        }
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a),
                                       py::arg("full_matrices") = false,
                                       py::arg("lapack_driver") = "gesvd");
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    throw std::invalid_argument("SVD algorithm not supported: " + algo);
}

const std::vector<std::string>&
NumpyBlockBackend::possible_svd_algorithms() const
{
    static const std::vector<std::string> algorithms = {
        "gesdd", "gesvd", "robust", "robust_silent"
    };
    return algorithms;
}

BlockPtr
NumpyBlockBackend::ones_block(const std::vector<int64>& shape,
                              Dtype dtype,
                              std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.ones(shape, dtype=self.backend_dtype_map[dtype])
     */
    (void)as_device(device);
    return wrap(np_attr("ones")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

BlockPtr
NumpyBlockBackend::zeros(const std::vector<int64>& shape,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    /* converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.zeros(shape, dtype=self.backend_dtype_map[dtype])
     */
    (void)as_device(device);
    return wrap(
      np_attr("zeros")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::from_hdf5(lasyten::hdf5::Loader& loader,
                             HighFive::Group& h5gr,
                             std::string const& subpath)
{
    (void)subpath;
    auto obj = NumpyBlockBackend::from_factory_shared("cpu");
    loader.memorize_load(h5gr.getId(), std::static_pointer_cast<void>(obj));
    return obj;
}

BlockPtr
NumpyBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                          const std::vector<py::array_t<int64>>& perms)
{
    py::object a = obj(block);
    py::module_ np = numpy_module();
    py::list ix_parts;
    for (auto const& p : perms)
        ix_parts.append(p);
    py::object ix = np.attr("ix_")(*ix_parts);
    return wrap(a[ix]);
}

BlockPtr
NumpyBlockBackend::linear_combination(const Scalar& a_coef,
                                      const BlockCPtr& v,
                                      const Scalar& b_coef,
                                      const BlockCPtr& w)
{
    return wrap(a_coef.to_numpy() * obj(v) + b_coef.to_numpy() * obj(w));
}

BlockPtr
NumpyBlockBackend::mul(const Scalar& a, const BlockCPtr& b)
{
    return wrap(a.to_numpy() * obj(b));
}

BlockPtr
NumpyBlockBackend::scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis)
{
    py::object a = obj(block);
    py::object f = obj(factors);
    py::module_ np = numpy_module();
    int64 ndim = static_cast<int64>(get_shape(block).size());
    py::list idx;
    for (int64 i = 0; i < ndim; ++i)
        idx.append(i == axis ? py::object(py::slice(py::none(), py::none(), py::none()))
                             : py::object(py::none()));
    return wrap(a * f.attr("__getitem__")(py::tuple(idx)));
}

BlockPtr
NumpyBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(obj(a) * obj(b));
}

} // namespace lasyten
