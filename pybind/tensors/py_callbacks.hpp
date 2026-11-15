#pragma once

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/sector.h>

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

inline py::dict
copy_kwargs(py::object obj)
{
    if (obj.is_none()) {
        return py::dict();
    }
    return py::dict(obj);
}

inline py::tuple
shape_as_tuple(std::vector<int64> const& shape)
{
    py::tuple t(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        t[i] = py::int_(shape[i]);
    }
    return t;
}

/// Wrap ``func(shape, **kwargs)`` / ``func(**{shape_kw: shape, **kwargs})`` as a C++ block
/// factory.
inline BlockFactoryFn
block_factory_from_python(py::function func,
                          py::object func_kwargs,
                          std::optional<std::string> shape_kw,
                          std::shared_ptr<BlockBackend> bb,
                          std::optional<Dtype> dtype,
                          std::optional<std::string> device)
{
    py::dict kwargs = copy_kwargs(func_kwargs);
    py::object shape_kw_obj = shape_kw.has_value() ? py::cast(*shape_kw) : py::none();
    return [func, kwargs, shape_kw_obj, bb, dtype, device](std::vector<int64> const& shape) {
        py::object block;
        auto shape_t = shape_as_tuple(shape);
        if (shape_kw_obj.is_none()) {
            block = func(shape_t, **kwargs);
        } else {
            py::dict call_kwargs = py::dict(kwargs);
            call_kwargs[shape_kw_obj] = shape_t;
            block = func(**call_kwargs);
        }
        return bb->as_block(block, dtype, device);
    };
}

/// Wrap ``func(shape, coupled, **kwargs) -> BlockLike`` as a C++ sector-block factory.
inline SectorBlockFactoryFn
sector_block_factory_from_python(py::function func,
                                 py::object func_kwargs,
                                 std::shared_ptr<BlockBackend> bb,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    py::dict kwargs = copy_kwargs(func_kwargs);
    return
      [func, kwargs, bb, dtype, device](std::vector<int64> const& shape, Sector const& coupled) {
          py::object block = func(shape_as_tuple(shape), py::cast(coupled), **kwargs);
          return bb->as_block(block, dtype, device);
      };
}

/// Wrap a Python ``func(shape, coupled) -> Block`` (already a Block, no as_block).
inline SectorBlockFactoryFn
sector_block_factory_from_python(py::function func)
{
    return [func](std::vector<int64> const& shape, Sector const& coupled) {
        return func(shape_as_tuple(shape), py::cast(coupled)).cast<BlockBackend::BlockPtr>();
    };
}

inline BlockUnaryFn
block_unary_from_python(py::function func, py::object func_kwargs = py::none())
{
    py::dict kwargs = copy_kwargs(func_kwargs);
    return [func, kwargs](BlockBackend::BlockPtr const& block) {
        return func(py::cast(block), **kwargs).cast<BlockBackend::BlockPtr>();
    };
}

inline BlockBinaryFn
block_binary_from_python(py::function func, py::object func_kwargs = py::none())
{
    py::dict kwargs = copy_kwargs(func_kwargs);
    return [func, kwargs](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return func(py::cast(a), py::cast(b), **kwargs).cast<BlockBackend::BlockPtr>();
    };
}

inline BlockToScalarFn
block_to_scalar_from_python(py::function func)
{
    return [func](BlockBackend::BlockPtr const& block) {
        return func(py::cast(block)).cast<BlockBackend::Scalar>();
    };
}

inline ScalarReduceFn
scalar_reduce_from_python(py::function func)
{
    return [func](std::vector<BlockBackend::Scalar> const& xs) {
        return func(py::cast(xs)).cast<BlockBackend::Scalar>();
    };
}

inline std::optional<DtypeMapFn>
dtype_map_from_python(py::object dtype_map)
{
    if (dtype_map.is_none()) {
        return std::nullopt;
    }
    return [dtype_map](Dtype d) { return dtype_map(py::cast(d)).cast<Dtype>(); };
}

/// Adapt a numpy-oriented bool function so backends can call it on Block objects.
inline BlockUnaryFn
adapt_block_bool_unary(py::function func, std::shared_ptr<BlockBackend> bb)
{
    return [func, bb](BlockBackend::BlockPtr const& block) {
        auto arr = bb->to_numpy(block, py::module_::import("builtins").attr("bool"));
        auto out = func(arr);
        return bb->as_block(out, Dtype::Bool, block->device());
    };
}

inline BlockBinaryFn
adapt_block_bool_binary(py::function func, std::shared_ptr<BlockBackend> bb)
{
    return [func, bb](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        auto arr_a = bb->to_numpy(a, py::module_::import("builtins").attr("bool"));
        auto arr_b = bb->to_numpy(b, py::module_::import("builtins").attr("bool"));
        auto out = func(arr_a, arr_b);
        return bb->as_block(out, Dtype::Bool, a->device());
    };
}

} // namespace lasyten
