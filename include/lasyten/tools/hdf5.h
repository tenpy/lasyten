#pragma once

/// Native HDF5 I/O helpers for lasyten ``src/`` (Python-free headers).

#include <tenpy_hdf5/core_io.h>
#include <tenpy_hdf5/h5_ops.h>

#include <highfive/highfive.hpp>
#include <string>

namespace lasyten::hdf5 {

using Saver = ::tenpy_hdf5::core::Saver;
using Loader = ::tenpy_hdf5::core::Loader;

inline std::string
ensure_slash(std::string path)
{
    if (path.empty() || path.back() != '/')
        path.push_back('/');
    return path;
}

/// Saver rooted at an instance group (children use short relative names).
inline Saver
local_saver(HighFive::Group& h5gr)
{
    return Saver(h5gr);
}

inline Loader
local_loader(HighFive::Group& h5gr, bool ignore_unknown = true)
{
    return Loader(h5gr, ignore_unknown);
}

} // namespace lasyten::hdf5
