Installation instructions
=========================

.. todo ::
   This doesn't work yet, we still need to set this up, once we have the first (beta) release.
   Meanwhile, install the build requirements (including PyTorch; see below) and run
   ``pip install --no-build-isolation .`` from the top folder of the repo.

With the `conda package manager <https://docs.conda.io>`_ you can install python with::

    conda install --channel=conda-forge lasyten  # TODO doesn't work yet

If you don't have conda, but you have `pip <https://pip.pypa.io>`_, you can::

    pip install lasyten   # TODO doesn't work yet

Building from source
++++++++++++++++++++

To build lasyten locally on your machine, install the following requirements
(currently only tested on standard linux distros like ubuntu, MacOS specific notes see below,
no Windows support yet, use WSL):

- C++ compiler with at least C++17 standard. In a conda env this **must** be
  conda-forge ``cxx-compiler`` (included in ``environment.yml``), not a newer
  system ``g++``. Python already loads conda's ``libstdc++``; objects compiled
  with a newer GCC fail at ``import lasyten`` with missing ``GLIBCXX_*`` symbols.
- CMake, make
- Python >= 3.12, with numpy>=2.0, scipy and a few other python packages as listed in ``environment.yml``
- **PyTorch** (``torch`` / conda-forge ``pytorch``): required at build and runtime.
  CMake finds libtorch via ``torch.utils.cmake_prefix_path`` and links ``lasyten._core`` against
  the same shared libraries as ``import torch``, so PyTorch must be installed in the build
  environment before compiling.
  Avoid a plain isolated ``pip install pytorch`` against PyPI ``torch`` on machines without the
  CUDA toolkit: the default Linux wheels are CUDA-enabled and CMake's ``find_package(Torch)``
  then fails looking for CUDA.
  Prefer conda-forge ``pytorch`` (CPU) or install a CPU wheel, e.g.
  ``pip install torch --index-url https://download.pytorch.org/whl/cpu``.
- scikit-build

The easiest way to install all of those is to create a conda environment from the ``environment.yml``
(which includes PyTorch) and then pip-install the package
(use ``docs/environment.yml`` if you plan to build the documentation as well)::

    conda env create -f environment.yml -n lasyten
    conda activate lasyten
    conda install -c conda-forge _openmp_mutex=*=*_llvm # on Linux/WSL only
    pip install -v --no-build-isolation .

Use ``--no-build-isolation`` so the build sees the conda-installed PyTorch (and other build
deps) instead of resolving them in an isolated pip environment.

If needed, you can add defines for the CMake build as options to pip, e.g. ``pip install -v -C cmake.define.=ON .``.


For a debug build, you can even enable automatic rebuild upon python import::

    pip install -v --no-build-isolation -C editable.rebuild=true -e .

CMake options for the block backends and a Python-free build
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``LASYTEN_WITH_NUMPY_BACKEND`` (default ON if ``LASYTEN_WITH_PYTHON`` is ON)
    Build the transitional Python-dispatch backends ``NumpyBlockBackend`` and ``ArrayApiBlockBackend``.
    The preferred CPU backend is the native ``EigenBlockBackend`` (factory name ``"eigen"``), and
    ``TorchBlockBackend`` is for GPUs. With ``-DLASYTEN_WITH_NUMPY_BACKEND=OFF`` the two Python-dispatch
    backends are neither compiled nor exposed in ``lasyten``, and ``"cpu"`` resolves to Eigen::

        pip install -v --no-build-isolation -C cmake.define.LASYTEN_WITH_NUMPY_BACKEND=OFF .

``LASYTEN_WITH_PYTHON`` (default ON)
    The long-term goal is to be able to use the C++ library without any Python interpreter.
    With ``-DLASYTEN_WITH_PYTHON=OFF`` CMake does not look for Python, nanobind, numpy or PyTorch, and
    builds ``liblasyten`` from the Python-free sources: ``EigenBlockBackend``, the symmetry stack,
    ``Space`` / ``Leg``, the Tensor classes, ``NoSymmetryBackend`` / ``AbelianBackend`` /
    ``FusionTreeBackend`` (core paths only; mask/truncate/partial_trace and related OFF stubs are
    listed in ``docs/cpp_conversion/convert_FusionTreeBackend.md``), krylov, and native
    decompositions -- but no ``_core`` extension and no models / Numpy/ArrayApi/Torch backends.
    CTests ``test_eigen_block_backend_no_python``, ``test_symmetric_blocks_no_python``, and
    ``test_tensor_no_python`` run without an interpreter (``ldd`` shows no ``libpython``). Needs
    only a C++ compiler, CMake, Eigen3, LAPACK/LAPACKE and HDF5 (e.g. from the conda env)::

        cmake -S . -B /tmp/lasyten_off -G Ninja -DLASYTEN_WITH_PYTHON=OFF
        cmake --build /tmp/lasyten_off
        ctest --test-dir /tmp/lasyten_off --output-on-failure

    The configure step prints which parts are skipped.
    See ``docs/cpp_conversion/convert_nanobind.md`` (Phase 5 / leftover #1) for the status.

``LASYTEN_STABLE_ABI`` (default ON)
    Build ``lasyten._core`` (and liblasyten) against the Python Stable ABI: nanobind's ``STABLE_ABI`` mode with
    ``Py_LIMITED_API`` = 3.12, linked against ``nanobind-static-abi3``. The extension is then called
    ``_core.abi3.so`` and the wheel is tagged ``cp312-abi3`` (``wheel.py-api = "cp312"`` in ``pyproject.toml``),
    i.e. it works for every CPython >= 3.12. The option is silently turned OFF for Python < 3.12 or a
    non-CPython interpreter (then ``_core.cpython-3XX-....so`` is built). pybind11 is not used anymore: it
    cannot be compiled with ``Py_LIMITED_API``.


Building from source on MacOS
+++++++++++++++++++++++++++++
Building on MacOS is not carefully tested yet, but Jakob got it running on an M2 MacBook running
Sequoia Version 15.7.7.

In addition to the above instructions, do the following:

- Make sure you have an installation of ninja, e.g. via ``brew install ninja``.
- After creating the conda env, also install ``conda install -c conda-forge gxx gcc llvm-openmp`` before building
- The compile only went through with Clang, e.g. when the log shows
  ``The CXX compiler identification is Clang 21.1.8``.
  On Ubuntu, this would show GNU. For me, this did not need any extra steps.
- This working setup using Clang did not support all features that the GNU/Ubuntu setup did,
  e.g. ``std::numbers`` was not available. In this particular case, we could simply get rid of that
  dependency, but future updates may again introduce features that are not supported by the Clang/MacOS toolchain.

Hosting and packaging cutover (manual)
++++++++++++++++++++++++++++++++++++++

These steps are outside the git tree and must be done when renaming from the old
``cyten`` project:

1. **GitHub:** Rename ``tenpy/cyten`` → ``tenpy/lasyten`` (Settings → Rename).
   In-repo URLs already point at ``https://github.com/tenpy/lasyten``.
2. **Read the Docs:** Create or rename the project slug to ``lasyten`` so docs
   build at ``https://lasyten.readthedocs.io/``. Point the project at
   ``tenpy/lasyten`` and confirm the webhook; builds use ``.readthedocs.yml``
   and ``docs/environment.yml``. Retire any old ``cyten`` RTD project.
3. **PyPI (soon):** Reserve the name ``lasyten`` with a temporary source-only
   sdist (alpha version). Do not publish under ``cyten``.
4. **conda-forge (later):** Add a ``lasyten`` feedstock once a real release
   exists (build deps match ``environment.yml``, including PyTorch).
5. **Related repos:** Update TeNPy ``v2_main`` (submodule URL/path and imports)
   and, when convenient, example mentions of ``cyten`` in ``tenpy/hdf5_io``.
