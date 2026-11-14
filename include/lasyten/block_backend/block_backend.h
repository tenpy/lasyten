#pragma once

#include <iosfwd>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

class Leg; // for apply_basis_perm; defined in symmetries/spaces.h

/// Abstract base class that defines the operation on dense blocks.
class BlockBackend
{
  public:
    // forward declarations, defined below.
    class Block;
    class Scalar;

    using BlockPtr = std::shared_ptr<Block>;
    using BlockCPtr = std::shared_ptr<const Block>;
    /// Value-type leg handle (same as `lasyten::Leg` / `Leg::CPtr`).
    using LegCPtr = Leg;

    /// C++ slice for one axis (``std::nullopt`` = open end / default step), like Python
    /// ``slice(start, stop, step)``.
    struct AxisSlice
    {
        std::optional<int64> start{};
        std::optional<int64> stop{};
        std::optional<int64> step{};
        static AxisSlice all() { return {}; }
    };

    /// One axis indexer for native C++ ``get_item`` / ``set_item`` (no ``py::object``).
    /// Semantics match NumPy/torch basic + advanced indexing for a tuple of one indexer per axis
    /// (no Ellipsis / newaxis in this API).
    using BlockIndex = std::variant<int64,              // integer index (collapses that axis)
                                    AxisSlice,          // basic slice
                                    std::vector<int64>, // host index array
                                    BlockCPtr // index-array or bool-mask Block (device-aware)
                                    >;

    /// Convert a native indexer to a Python key object (for numpy / Array API bridges).
    static py::object block_index_to_py(const BlockIndex& idx);
    /// Convert a sequence of native indexers to a Python key (single object or tuple).
    static py::object block_indices_to_py(std::span<const BlockIndex> key);
    /// Try to parse a Python key into native ``BlockIndex`` entries. Returns nullopt for
    /// unsupported keys (Ellipsis, None/newaxis, plain lists, etc.).
    static std::optional<std::vector<BlockIndex>> try_py_key_to_block_indices(py::object key);

  public:
    /// Abstract base class for dense blocks. Subclassed per backend (e.g.
    /// NumpyBlockBackend::Block). Access to elements should be done exclusively through the
    /// BlockBackend.
    class Block : public std::enable_shared_from_this<Block>
    {
      public:
        // subclasses should have constructor from numpy array
        // explicit Block(py::array arr);
        // and

        virtual ~Block() = default;

        /// Return the backend for this block's device (e.g.
        /// NumpyBlockBackend::from_factory(device())).
        virtual BlockBackend* get_backend() const = 0;
        /// convert to numpy array, might be copy or (immutable) view
        virtual py::array to_numpy() const = 0;
        /// convert to numpy array with given Dtype
        // default impl: `to_numpy()` then `asarray(..., dtype))`.
        virtual py::array to_numpy(Dtype dtype) const;

        /// Shape of the block (one size per axis).
        virtual std::vector<int64> shape() const = 0;

        /// Number of axes/dimensions of the block = shape().size()
        virtual int64 ndim() const;

        /// Dtype of the block entries.
        virtual Dtype dtype() const = 0;

        /// Device string (e.g. "cpu", "cuda:0").
        virtual const std::string& device() const = 0;

        /// Elementwise addition with another block.
        virtual std::shared_ptr<Block> operator+(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator-(const Block& other) const = 0;
        /// Elementwise multiplication with another block.
        virtual std::shared_ptr<Block> operator*(const Block& other) const = 0;
        /// Elementwise division with another block.
        virtual std::shared_ptr<Block> operator/(const Block& other) const = 0;
        /// Elementwise comparisons with another block.
        virtual std::shared_ptr<Block> operator<(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator<=(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator>(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator>=(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator==(const Block& other) const = 0;
        virtual std::shared_ptr<Block> operator!=(const Block& other) const = 0;
        /// Scalar multiplication.
        std::shared_ptr<Block> operator*(const Scalar& s) const;
        /// Multiplication by inverse of a scalar.
        std::shared_ptr<Block> operator/(const Scalar& s) const;
        /// Elementwise absolute value (Python ``__abs__`` / ``abs(block)``).
        std::shared_ptr<Block> abs() const;
        // Elementwise power
        virtual std::shared_ptr<Block> pow(const Scalar& exponent) const = 0;
        virtual std::shared_ptr<Block> pow(const Block& exponent) const = 0;

        /// Arbitrary access by Python key; returns new block (shared_ptr).
        std::shared_ptr<const Block> operator[](py::object key) const;
        std::shared_ptr<Block> operator[](py::object key);
        /// Native C++ subblock access (slices / index arrays); does not route through py::object.
        std::shared_ptr<const Block> operator[](std::span<const BlockIndex> key) const;
        std::shared_ptr<Block> operator[](std::span<const BlockIndex> key);
        std::shared_ptr<const Block> operator[](std::initializer_list<BlockIndex> key) const;
        std::shared_ptr<Block> operator[](std::initializer_list<BlockIndex> key);

        /// Assign to whole block; uses native full-slice ``set_item``.
        Block& operator=(py::object rhs);

        /// Arbitrary getitem; implemented by backends (e.g. numpy __getitem__).
        virtual std::shared_ptr<Block> get_item(py::object key) = 0;
        virtual std::shared_ptr<const Block> get_item(py::object key) const = 0;
        /// Native C++ getitem by slices / ints / index arrays (one indexer per axis).
        virtual std::shared_ptr<Block> get_item(std::span<const BlockIndex> key) = 0;
        virtual std::shared_ptr<const Block> get_item(std::span<const BlockIndex> key) const = 0;
        /// Get a single element by integer multi-index; returns a Scalar.
        /// Default: delegates to BlockBackend::get_block_element.
        virtual Scalar get_item(const std::vector<int64>& key) const;
        /// Get a single element by integer index for 1D blocks; returns a Scalar.
        /// Default: delegates to get_item(std::vector{idx}) after checking ndim()==1.
        virtual Scalar get_item(int64 idx) const;
        /// Arbitrary setitem; implemented by backends (e.g. numpy __setitem__).
        virtual void set_item(py::object key, py::object value) = 0;
        /// Native C++ setitem by slices / ints / index arrays.
        virtual void set_item(std::span<const BlockIndex> key, const Block& value) = 0;
        /// Native C++ setitem from a Scalar value.
        virtual void set_item(std::span<const BlockIndex> key, const Scalar& value);
        /// Set a single element by integer multi-index from a Scalar.
        /// Default: set_item(tuple(key), value.to_numpy()).
        virtual void set_item(const std::vector<int64>& key, const Scalar& value);
        /// Set a single element by integer index for 1D blocks from a Scalar.
        /// Default: set_item(std::vector{idx}, value) after checking ndim()==1.
        virtual void set_item(int64 idx, const Scalar& value);
        // implicit conversion to Scalar, throws for Blocks which are not 0-D.
        virtual operator Scalar() { return Scalar(shared_from_this()); };
        friend class Scalar;
        /// Return the element of a zero-dimensional block. Raise if not 0-D
        virtual complex128 _item_as_complex128() const = 0;
        /// float etc can be cast to complex128 without loss, but not int64
        virtual int64 _item_as_int64() const = 0;

        /// Save block state to HDF5 (subclass implements payload).
        virtual void save_hdf5(lasyten::hdf5::Saver& saver,
                               HighFive::Group& h5gr,
                               std::string const& subpath) = 0;
        /// Load block from HDF5. Subclasses must override; base throws NotImplemented.
        static std::shared_ptr<Block> from_hdf5(lasyten::hdf5::Loader& loader,
                                                HighFive::Group& h5gr,
                                                std::string const& subpath);
    };

    /// Holds a single scalar value with a Dtype as a 0-d Block.
    // Use accessors to cast to the desired C++ type.
    class Scalar
    {
      public:
        /// (implicitly) convert from a block with trivial empty shape (ndim == 0).
        /// Throws if ndim != 0.
        Scalar(std::shared_ptr<Block> block);

        Dtype dtype() const { return block_->dtype(); }

        /// As a real (float64) scalar. Throws if dtype is not Float32 or Float64.
        float64 as_float64() const;
        /// As a float32 scalar. Throws if dtype is not Float32.
        float32 as_float32() const;
        /// As a complex64 scalar. Throws if dtype is not Complex64.
        complex64 as_complex64() const;
        /// As a complex128 scalar. Always valid (real/bool stored with zero imaginary part).
        complex128 as_complex128() const;
        /// As a bool. Throws if dtype is not Bool.
        int64 as_int64() const;
        /// As a bool scalar. Throws if dtype is not Int64.
        bool as_bool() const;
        /// Return as a numpy scalar (``np.bool_``, ``np.float32``, ``np.float64``,
        /// ``np.complex64``,
        /// ``np.complex128``).
        py::object to_numpy() const;

        Scalar operator+(const Scalar& other) const;
        Scalar operator-() const;
        Scalar operator-(const Scalar& other) const;
        Scalar operator*(const Scalar& other) const;
        Scalar operator/(const Scalar& other) const;
        Scalar operator+(float64 other) const;
        Scalar operator-(float64 other) const;
        Scalar operator*(float64 other) const;
        Scalar operator/(float64 other) const;
        Scalar operator+(complex128 other) const;
        Scalar operator-(complex128 other) const;
        Scalar operator*(complex128 other) const;
        Scalar operator/(complex128 other) const;

        Scalar operator<(const Scalar& other) const;
        Scalar operator>(const Scalar& other) const;
        Scalar operator<=(const Scalar& other) const;
        Scalar operator>=(const Scalar& other) const;
        Scalar operator<(float64 other) const;
        Scalar operator>(float64 other) const;
        Scalar operator<=(float64 other) const;
        Scalar operator>=(float64 other) const;
        /// The inverse of the scalar, 1./self
        Scalar inverse() const;

        /// convenience access for further methods, delegating to block_backend
        Scalar real() const;
        Scalar imag() const;
        /// Complex conjugate. Real dtypes are returned unchanged.
        Scalar conj() const;
        Scalar abs() const;
        Scalar sqrt() const;
        /// If close to real, return the real part; otherwise return `*this`.
        ///
        /// `tol` is in multiples of machine epsilon for the dtype (same convention as
        /// `numpy.real_if_close`; default `100`).
        Scalar real_if_close(float64 tol = 100.) const;
        /// The *elementwise* exponential.
        ///
        /// Not to be confused with `matrix_exp`, the *matrix* exponential.
        Scalar exp() const;
        /// The *elementwise* natural logarithm.
        ///
        /// Not to be confused with the matrix logarithm (not implemented).
        Scalar log() const;
        Scalar pow(const Scalar& exponent) const;

        std::shared_ptr<const Block> _block() const;

        /// Save scalar (via underlying 0-d block) to HDF5.
        void save_hdf5(lasyten::hdf5::Saver& saver,
                       HighFive::Group& h5gr,
                       std::string const& subpath);
        /// Load scalar from HDF5.
        static Scalar from_hdf5(lasyten::hdf5::Loader& loader,
                                HighFive::Group& h5gr,
                                std::string const& subpath);

      private:
        std::shared_ptr<Block> block_;
    };

  public:
    virtual Scalar as_scalar(complex128 value, Dtype dtype) = 0;
    virtual Scalar as_scalar(py::object value, Dtype dtype) = 0;
    virtual Scalar as_scalar(const Scalar& value);
    virtual Scalar as_scalar(bool b) = 0;
    virtual Scalar as_scalar(int64 x) = 0;
    virtual Scalar as_scalar(float32 x) = 0;
    virtual Scalar as_scalar(float64 x) = 0;
    virtual Scalar as_scalar(complex64 z) = 0;
    virtual Scalar as_scalar(complex128 z) = 0;

  public:
    std::string default_device;

  public:
    /// Get the backend instance for the given device. Implemented in subclasses (e.g.
    /// NumpyBlockBackend).
    static BlockBackend* from_factory(std::string device = "cpu");

  public:
    /// Public constructor for Python Subclasses.
    /// C++ Subclasses should have a protected constructor and enforce instantiation via
    /// from_factory.
    explicit BlockBackend(std::string default_device);

  public:
    virtual ~BlockBackend() = default;

    /// Name of the backend class for __repr__ / __str__ (e.g. "NumpyBlockBackend").
    virtual std::string get_backend_name() const;

    /// Semantic equality: same backend class and equivalent instance state (e.g. device).
    /// Subclasses with extra state (e.g. Array API namespace) should override.
    [[nodiscard]] virtual bool operator==(BlockBackend const& other) const;
    bool operator!=(BlockBackend const& other) const { return !(*this == other); }

    /// The absolute value of a complex number, elementwise.
    virtual BlockPtr abs(const BlockCPtr& a) = 0;
    /// Apply `basis_perm` (or its inverse) on every axis.
    ///
    /// Apply ``basis_perm`` of a `Leg` (or its inverse)
    /// on every axis of a dense block.
    BlockPtr apply_basis_perm(const BlockCPtr& block,
                              const std::vector<LegCPtr>& legs,
                              bool inv = false);
    /// Apply permutations to every axis of a dense block
    virtual BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                            const std::vector<py::array_t<int64>>& perms) = 0;
    /// Convert objects to blocks.
    ///
    /// Should support blocks, numpy arrays, nested python containers. May support more.
    /// If `a` is already a block of correct dtype on the correct device, it may be returned
    /// un-modified.
    ///
    /// @returns block: Block The new block
    ///
    /// block_copy
    ///     Guarantees an independent copy.
    virtual BlockPtr as_block(py::object a,
                              std::optional<Dtype> dtype = std::nullopt,
                              std::optional<std::string> device = std::nullopt) = 0;
    /// Convert input string to unambiguous device name.
    ///
    /// In particular, this should map any possible aliases to one unique name, e.g.
    /// for PyTorch, map ``'cuda'`` to ``'cuda:0'``.
    /// Also checks if that device is valid and available.
    virtual std::string as_device(std::optional<std::string> device) = 0;
    /// Return the indices (one per axis) of the largest entry (by magnitude) of the block
    virtual std::vector<int64> abs_argmax(const BlockCPtr& block) = 0;
    /// Return the indices (one per axis) of the smallest entry of the block.
    /// Requires a real dtype. On ties, the first occurrence (C-order flatten).
    virtual std::vector<int64> argmin(const BlockCPtr& block) = 0;
    virtual BlockPtr add_axis(const BlockCPtr& a, int64 pos) = 0;
    /// Require a boolean block. If all of its entries are True
    virtual bool all(const BlockCPtr& a) = 0;
    virtual bool allclose(const BlockCPtr& a,
                          const BlockCPtr& b,
                          float64 rtol = 1e-5,      // NOLINT(readability-magic-numbers)
                          float64 atol = 1e-8) = 0; // NOLINT(readability-magic-numbers)
    /// The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.
    virtual BlockPtr angle(const BlockCPtr& a) = 0;
    /// Require a boolean block. If any of its entries are True
    virtual bool any(const BlockCPtr& a) = 0;
    /// Apply a mask (1D boolean block) to a block, slicing/projecting that axis
    virtual BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) = 0;
    /// Return the permutation that would sort a block along one axis.
    ///
    /// @param block The block to sort.
    /// @param sort Specify how the arguments should be sorted.  ====================
    /// ============================= `sort`               order ====================
    /// ============================= ``'m>', 'LM'``       Largest magnitude first
    /// -------------------- ----------------------------- ``'m<', 'SM'``       Smallest magnitude
    /// first -------------------- ----------------------------- ``'>', 'LR', 'LA'``  Largest real
    /// part first -------------------- ----------------------------- ``'<', 'SR', 'SA'``  Smallest
    /// real part first -------------------- ----------------------------- ``'LI'`` Largest
    /// imaginary part first -------------------- ----------------------------- ``'SI'`` Smallest
    /// imaginary part first ==================== =============================
    /// @param axis The axis along which to sort
    /// @returns The indices that would sort the block
    BlockPtr argsort(const BlockCPtr& block,
                     std::optional<std::string> sort = std::nullopt,
                     int64 axis = 0);
    /// Like `argsort` but can assume real valued block, and sort ascending
    virtual BlockPtr _argsort(const BlockCPtr& block, int64 axis) = 0;
    /// Combine each group of legs in `leg_idcs_combine` into a single leg.
    ///
    /// The group of legs in each entry of `leg_idcs_combine` must be contiguous.
    /// The legs can be combined in C style (default) or F style; the style can
    /// be specified for each group of legs independently.
    BlockPtr combine_legs(const BlockCPtr& a,
                          const std::vector<std::vector<int64>>& leg_idcs_combine,
                          const std::vector<bool>& cstyles);
    BlockPtr combine_legs(const BlockCPtr& a,
                          const std::vector<std::vector<int64>>& leg_idcs_combine,
                          bool cstyles = true);
    /// Complex conjugate of a block
    virtual BlockPtr conj(const BlockCPtr& a) = 0;
    /// Create a new, independent block with the same data
    ///
    /// @param a The block to copy
    /// @param device The device for the new block. Per default, use the same device as the old
    /// block. as_block
    ///     Function to guarantee dtype and device, without forcing copies.
    virtual BlockPtr copy_block(const BlockCPtr& a,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    virtual BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) = 0;
    /// Permute axes to reverse order and elementwise conj.
    BlockPtr dagger(const BlockCPtr& a);
    Dtype get_dtype(const BlockCPtr& a);
    /// Eigenvalue decomposition of a 2D hermitian block.
    ///
    /// Return a 1D block of eigenvalues and a 2D block of eigenvectors
    ///
    /// @param block The block to decompose
    /// @param sort How the eigenvalues are sorted
    virtual std::tuple<BlockPtr, BlockPtr> eigh(
      const BlockCPtr& block,
      std::optional<std::string> sort = std::nullopt) = 0;
    /// Eigenvalue decomposition of a general 2D square block.
    ///
    /// Return a 1D block of (generally complex) eigenvalues and a 2D block of eigenvectors.
    ///
    /// @param block The block to decompose
    /// @param sort How the eigenvalues are sorted
    virtual std::tuple<BlockPtr, BlockPtr> eig(const BlockCPtr& block,
                                               std::optional<std::string> sort = std::nullopt) = 0;
    /// Eigenvalues of a 2D hermitian block.
    ///
    /// Return a 1D block of eigenvalues
    ///
    /// @param block The block to decompose
    /// @param sort How the eigenvalues are sorted
    virtual BlockPtr eigvalsh(const BlockCPtr& block,
                              std::optional<std::string> sort = std::nullopt) = 0;
    /// Eigenvalues of a general 2D square block.
    ///
    /// Return a 1D block of (generally complex) eigenvalues.
    ///
    /// @param block The block to decompose
    /// @param sort How the eigenvalues are sorted
    virtual BlockPtr eigvals(const BlockCPtr& block,
                             std::optional<std::string> sort = std::nullopt) = 0;
    virtual BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) = 0;
    /// The *elementwise* exponential.
    virtual BlockPtr exp(const BlockCPtr& a) = 0;
    /// Return a 2D square block that has the 1D ``diag`` on the diagonal
    virtual BlockPtr block_from_diagonal(const BlockCPtr& diag) = 0;
    /// Convert a mask to a full block.
    ///
    /// Return a (N, M) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
    /// where N is the number of True entries. The result is the coefficient matrix of the
    /// projection map.
    virtual BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) = 0;
    virtual BlockPtr block_from_numpy(const py::array& a,
                                      std::optional<Dtype> dtype = std::nullopt,
                                      std::optional<std::string> device = std::nullopt) = 0;
    const std::string& get_device(const BlockCPtr& a);
    /// Get the diagonal of a 2D block as a 1D block
    virtual BlockPtr get_diagonal(const BlockCPtr& a,
                                  std::optional<float64> tol = std::nullopt) = 0;
    /// The imaginary part of a complex number, elementwise.
    virtual BlockPtr imag(const BlockCPtr& a) = 0;
    /// Dense block version of tensors.inner.
    ///
    /// If do dagger, ``sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])``
    /// otherwise, ``sum(a[i1, ..., iN] * b[iN, ..., i2, i1])``.
    virtual Scalar inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger);
    /// If the block is comprised of real numbers.
    ///
    /// Complex numbers with small or zero imaginary part still cause a `False` return.
    bool is_real(const BlockCPtr& a);
    /// Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python
    /// float or complex
    virtual Scalar item(const BlockCPtr& a) = 0;
    /// The kronecker product.
    ///
    /// @param a, b Two blocks with the same number of dimensions.
    ///
    /// Notes:
    ///
    /// The elements are products of elements from `a` and `b`::
    ///     kron(a, b)[k0, k1, ..., kN] = a[i0, i1, ..., iN] * b[j0, j1, ..., jN]
    ///
    /// where::
    ///     kt = it * st + jt,  t = 0,...,N
    ///
    /// (Taken from numpy docs)
    virtual BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr linear_combination(const Scalar& a_coef,
                                        const BlockCPtr& v,
                                        const Scalar& b_coef,
                                        const BlockCPtr& w);
    /// The *elementwise* natural logarithm.
    virtual BlockPtr log(const BlockCPtr& a) = 0;
    virtual Scalar max(const BlockCPtr& a) = 0;
    virtual Scalar max_abs(const BlockCPtr& a) = 0;
    virtual Scalar min(const BlockCPtr& a) = 0;
    virtual BlockPtr mul(const Scalar& a, const BlockCPtr& b);
    virtual BlockPtr mul(float64 a, const BlockCPtr& b);
    virtual BlockPtr mul(complex128 a, const BlockCPtr& b);
    /// The p-norm vector-norm of a block.
    ///
    /// @param a The block.
    /// @param order The order @f$ p @f$ of the norm. Unlike numpy, we always compute vector norms,
    /// never matrix norms. We only support p-norms @f$ \Vert x \Vert = \sqrt[p]{\sum_i
    /// \abs{x_i}^p} @f$.
    /// @param axis ``axis=None`` means "all axes", i.e. norm of the flattened block. An integer
    /// means to broadcast the norm over all other axes.
    virtual Scalar norm(const BlockCPtr& a,
                        float64 order = 2,
                        std::optional<int64> axis = std::nullopt) = 0;
    /// Outer product of blocks.
    ///
    /// ``res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM]``
    virtual BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) = 0;
    /// For a matrix `a` with two combined multi-indices, permute the sub-indices.
    ///
    /// @param block A matrix with combined axes ``[(m1.m2...mJ), (n1.n2...nK)]``.
    /// @param dims1 The dimensions of the subindices ``[m1, m2, ..., mJ]``.
    /// @param idcs1 Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the
    /// first multi-index of the result.
    /// @param dims2 The dimensions of the subindices ``[n1, n2, ..., nK]``.
    /// @param idcs2 Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the
    /// second multi-index of the result.
    /// @returns A matrix with the same entries as `a`, but rearranged to the new axis order, e.g.
    /// ``[M, N]``, where ``M == combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs1])`` and ``N ==
    /// combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs2])``.
    ///
    /// permute_combined_idx
    BlockPtr permute_combined_matrix(const BlockCPtr& block,
                                     const std::vector<int64>& dims1,
                                     const std::vector<int64>& idcs1,
                                     const std::vector<int64>& dims2,
                                     const std::vector<int64>& idcs2);
    /// For a matrix `a` with a single combined multi-index, permute sub-indices.
    ///
    /// @param block A matrix with axes ``[M, N]``, where either ``M = (m1.m2...mJ)`` or ``N =
    /// (n1.n2...nK)`` is a multi-index *but not both*.
    /// @param axis Which of the two axes has the multi-indices
    /// @param dims The dimensions of the sub-indices, e.g. ``[m1, m2, ..., mJ]``.
    /// @param idcs The order of the sub-indices in the results, such that the result has axes
    /// ``[[m1, m2, ..., mJ][i] for i in idcs]``.
    /// @returns A matrix with the same entries as `a`, but rearranged to the new axis order, i.e.
    /// ``[M_new, N_new]`` where e.g. ``M_new = combined([m1, m2, ..., mJ][idcs])``.
    ///
    /// permute_combined_matrix
    BlockPtr permute_combined_idx(const BlockCPtr& block,
                                  int64 axis,
                                  const std::vector<int64>& dims,
                                  const std::vector<int64>& idcs);
    virtual BlockPtr random_normal(const std::vector<int64>& dims,
                                   Dtype dtype,
                                   float64 sigma,
                                   std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr random_uniform(const std::vector<int64>& dims,
                                    Dtype dtype,
                                    std::optional<std::string> device = std::nullopt) = 0;
    /// The real part of a complex number, elementwise.
    virtual BlockPtr real(const BlockCPtr& a) = 0;
    /// If a block is close to its real part, return the real part.
    ///
    /// Otherwise the original block. Elementwise.
    virtual BlockPtr real_if_close(const BlockCPtr& a, float64 tol) = 0;
    /// Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.
    virtual BlockPtr tile(const BlockCPtr& a, int64 repeats) = 0;
    virtual std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                                       const std::string& indent,
                                                       int64 max_width,
                                                       int64 max_lines) = 0;
    virtual BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) = 0;
    /// Multiply block with the factors (a 1D block), along a given axis.
    ///
    /// E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
    /// ``block * factors[None, None, :, None]``.
    virtual BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) = 0;
    std::vector<int64> get_shape(const BlockCPtr& a);
    /// Split legs into groups of legs with specified dimensions.
    ///
    /// The splitting of a leg can be in C style (default) or F style. In the
    /// latter case, the specified dimensions of the resulting group of legs
    /// *are reversed*. The style can be specified for each group of legs
    /// independently.
    BlockPtr split_legs(const BlockCPtr& a,
                        const std::vector<int64>& idcs,
                        const std::vector<std::vector<int64>>& dims,
                        const std::vector<bool>& cstyles);
    BlockPtr split_legs(const BlockCPtr& a,
                        const std::vector<int64>& idcs,
                        const std::vector<std::vector<int64>>& dims,
                        bool cstyles = true);
    /// The elementwise square root
    virtual BlockPtr sqrt(const BlockCPtr& a) = 0;
    virtual BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) = 0;
    /// Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.
    virtual BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) = 0;
    /// The sum over a single axis.
    virtual BlockPtr sum(const BlockCPtr& a, int64 ax) = 0;
    /// The sum of all entries of the block.
    ///
    /// If the block contains boolean values, this should return the number of ``True`` entries.
    virtual Scalar sum_all(const BlockCPtr& a) = 0;
    virtual BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) = 0; // elementwise
    virtual BlockPtr tdot(const BlockCPtr& a,
                          const BlockCPtr& b,
                          const std::vector<int64>& idcs_a,
                          const std::vector<int64>& idcs_b) = 0;
    /// Version of ``tensors.outer`` on blocks.
    ///
    /// Note the different leg order to usual outer products::
    ///
    ///     res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]
    ///
    /// intended to be used with ``K == a_num_codomain_legs``.
    BlockPtr tensor_outer(const BlockCPtr& a, const BlockCPtr& b, int64 K);
    virtual BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) = 0;
    py::object to_numpy(const BlockCPtr& a, std::optional<py::object> numpy_dtype = std::nullopt);
    virtual Scalar trace_full(const BlockCPtr& a) = 0;
    virtual BlockPtr trace_partial(const BlockCPtr& a,
                                   const std::vector<int64>& idcs1,
                                   const std::vector<int64>& idcs2,
                                   const std::vector<int64>& remaining_idcs) = 0;
    /// The identity matrix, reshaped to a block.
    ///
    /// Note the unusual leg order ``[m1,...,mJ,mJ*,...,m1*]``,
    /// which is chosen to match `eye_data`.
    ///
    /// Note also that the ``legs`` only specify the dimensions of the first half,
    /// namely ``m1,...,mJ``.
    BlockPtr eye_block(const std::vector<int64>& legs,
                       Dtype dtype,
                       std::optional<std::string> device = std::nullopt);
    /// The ``dim x dim`` identity matrix
    virtual BlockPtr eye_matrix(int64 dim,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual Scalar get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) = 0;
    /// Get an element of a mask.
    ///
    /// Mask elements are `True` if the entry `a[large_leg_idx]` is the `small_leg_idx`-th `True`
    /// in the block.
    ///
    /// @param a The mask block
    /// @param large_leg_idx, small_leg_idx The block indices
    /// @param sum_block Number of `True` entries in the block, i.e., ``sum_block ==
    /// self.sum_all(a)``. Agrees with the sector multiplicity of the small leg. (Only important if
    /// the sector dimension is larger than 1.)
    virtual Scalar get_block_mask_element(const BlockCPtr& a,
                                          int64 large_leg_idx,
                                          int64 small_leg_idx,
                                          int64 sum_block = 0);
    /// As in numpy.dot, both a and b might be matrix or vector.
    virtual BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr matrix_exp(const BlockCPtr& matrix) = 0;
    std::tuple<BlockPtr, BlockPtr> matrix_lq(const BlockCPtr& a, bool full);
    /// QR decomposition of a 2D block
    virtual std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) = 0;
    /// Internal version of `matrix_svd`, to be implemented by subclasses.
    virtual std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm = std::nullopt) = 0;
    /// Possible SVD algorithms for this backend.
    virtual const std::vector<std::string>& possible_svd_algorithms() const = 0;
    virtual BlockPtr ones_block(const std::vector<int64>& shape,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// Wait for asynchronous processes (if any) to finish
    void synchronize();
    /// Assert block type and optional shape/dtype/device. Throws std::runtime_error if any check
    /// fails.
    void test_block_sanity(const BlockCPtr& block,
                           std::optional<std::vector<int64>> expect_shape = std::nullopt,
                           std::optional<Dtype> expect_dtype = std::nullopt,
                           std::optional<std::string> expect_device = std::nullopt);
    virtual BlockPtr zeros(const std::vector<int64>& shape,
                           Dtype dtype,
                           std::optional<std::string> device = std::nullopt) = 0;

    /// Save backend state to HDF5.
    void save_hdf5(lasyten::hdf5::Saver& saver, HighFive::Group& h5gr, std::string const& subpath);
    /// Load backend from HDF5.
    static std::shared_ptr<BlockBackend> from_hdf5(lasyten::hdf5::Loader& loader,
                                                   HighFive::Group& h5gr,
                                                   std::string const& subpath);

  protected:
    /// Return true if block is of the backend's block type. Used by test_block_sanity.
    virtual bool is_correct_block_type(const BlockCPtr& block) const = 0;
};

using BlockPtr = std::shared_ptr<BlockBackend::Block>;
using BlockCPtr = std::shared_ptr<const BlockBackend::Block>;

BlockBackend::Scalar operator+(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator-(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator*(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator/(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator+(complex128 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator-(complex128 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator*(complex128 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator/(complex128 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator<(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator>(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator<=(float64 left, const BlockBackend::Scalar& right);
BlockBackend::Scalar operator>=(float64 left, const BlockBackend::Scalar& right);

BlockPtr operator*(const BlockBackend::Scalar& left, const BlockBackend::Block& right);
BlockPtr operator<(const BlockBackend::Block& left, const BlockBackend::Scalar& right);
BlockPtr operator>(const BlockBackend::Block& left, const BlockBackend::Scalar& right);
BlockPtr operator<=(const BlockBackend::Block& left, const BlockBackend::Scalar& right);
BlockPtr operator>=(const BlockBackend::Block& left, const BlockBackend::Scalar& right);
BlockPtr operator<(const BlockBackend::Scalar& left, const BlockBackend::Block& right);
BlockPtr operator>(const BlockBackend::Scalar& left, const BlockBackend::Block& right);
BlockPtr operator<=(const BlockBackend::Scalar& left, const BlockBackend::Block& right);
BlockPtr operator>=(const BlockBackend::Scalar& left, const BlockBackend::Block& right);
BlockPtr operator<(const BlockBackend::Block& left, float64 right);
BlockPtr operator>(const BlockBackend::Block& left, float64 right);
BlockPtr operator<=(const BlockBackend::Block& left, float64 right);
BlockPtr operator>=(const BlockBackend::Block& left, float64 right);
BlockPtr operator<(float64 left, const BlockBackend::Block& right);
BlockPtr operator>(float64 left, const BlockBackend::Block& right);
BlockPtr operator<=(float64 left, const BlockBackend::Block& right);
BlockPtr operator>=(float64 left, const BlockBackend::Block& right);

std::ostream& operator<<(std::ostream& os, const BlockBackend::Block& block);
std::ostream& operator<<(std::ostream& os, const BlockBackend::Scalar& scalar);

} // namespace lasyten
