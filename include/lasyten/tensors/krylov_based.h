#pragma once

#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/tensors/sparse.h>
#include <lasyten/tensors/vector_like.h>

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace lasyten {

/// Typed Krylov / GMRES options (replaces the old ``py::dict`` member for Python-free builds).
using KrylovOptionValue = std::variant<bool, int64, float64, std::string>;
using KrylovOptions = std::map<std::string, KrylovOptionValue>;

#if LASYTEN_HAS_PYTHON
[[nodiscard]] KrylovOptions krylov_options_from_py(py::object options);
[[nodiscard]] py::dict krylov_options_to_py(KrylovOptions const& options);
#endif

/// Base class for iterative algorithms building a Krylov basis with lasyten tensors.
///
/// Algorithms like `LanczosGroundState` and `Arnoldi`
/// are based on iteratively building an orthonormal basis of the Krylov space spanned by
/// ``|psi0>, H|psi0>, H^2|psi0>, ... H^N |psi0>``, where `N` is the number of iterations
/// performed so far, and ``|psi0>`` is an initial guess and starting vector.
/// During that iteration, the projection of `H` into the Krylov space is built, where it can
/// be solved effectively (with `H` being just a N by N matrix), yielding the "Ritz" eigenvalues/
/// eigenvectors. Finally, the solution can be translated back into the original space using the
/// basis.
///
/// An important strategy is also to (implicitly) restart the algorithm after some number of steps.
/// This is **not** done here: when we use these classes, we usually have an explicit outer loop
/// performed until convergence, e.g., the "sweeps" in DMRG.
///
/// @param H A hermitian linear operator. To use tensors, see `TensorLinearOperator`.
///     The operator must map tensors to tensors with the same legs.
/// @param psi0 The starting vector defining the Krylov basis. For finding the ground state,
///     this should be the best guess available. A `Tensor` of any rank, or a `DirectSum`
///     of tensors, is allowed.
/// @param options Further optional parameters as described below. The algorithm stops if
///     *both* criteria for `E_tol` and `P_tol` are met or if the maximum number of steps
///     was reached.
///
/// Options:
///
/// N_min : int
///     Minimum number of steps to perform.
/// N_max : int
///     Maximum number of steps to perform.
/// P_tol : float
///     Tolerance for the error estimate from the Ritz Residual,
///     stop if ``(RitzRes/gap)**2 < P_tol``
/// min_gap : float
///     Lower cutoff for the gap estimate used in the P_tol criterion.
/// cutoff : float
///     Cutoff to abort if the norm of the new Krylov vector is too small.
///     This is necessary if the rank of `H` is smaller than `N_max`, but it's *not* the error
///     tolerance for final values!
/// E_shift : float
///     Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
///     `ShiftedLinearOperator`.
///     The ground state energy `E0` returned by `run` is made independent of the shift.
///     This option is useful if the `ProjectedLinearOperator` is used: the orthogonal vectors
///     are *exact* eigenvectors with eigenvalue 0 independent of the shift, so you can use it
///     to ensure that the energy is smaller than zero to avoid getting those.
/// reortho : bool
///     For poorly conditioned matrices, one can quickly lose orthogonality of the
///     generated Krylov basis.
///     If `reortho` is True, we re-orthogonalize against all the
///     vectors kept in cache to avoid that problem.
///
/// Attributes:
///
/// options : dict_like
///     Optional parameters.
/// H : `LinearOperator`
///     The linear operator used for building the Krylov space.
/// psi0 : `VectorLike`
///     The *normalized* starting vector.
/// N_min, N_max, P_tol, min_gap, _cutoff, E_shift:
///     Parameters as described in the options.
/// Es : ndarray, shape(N_max, N_max)
///     ``Es[n, :]`` contains the energies of ``_h_krylov[:n+1, :n+1]`` in step `n`.
/// _h_krylov : ndarray, shape (N_max + 1, N_max +1)
///     The matrix representing `H` projected onto the orthonormalized Krylov basis.
/// _psi0_norm : float
///     Initial norm of the `psi0` parameter. Note that ``self.psi0`` gets normalized.
/// _cache : list of psi0-like vectors
///     The ONB of the Krylov space generated during the iteration.
///     FIFO (first in first out) cache of at most `N_cache` vectors.
/// _result_krylov : ndarray
///     Result in the ONB of the Krylov space, e.g. the ground state of `_h_krylov`.
///     What exactly this is depends on the subclass.
///
/// Notes:
///
/// The Ritz residual `RitzRes` is computed according to
/// http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
/// Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
/// ``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.
class KrylovBased
{
  public:
    using Ptr = std::shared_ptr<KrylovBased>;

    LinearOperator::Ptr H;
    VectorLike::Ptr psi0;
    KrylovOptions options;
    int64 N_min = 2;
    int64 N_max = 20;
    int64 N_cache = 20;
    float64 P_tol = 1.0e-14;
    float64 min_gap = 1.0e-12;
    bool reortho = false;
    std::optional<float64> E_shift;
    float64 _cutoff = 0.;
    std::optional<float64> _psi0_norm;
    std::vector<VectorLike::Ptr> _cache;
    /// ``Es[n, :]`` contains Ritz values of the n-step projected operator.
    std::vector<complex128> Es;
    /// H projected onto the Krylov ONB, shape ``(N_max+1, N_max+1)``, row-major.
    std::vector<complex128> _h_krylov;
    /// Result in the Krylov ONB (vector or column-major matrix, see ``_result_krylov_cols``).
    std::vector<complex128> _result_krylov;
    int64 _result_krylov_rows = 0;
    int64 _result_krylov_cols = 0;
    Dtype _dtype_h_krylov = Dtype::Complex128;
    Dtype _dtype_E = Dtype::Complex128;

    KrylovBased(LinearOperator::Ptr H,
                VectorLike::Ptr psi0,
                KrylovOptions options = {},
                Dtype dtype_h_krylov = Dtype::Complex128,
                Dtype dtype_E = Dtype::Complex128);
#if LASYTEN_HAS_PYTHON
    KrylovBased(LinearOperator::Ptr H,
                VectorLike::Ptr psi0,
                py::object options,
                Dtype dtype_h_krylov = Dtype::Complex128,
                Dtype dtype_E = Dtype::Complex128);
#endif
    virtual ~KrylovBased() = default;

    [[nodiscard]] virtual int64 _build_krylov() = 0;
    virtual void _calc_result_krylov(int64 k) = 0;
    [[nodiscard]] virtual bool _converged(int64 k) = 0;

    /// Clear cached Krylov vectors and the projected Hessenberg matrix.
    void _reset_krylov_state();
    /// Transform ``_result_krylov`` from the Krylov ONB to the original basis (single vector).
    [[nodiscard]] VectorLike::Ptr _calc_result_full(int64 N);
    /// Add psi to cache, keep at most ``N_cache``.
    virtual void _to_cache(VectorLike::Ptr psi);
    /// Rebuild uncached Krylov vectors while accumulating the result (Lanczos).
    [[nodiscard]] virtual VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif,
                                                                          int64 N_max);

    [[nodiscard]] complex128 h_krylov(int64 i, int64 j) const;
    void set_h_krylov(int64 i, int64 j, complex128 value);
    [[nodiscard]] complex128 Es_at(int64 i, int64 j) const;
    void set_Es(int64 i, int64 j, complex128 value);
#if LASYTEN_HAS_PYTHON
    [[nodiscard]] py::array Es_numpy() const;
    [[nodiscard]] py::array h_krylov_numpy() const;
    [[nodiscard]] py::array result_krylov_numpy() const;
#endif

  protected:
    [[nodiscard]] int64 h_stride() const { return N_max + 1; }
};

/// GMRES solver for ``A x = b`` with lasyten tensors.
///
/// @param A Linear operator. Must implement `matvec`.
/// @param x Initial guess. Copied; the caller's vector is not modified.
/// @param b Right-hand side.
/// @param options Solver options.
///
/// Options:
///
/// N_min : int
///     Minimum number of Arnoldi steps per restart cycle before checking convergence.
/// N_max : int
///     Maximum Krylov dimension per restart cycle.
/// restart : int
///     Maximum number of restart cycles.
/// res : float
///     Relative residual tolerance ``|A x - b| / |b|``.
class GMRES
{
  public:
    using Ptr = std::shared_ptr<GMRES>;

    KrylovOptions options;
    int64 N_min = 5;
    int64 N_max = 20;
    int64 restart = 10;
    float64 res = 1.0e-8;
    LinearOperator::Ptr A;
    VectorLike::Ptr b;
    VectorLike::Ptr x;
    std::vector<VectorLike::Ptr> rs;
    std::vector<VectorLike::Ptr> qs;
    float64 b_norm = 0.;
    float64 r_norm = 0.;
    std::vector<std::vector<float64>> total_error;
    std::vector<int64> total_iters;
    std::vector<complex128> sine;
    std::vector<complex128> cosine;
    std::vector<complex128> e1;
    std::vector<complex128> H; // (N_max+1, N_max) row-major
    std::vector<complex128> y;

    GMRES(LinearOperator::Ptr A, VectorLike::Ptr x, VectorLike::Ptr b, KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
    GMRES(LinearOperator::Ptr A, VectorLike::Ptr x, VectorLike::Ptr b, py::object options);
#endif

    void _init_hessenberg();
    /// Returns ``(x, rel_residual, total_error, total_iters)``.
    std::tuple<VectorLike::Ptr, float64, std::vector<std::vector<float64>>, std::vector<int64>>
    run();
    void arnoldi(int64 k);
    void apply_givens_rotation(int64 k);
    void givens_rotation(int64 k);
    void backsolve(int64 k);
    void reset();

#if LASYTEN_HAS_PYTHON
    [[nodiscard]] py::array H_numpy() const;
#endif
};

/// Arnoldi method for diagonalizing square, non-hermitian/symmetric matrices.
///
/// Generalization of `LanczosGroundState`, allowing general, square matrices.
///
/// Options:
///
/// Also accepts `KrylovBased` options.
///
/// E_tol : float
///     Stop if energy difference per step < `E_tol`
/// which : ``'LM' | 'LR' | 'SR'``
///     Determines which (extremal) eigenvalues to look for, namely
///     largest magnitude (in absolute value, ``'LM'``), or
///     largest or smallest real part (``'LR'`` and ``'SR'``, respectively).
/// num_ev : int
///     Number of eigenvectors to look for/return in `run`.
class Arnoldi : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<Arnoldi>;

    float64 E_tol = std::numeric_limits<float64>::infinity();
    std::string which = "LM";
    int64 num_ev = 1;

    Arnoldi(LinearOperator::Ptr H, VectorLike::Ptr psi0, KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
    Arnoldi(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);
#endif

    /// Find the ground state of self.H.
    ///
    /// @returns
    ///     E0s : Best eigenvalue estimates, `num_ev` entries, sorted according to `which`.
    ///     psis : Corresponding best eigenvectors (estimates).
    ///     N : Used dimension of the Krylov space, i.e., how many iterations were performed.
    std::tuple<std::vector<complex128>, std::vector<VectorLike::Ptr>, int64> run();
    int64 _build_krylov() override;
    void _calc_result_krylov(int64 k) override;
    /// Transform Krylov eigenvectors back to the original basis.
    [[nodiscard]] std::vector<VectorLike::Ptr> _calc_result_full_multi(int64 N);
    void _to_cache(VectorLike::Ptr psi) override;
    bool _converged(int64 k) override;
};

/// Compute @f$ exp(\delta H) |\psi_0\rangle @f$ using Arnoldi for non-Hermitian `H`.
///
/// Drop-in replacement for `LanczosEvolution` when `H` is not Hermitian.
/// Builds an upper Hessenberg projection of `H` via full Gram-Schmidt orthogonalization
/// (Arnoldi iteration), then computes the matrix exponential of the small projected matrix
/// via eigendecomposition (``numpy.linalg.eig`` + pointwise scalar exponentials).
///
/// @param H, psi0, options Same as `Arnoldi`. Note that `H` need not be Hermitian.
///
/// Options:
///
/// Also accepts `Arnoldi` options.
///
/// E_tol, which, num_ev :
///     Inherited but ignored.
///
/// Attributes:
///
/// delta : float/complex or None
///     Prefactor of H in the exponential.
/// _result_norm : float
///     Norm of the result vector.
class ArnoldiEvolution : public Arnoldi
{
  public:
    using Ptr = std::shared_ptr<ArnoldiEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    ArnoldiEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
    ArnoldiEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);
#endif

    /// Compute ``expm(delta * H).dot(psi0)`` using Arnoldi.
    ///
    /// @param delta Prefactor of H in the exponential. Note that the complex ``i`` is *not*
    /// included.
    /// @param normalize Whether to normalize the result. Defaults to ``False``. Unlike
    /// `LanczosEvolution` (which defaults to ``np.real(delta) == 0``), non-Hermitian evolution
    /// does not in general preserve the norm, so normalization would strip physically meaningful
    /// decay or growth and is off by default.
    /// @returns
    ///     psi_f : Best approximation for ``expm(delta * H).dot(psi0)``.
    ///     N : Krylov space dimension used.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
    [[nodiscard]] VectorLike::Ptr _calc_result_full_evolution(int64 N);
};

/// Lanczos algorithm to find the ground state.
///
/// **Assumes** that `H` is hermitian.
///
/// Options:
///
/// Also accepts `KrylovBased` options.
///
/// E_tol : float
///     Stop if energy difference per step < `E_tol`
/// N_cache : int
///     The maximum number of `psi` to keep in memory during the first iteration.
///     By default, we keep all states (up to N_max).
///     Set this to a number >= 2 if you are short on memory.
///     The penalty is that one needs another Lanczos iteration to
///     determine the ground state in the end, i.e., runtime is large.
class LanczosGroundState : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<LanczosGroundState>;

    float64 E_tol = std::numeric_limits<float64>::infinity();

    LanczosGroundState(LinearOperator::Ptr H, VectorLike::Ptr psi0, KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
    LanczosGroundState(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);
#endif

    /// Find the ground state of H.
    ///
    /// @returns
    ///     E0 : Ground state energy (estimate).
    ///     psi0 : Ground state vector (estimate).
    ///     N : Used dimension of the Krylov space, i.e., how many iterations were performed.
    std::tuple<float64, VectorLike::Ptr, int64> run();
    int64 _build_krylov() override;
    bool _converged(int64 k) override;
    VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max) override;
    void _calc_result_krylov(int64 k) override;
};

/// Calculate @f$ exp(delta H) |psi0> @f$ using Lanczos.
///
/// It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
/// applied to the starting vector. Instead of diagonalizing the tri-diagonal `h` and taking the
/// ground state, we now calculate ``exp(delta h) e_0`` in the Krylov ONB, where
/// ``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.
///
/// @param H, psi0, options Hamiltonian, starting vector and parameters as defined in
///     `LanczosGroundState`. The option `P_tol` defines when convergence is reached,
///     see `_converged` for details.
///
/// Options:
///
/// Also accepts `LanczosGroundState` options.
///
/// E_tol :
///     Ignored.
/// min_gap :
///     Ignored.
///
/// Attributes:
///
/// delta : float/complex
///     Prefactor of H in the exponential.
/// _result_norm : float
///     Norm of the resulting vector.
class LanczosEvolution : public LanczosGroundState
{
  public:
    using Ptr = std::shared_ptr<LanczosEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    LanczosEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
    LanczosEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);
#endif

    /// Calculate ``expm(delta H).dot(psi0)`` using Lanczos.
    ///
    /// @param delta Time step by which we should evolve psi0: prefactor of H in the exponential.
    /// Note that the complex `i` is *not* included!
    /// @param normalize Whether to normalize the resulting state. Defaults to ``np.real(delta) ==
    /// 0``.
    /// @returns
    ///     psi_f : Best approximation for ``expm(delta H).dot(psi0)``. If `E_shift` is used,
    ///     it's an approximation for ``expm(delta (H + E_shift)).dot(psi)``.
    ///     N : Krylov space dimension used.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
};

/// Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``.
///
/// @param H, psi, options See `LanczosGroundState`.
/// @returns E0, psi0, N : See `LanczosGroundState::run`.
std::tuple<float64, VectorLike::Ptr, int64> lanczos(LinearOperator::Ptr H,
                                                    VectorLike::Ptr psi,
                                                    KrylovOptions options = {});
#if LASYTEN_HAS_PYTHON
std::tuple<float64, VectorLike::Ptr, int64> lanczos(LinearOperator::Ptr H,
                                                    VectorLike::Ptr psi,
                                                    py::object options);
#endif

} // namespace lasyten
