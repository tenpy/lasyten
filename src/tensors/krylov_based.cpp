#include <lasyten/detail/python_api.h>
#include <lasyten/tensors/krylov_based.h>

#include <lasyten/block_backend/dtypes.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lasyten {

namespace {

// Eigen is not a project dependency; Krylov projected problems are small (N_max ~ 20),
// so we use dense C++ helpers instead of numpy.linalg.

void
krylov_log_warning(std::string const& msg)
{
    std::cerr << "WARNING:lasyten.tensors.krylov_based: " << msg << '\n';
}

void
krylov_log_debug(std::string const& msg)
{
    // Mirror Python logging.getLogger(...).debug — keep quiet unless useful for tracing.
    (void)msg;
}

py::dict
parse_options(py::object options)
{
    if (options.is_none()) {
        return py::dict();
    }
    if (py::isinstance<py::dict>(options)) {
        return options.cast<py::dict>();
    }
    return py::dict(options);
}

template<typename T>
T
dict_get(py::dict const& d, char const* key, T const& def)
{
    if (d.contains(key) && !d[key].is_none()) {
        return d[key].cast<T>();
    }
    return def;
}

float64
abs_number(BlockBackend::Scalar const& s)
{
    return std::abs(s.as_complex128());
}

complex128
to_number(BlockBackend::Scalar const& s)
{
    return s.as_complex128();
}

BlockBackend::Scalar
as_scalar(VectorLike const& v, complex128 z)
{
    auto dt = v.vector_dtype();
    if (z.imag() != 0.0 && dtype::is_real(dt)) {
        dt = dtype::to_complex(dt);
    }
    return v.vector_backend()->block_backend->as_scalar(z, dt);
}

BlockBackend::Scalar
as_scalar(VectorLike const& v, float64 x)
{
    return v.vector_backend()->block_backend->as_scalar(x);
}

VectorLike::Ptr
scaled_num(VectorLike::CPtr v, complex128 z)
{
    return v->scaled(as_scalar(*v, z));
}

VectorLike::Ptr
add_vec(VectorLike::Ptr a, VectorLike::CPtr b)
{
    return a->axpy(as_scalar(*a, 1.0), std::move(b));
}

VectorLike::Ptr
sub_vec(VectorLike::Ptr a, VectorLike::CPtr b)
{
    return b->axpy(as_scalar(*b, -1.0), std::move(a));
}

std::size_t
idx2(int64 i, int64 j, int64 cols)
{
    return static_cast<std::size_t>(i * cols + j);
}

py::array
vector_to_numpy_2d(std::vector<complex128> const& data, int64 rows, int64 cols, Dtype dt)
{
    if (dt == Dtype::Float64) {
        py::array_t<float64> arr({ rows, cols });
        auto r = arr.mutable_unchecked<2>();
        for (int64 i = 0; i < rows; ++i) {
            for (int64 j = 0; j < cols; ++j) {
                r(i, j) = data[idx2(i, j, cols)].real();
            }
        }
        return arr;
    }
    py::array_t<complex128> arr({ rows, cols });
    auto r = arr.mutable_unchecked<2>();
    for (int64 i = 0; i < rows; ++i) {
        for (int64 j = 0; j < cols; ++j) {
            r(i, j) = data[idx2(i, j, cols)];
        }
    }
    return arr;
}

std::vector<complex128>
extract_submatrix(std::vector<complex128> const& data, int64 stride, int64 n, bool as_real)
{
    std::vector<complex128> out(static_cast<std::size_t>(n * n));
    for (int64 i = 0; i < n; ++i) {
        for (int64 j = 0; j < n; ++j) {
            complex128 z = data[idx2(i, j, stride)];
            out[idx2(i, j, n)] = as_real ? complex128(z.real(), 0.0) : z;
        }
    }
    return out;
}

float64
dense_norm(std::vector<complex128> const& v)
{
    float64 s = 0.0;
    for (auto z : v) {
        s += std::norm(z);
    }
    return std::sqrt(s);
}

std::vector<complex128>
real_if_close_vec(std::vector<complex128> const& v, float64 tol = 100.)
{
    float64 max_abs = 0.0;
    for (auto z : v) {
        max_abs = std::max(max_abs, std::abs(z));
    }
    float64 thresh = tol * std::numeric_limits<float64>::epsilon() * std::max(max_abs, 1.0);
    std::vector<complex128> out = v;
    for (auto& z : out) {
        if (std::abs(z.imag()) < thresh) {
            z = complex128(z.real(), 0.0);
        }
    }
    return out;
}

/// Dense linear solve A x = b (row-major A, length n). Partial pivoting.
std::vector<complex128>
dense_solve(std::vector<complex128> A, std::vector<complex128> b, int64 n)
{
    std::vector<int64> piv(static_cast<std::size_t>(n));
    std::iota(piv.begin(), piv.end(), int64(0));
    for (int64 k = 0; k < n; ++k) {
        int64 pivrow = k;
        float64 best = std::abs(A[idx2(k, k, n)]);
        for (int64 i = k + 1; i < n; ++i) {
            float64 a = std::abs(A[idx2(i, k, n)]);
            if (a > best) {
                best = a;
                pivrow = i;
            }
        }
        if (best == 0.0) {
            throw std::runtime_error("dense_solve: singular matrix");
        }
        if (pivrow != k) {
            for (int64 j = 0; j < n; ++j) {
                std::swap(A[idx2(k, j, n)], A[idx2(pivrow, j, n)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivrow)]);
            std::swap(piv[static_cast<std::size_t>(k)], piv[static_cast<std::size_t>(pivrow)]);
        }
        for (int64 i = k + 1; i < n; ++i) {
            complex128 f = A[idx2(i, k, n)] / A[idx2(k, k, n)];
            A[idx2(i, k, n)] = f;
            for (int64 j = k + 1; j < n; ++j) {
                A[idx2(i, j, n)] -= f * A[idx2(k, j, n)];
            }
            b[static_cast<std::size_t>(i)] -= f * b[static_cast<std::size_t>(k)];
        }
    }
    for (int64 i = n - 1; i >= 0; --i) {
        complex128 s = b[static_cast<std::size_t>(i)];
        for (int64 j = i + 1; j < n; ++j) {
            s -= A[idx2(i, j, n)] * b[static_cast<std::size_t>(j)];
        }
        b[static_cast<std::size_t>(i)] = s / A[idx2(i, i, n)];
    }
    return b;
}

struct DenseEig
{
    std::vector<complex128> values;  // length n
    std::vector<complex128> vectors; // n x n, column j is eigenvector j (row-major)
};

/// Classical Jacobi eigenvalue decomposition for real symmetric matrices.
DenseEig
dense_eigh(std::vector<complex128> const& A_in, int64 n)
{
    std::vector<float64> A(static_cast<std::size_t>(n * n));
    for (int64 i = 0; i < n * n; ++i) {
        A[static_cast<std::size_t>(i)] = A_in[static_cast<std::size_t>(i)].real();
    }
    std::vector<float64> V(static_cast<std::size_t>(n * n), 0.0);
    for (int64 i = 0; i < n; ++i) {
        V[idx2(i, i, n)] = 1.0;
    }
    constexpr int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float64 off = 0.0;
        for (int64 i = 0; i < n; ++i) {
            for (int64 j = i + 1; j < n; ++j) {
                off += std::abs(A[idx2(i, j, n)]);
            }
        }
        if (off < 1e-15 * static_cast<float64>(n)) {
            break;
        }
        for (int64 p = 0; p < n; ++p) {
            for (int64 q = p + 1; q < n; ++q) {
                float64 app = A[idx2(p, p, n)];
                float64 aqq = A[idx2(q, q, n)];
                float64 apq = A[idx2(p, q, n)];
                if (std::abs(apq) < 1e-30) {
                    continue;
                }
                float64 tau = (aqq - app) / (2.0 * apq);
                float64 t = std::copysign(1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
                float64 c = 1.0 / std::sqrt(1.0 + t * t);
                float64 s = t * c;
                A[idx2(p, p, n)] = app - t * apq;
                A[idx2(q, q, n)] = aqq + t * apq;
                A[idx2(p, q, n)] = A[idx2(q, p, n)] = 0.0;
                for (int64 r = 0; r < n; ++r) {
                    if (r == p || r == q) {
                        continue;
                    }
                    float64 arp = A[idx2(r, p, n)];
                    float64 arq = A[idx2(r, q, n)];
                    A[idx2(r, p, n)] = A[idx2(p, r, n)] = c * arp - s * arq;
                    A[idx2(r, q, n)] = A[idx2(q, r, n)] = s * arp + c * arq;
                }
                for (int64 r = 0; r < n; ++r) {
                    float64 vrp = V[idx2(r, p, n)];
                    float64 vrq = V[idx2(r, q, n)];
                    V[idx2(r, p, n)] = c * vrp - s * vrq;
                    V[idx2(r, q, n)] = s * vrp + c * vrq;
                }
            }
        }
    }
    // Sort eigenvalues ascending (numpy.linalg.eigh convention).
    std::vector<int64> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), int64(0));
    std::ranges::sort(order,
                      [&](int64 a, int64 b) { return A[idx2(a, a, n)] < A[idx2(b, b, n)]; });
    DenseEig out;
    out.values.resize(static_cast<std::size_t>(n));
    out.vectors.assign(static_cast<std::size_t>(n * n), complex128(0.));
    for (int64 j = 0; j < n; ++j) {
        int64 src = order[static_cast<std::size_t>(j)];
        out.values[static_cast<std::size_t>(j)] = A[idx2(src, src, n)];
        for (int64 i = 0; i < n; ++i) {
            out.vectors[idx2(i, j, n)] = V[idx2(i, src, n)];
        }
    }
    return out;
}

/// General (non-hermitian) eigendecomposition of a dense complex matrix (small n).
///
/// Reduces `A` to upper Hessenberg form with Householder reflections, then to complex Schur
/// form ``A = Z T Z^H`` with the shifted QR algorithm (Wilkinson shifts, deflation, exceptional
/// shifts), and finally computes the eigenvectors of the triangular `T` by back-substitution.
/// The eigenvectors are normalized and their phase is fixed such that the largest entry is real
/// and positive. In particular, eigenvectors of a real matrix for real eigenvalues are real.
DenseEig
dense_eig(std::vector<complex128> A, int64 n)
{
    constexpr float64 eps = std::numeric_limits<float64>::epsilon();
    auto const N = static_cast<std::size_t>(n);
    std::vector<complex128> Z(N * N, complex128(0.));
    for (int64 i = 0; i < n; ++i) {
        Z[idx2(i, i, n)] = 1.0;
    }
    // 1) Householder reduction to upper Hessenberg form: A = Z H Z^H
    std::vector<complex128> v(N);
    for (int64 k = 0; k + 2 < n; ++k) {
        float64 xnorm = 0.0;
        for (int64 i = k + 1; i < n; ++i) {
            xnorm += std::norm(A[idx2(i, k, n)]);
        }
        xnorm = std::sqrt(xnorm);
        if (xnorm == 0.0) {
            continue;
        }
        complex128 x0 = A[idx2(k + 1, k, n)];
        complex128 phase = (x0 == complex128(0.)) ? complex128(1.) : x0 / std::abs(x0);
        float64 vnorm = 0.0;
        for (int64 i = k + 1; i < n; ++i) {
            v[static_cast<std::size_t>(i)] = A[idx2(i, k, n)];
        }
        v[static_cast<std::size_t>(k + 1)] += phase * xnorm;
        for (int64 i = k + 1; i < n; ++i) {
            vnorm += std::norm(v[static_cast<std::size_t>(i)]);
        }
        vnorm = std::sqrt(vnorm);
        for (int64 i = k + 1; i < n; ++i) {
            v[static_cast<std::size_t>(i)] /= vnorm;
        }
        // A[k+1:, k:] -= 2 v (v^H A[k+1:, k:])
        for (int64 j = k; j < n; ++j) {
            complex128 s = 0.;
            for (int64 i = k + 1; i < n; ++i) {
                s += std::conj(v[static_cast<std::size_t>(i)]) * A[idx2(i, j, n)];
            }
            for (int64 i = k + 1; i < n; ++i) {
                A[idx2(i, j, n)] -= 2.0 * v[static_cast<std::size_t>(i)] * s;
            }
        }
        // A[:, k+1:] -= 2 (A[:, k+1:] v) v^H, and the same for Z
        for (auto* M : { &A, &Z }) {
            for (int64 i = 0; i < n; ++i) {
                complex128 s = 0.;
                for (int64 j = k + 1; j < n; ++j) {
                    s += (*M)[idx2(i, j, n)] * v[static_cast<std::size_t>(j)];
                }
                for (int64 j = k + 1; j < n; ++j) {
                    (*M)[idx2(i, j, n)] -= 2.0 * s * std::conj(v[static_cast<std::size_t>(j)]);
                }
            }
        }
        for (int64 i = k + 2; i < n; ++i) {
            A[idx2(i, k, n)] = 0.;
        }
    }
    auto& H = A;
    float64 norm_H = 0.0;
    for (auto z : H) {
        norm_H += std::abs(z);
    }
    norm_H = std::max(norm_H, std::numeric_limits<float64>::min());
    // 2) complex QR iteration on the Hessenberg matrix -> upper triangular (Schur form) T
    std::vector<float64> rot_c(N);
    std::vector<complex128> rot_s(N);
    int64 hi = n - 1;
    int64 iter = 0;
    int64 total_iter = 0;
    while (hi > 0) {
        // find the start `l` of the active (unreduced) block ending at `hi`
        int64 l = hi;
        for (; l > 0; --l) {
            float64 s = std::abs(H[idx2(l - 1, l - 1, n)]) + std::abs(H[idx2(l, l, n)]);
            if (s == 0.0) {
                s = norm_H;
            }
            if (std::abs(H[idx2(l, l - 1, n)]) <= eps * s) {
                H[idx2(l, l - 1, n)] = 0.;
                break;
            }
        }
        if (l == hi) {
            // eigenvalue H[hi, hi] converged
            --hi;
            iter = 0;
            continue;
        }
        ++iter;
        if (++total_iter > 100 * n) {
            throw std::runtime_error("dense_eig: QR iteration did not converge");
        }
        complex128 mu;
        complex128 d = H[idx2(hi, hi, n)];
        if (iter % 10 == 0) {
            // exceptional shift to break possible cycles
            mu = d + std::abs(H[idx2(hi, hi - 1, n)]) * complex128(0.75, 0.5);
        } else {
            // Wilkinson shift: eigenvalue of the trailing 2x2 block closer to H[hi, hi]
            complex128 a = H[idx2(hi - 1, hi - 1, n)];
            complex128 b = H[idx2(hi - 1, hi, n)];
            complex128 c = H[idx2(hi, hi - 1, n)];
            complex128 half_tr = 0.5 * (a + d);
            complex128 disc = std::sqrt(0.25 * (a - d) * (a - d) + b * c);
            complex128 mu1 = half_tr + disc;
            complex128 mu2 = half_tr - disc;
            mu = (std::abs(mu1 - d) < std::abs(mu2 - d)) ? mu1 : mu2;
        }
        for (int64 k = l; k <= hi; ++k) {
            H[idx2(k, k, n)] -= mu;
        }
        // QR decomposition of the active block with Givens rotations, applied from the left ...
        for (int64 k = l; k < hi; ++k) {
            complex128 x = H[idx2(k, k, n)];
            complex128 y = H[idx2(k + 1, k, n)];
            float64 r = std::hypot(std::abs(x), std::abs(y));
            float64 cs;
            complex128 sn;
            if (r == 0.0) {
                cs = 1.0;
                sn = 0.;
            } else if (x == complex128(0.)) {
                cs = 0.0;
                sn = std::conj(y) / std::abs(y);
            } else {
                cs = std::abs(x) / r;
                sn = (x / std::abs(x)) * std::conj(y) / r;
            }
            rot_c[static_cast<std::size_t>(k)] = cs;
            rot_s[static_cast<std::size_t>(k)] = sn;
            for (int64 j = k; j < n; ++j) {
                complex128 t1 = H[idx2(k, j, n)];
                complex128 t2 = H[idx2(k + 1, j, n)];
                H[idx2(k, j, n)] = cs * t1 + sn * t2;
                H[idx2(k + 1, j, n)] = -std::conj(sn) * t1 + cs * t2;
            }
        }
        // ... and their adjoints from the right (also accumulated into Z), giving R Q
        for (int64 k = l; k < hi; ++k) {
            float64 cs = rot_c[static_cast<std::size_t>(k)];
            complex128 sn = rot_s[static_cast<std::size_t>(k)];
            int64 top = std::min(k + 2, hi);
            for (int64 i = 0; i <= top; ++i) {
                complex128 t1 = H[idx2(i, k, n)];
                complex128 t2 = H[idx2(i, k + 1, n)];
                H[idx2(i, k, n)] = cs * t1 + std::conj(sn) * t2;
                H[idx2(i, k + 1, n)] = -sn * t1 + cs * t2;
            }
            for (int64 i = 0; i < n; ++i) {
                complex128 t1 = Z[idx2(i, k, n)];
                complex128 t2 = Z[idx2(i, k + 1, n)];
                Z[idx2(i, k, n)] = cs * t1 + std::conj(sn) * t2;
                Z[idx2(i, k + 1, n)] = -sn * t1 + cs * t2;
            }
        }
        for (int64 k = l; k <= hi; ++k) {
            H[idx2(k, k, n)] += mu;
        }
    }
    // 3) eigenvectors of the triangular T = H by back-substitution, then transform with Z
    DenseEig out;
    out.values.resize(N);
    out.vectors.assign(N * N, complex128(0.));
    for (int64 i = 0; i < n; ++i) {
        out.values[static_cast<std::size_t>(i)] = H[idx2(i, i, n)];
    }
    float64 small = std::max(eps * norm_H, std::numeric_limits<float64>::min());
    std::vector<complex128> y(N);
    std::vector<complex128> x(N);
    for (int64 k = 0; k < n; ++k) {
        complex128 lambda = out.values[static_cast<std::size_t>(k)];
        y[static_cast<std::size_t>(k)] = 1.0;
        for (int64 j = k - 1; j >= 0; --j) {
            complex128 s = 0.;
            for (int64 m = j + 1; m <= k; ++m) {
                s += H[idx2(j, m, n)] * y[static_cast<std::size_t>(m)];
            }
            complex128 denom = H[idx2(j, j, n)] - lambda;
            if (std::abs(denom) < small) {
                denom = small;
            }
            y[static_cast<std::size_t>(j)] = -s / denom;
        }
        float64 max_abs = -1.0;
        int64 i_max = 0;
        for (int64 i = 0; i < n; ++i) {
            complex128 s = 0.;
            for (int64 j = 0; j <= k; ++j) {
                s += Z[idx2(i, j, n)] * y[static_cast<std::size_t>(j)];
            }
            x[static_cast<std::size_t>(i)] = s;
            if (std::abs(s) > max_abs) {
                max_abs = std::abs(s);
                i_max = i;
            }
        }
        // fix the phase (largest entry real and positive) and normalize
        complex128 phase = std::conj(x[static_cast<std::size_t>(i_max)]) / max_abs;
        float64 xnorm = dense_norm(x) * std::abs(phase);
        for (int64 i = 0; i < n; ++i) {
            out.vectors[idx2(i, k, n)] = x[static_cast<std::size_t>(i)] * phase / xnorm;
        }
    }
    return out;
}

std::vector<int64>
argsort_which(std::vector<complex128> const& values, std::string const& which)
{
    std::vector<int64> idx(static_cast<std::size_t>(values.size()));
    std::iota(idx.begin(), idx.end(), int64(0));
    auto key = [&](complex128 z) -> float64 {
        if (which == "LM" || which == "m>") {
            return -std::abs(z);
        }
        if (which == "SM" || which == "m<") {
            return std::abs(z);
        }
        if (which == "LR" || which == ">" || which == "LA") {
            return -z.real();
        }
        if (which == "SR" || which == "<" || which == "SA") {
            return z.real();
        }
        if (which == "LI") {
            return -z.imag();
        }
        if (which == "SI") {
            return z.imag();
        }
        return z.real();
    };
    std::ranges::sort(idx, [&](int64 a, int64 b) {
        return key(values[static_cast<std::size_t>(a)]) < key(values[static_cast<std::size_t>(b)]);
    });
    return idx;
}

} // namespace

complex128
KrylovBased::h_krylov(int64 i, int64 j) const
{
    return _h_krylov[idx2(i, j, h_stride())];
}

void
KrylovBased::set_h_krylov(int64 i, int64 j, complex128 value)
{
    _h_krylov[idx2(i, j, h_stride())] = value;
}

complex128
KrylovBased::Es_at(int64 i, int64 j) const
{
    return Es[idx2(i, j, N_max)];
}

void
KrylovBased::set_Es(int64 i, int64 j, complex128 value)
{
    Es[idx2(i, j, N_max)] = value;
}

py::array
KrylovBased::Es_numpy() const
{
    return vector_to_numpy_2d(Es, N_max, N_max, _dtype_E);
}

py::array
KrylovBased::h_krylov_numpy() const
{
    auto n = h_stride();
    return vector_to_numpy_2d(_h_krylov, n, n, _dtype_h_krylov);
}

py::array
KrylovBased::result_krylov_numpy() const
{
    if (_result_krylov_cols <= 1) {
        if (_dtype_h_krylov == Dtype::Float64) {
            py::array_t<float64> arr(_result_krylov_rows);
            auto r = arr.mutable_unchecked<1>();
            for (int64 i = 0; i < _result_krylov_rows; ++i) {
                r(i) = _result_krylov[static_cast<std::size_t>(i)].real();
            }
            return arr;
        }
        py::array_t<complex128> arr(_result_krylov_rows);
        auto r = arr.mutable_unchecked<1>();
        for (int64 i = 0; i < _result_krylov_rows; ++i) {
            r(i) = _result_krylov[static_cast<std::size_t>(i)];
        }
        return arr;
    }
    return vector_to_numpy_2d(
      _result_krylov, _result_krylov_rows, _result_krylov_cols, _dtype_h_krylov);
}

KrylovBased::KrylovBased(LinearOperator::Ptr H_,
                         VectorLike::Ptr psi0_,
                         py::object options_,
                         Dtype dtype_h_krylov,
                         Dtype dtype_E)
  : H(std::move(H_))
  , psi0(psi0_ ? psi0_->clone() : nullptr)
  , _dtype_h_krylov(dtype_h_krylov)
  , _dtype_E(dtype_E)
{
    if (!H) {
        throw std::invalid_argument("H must not be null");
    }
    if (!psi0) {
        throw std::invalid_argument("psi0 must not be null");
    }
    options = parse_options(std::move(options_));
    N_min = dict_get<int64>(options, "N_min", int64(2));
    N_max = dict_get<int64>(options, "N_max", int64(20));
    N_cache = N_max;
    P_tol = dict_get<float64>(options, "P_tol", 1.0e-14);
    min_gap = dict_get<float64>(options, "min_gap", 1.0e-12);
    reortho = dict_get<bool>(options, "reortho", false);
    if (options.contains("E_shift") && !options["E_shift"].is_none()) {
        E_shift = options["E_shift"].cast<float64>();
    }
    if (N_min < 2) {
        throw std::invalid_argument("Should perform at least 2 steps.");
    }
    _cutoff = dict_get<float64>(options, "cutoff", dtype::eps(psi0->vector_dtype()) * 100);
    if (E_shift.has_value()) {
        if (auto proj = std::dynamic_pointer_cast<ProjectedLinearOperator>(H)) {
            proj->original_operator =
              std::make_shared<ShiftedLinearOperator>(proj->original_operator, *E_shift);
        } else {
            H = std::make_shared<ShiftedLinearOperator>(H, *E_shift);
        }
    }
    Es.assign(static_cast<std::size_t>(N_max * N_max), complex128(0.));
    _h_krylov.assign(static_cast<std::size_t>((N_max + 1) * (N_max + 1)), complex128(0.));
}

void
KrylovBased::_reset_krylov_state()
{
    _cache.clear();
    std::fill(_h_krylov.begin(), _h_krylov.end(), complex128(0.));
    std::fill(Es.begin(), Es.end(), complex128(0.));
}

VectorLike::Ptr
KrylovBased::_calc_result_full(int64 N)
{
    // this implementation assumes there is a single state
    auto const& vf = _result_krylov;
    auto const len_vf = static_cast<int64>(vf.size());
    if (!(N == len_vf && len_vf > 1)) {
        throw std::runtime_error("KrylovBased._calc_result_full: expected N == len(vf) > 1");
    }
    auto psif = scaled_num(psi0, vf[0]);
    auto const len_cache = static_cast<int64>(_cache.size());
    auto const n_loop = std::min(len_cache + 1, N);
    for (int64 k = 1; k < n_loop; ++k) {
        psif = add_vec(std::move(psif),
                       scaled_num(_cache[static_cast<std::size_t>(len_cache - k)],
                                  vf[static_cast<std::size_t>(N - k)]));
    }
    // other vectors are not cached, so we need to restart the Lanczos iteration.
    _cache.clear(); // free memory: we need at least two more vectors

    psif = _rebuild_krylov_for_result_full(std::move(psif), N - len_cache - 1);

    auto psif_norm = abs_number(norm(VectorLikeCPtr(psif)));
    if (std::abs(1.0 - psif_norm) > 1.0e-5) {
        // One reason can be that `H` is not Hermitian
        // Otherwise, the matrix (even if small) might be ill conditioned.
        // If you get this warning, you can try to set the parameters
        // `reortho`=True and `N_cache` >= `N_max`
        krylov_log_warning(
          std::format("poorly conditioned H matrix in KrylovBased! |psi_0| = {}", psif_norm));
    }
    return scalar_multiply(as_scalar(*psif, 1.0 / psif_norm), psif);
}

void
KrylovBased::_to_cache(VectorLike::Ptr psi)
{
    _cache.push_back(std::move(psi));
    if (static_cast<int64>(_cache.size()) > N_cache) {
        _cache.erase(_cache.begin()); // remove *first* entry
    }
}

VectorLike::Ptr
KrylovBased::_rebuild_krylov_for_result_full(VectorLike::Ptr /*psif*/, int64 /*N_max*/)
{
    throw NotImplemented("KrylovBased._rebuild_krylov_for_result_full");
}

GMRES::GMRES(LinearOperator::Ptr A_, VectorLike::Ptr x_, VectorLike::Ptr b_, py::object options_)
  : A(std::move(A_))
  , b(b_ ? b_->clone() : nullptr)
  , x(x_ ? x_->clone() : nullptr)
{
    if (!A) {
        throw std::invalid_argument("A must not be null");
    }
    if (!x || !b) {
        throw std::invalid_argument("x and b must not be null");
    }
    options = parse_options(std::move(options_));
    N_min = dict_get<int64>(options, "N_min", int64(5));
    N_max = dict_get<int64>(options, "N_max", int64(20));
    restart = dict_get<int64>(options, "restart", int64(10));
    res = dict_get<float64>(options, "res", 1.0e-8);

    auto r0 = sub_vec(b->clone(), A->matvec(x));
    rs = { r0 };
    b_norm = abs_number(norm(VectorLikeCPtr(b)));
    r_norm = abs_number(norm(VectorLikeCPtr(r0)));
    auto denom = b_norm != 0.0 ? b_norm : 1.0;
    total_error = { { r_norm / denom } };
    if (r_norm > 0) {
        qs = { scalar_multiply(as_scalar(*r0, 1.0 / r_norm), r0) };
    } else {
        qs = { r0 };
    }
    _init_hessenberg();
}

void
GMRES::_init_hessenberg()
{
    sine.assign(static_cast<std::size_t>(N_max), complex128(0.));
    cosine.assign(static_cast<std::size_t>(N_max), complex128(0.));
    e1.assign(static_cast<std::size_t>(N_max + 1), complex128(0.));
    e1[0] = r_norm;
    H.assign(static_cast<std::size_t>((N_max + 1) * N_max), complex128(0.));
}

py::array
GMRES::H_numpy() const
{
    return vector_to_numpy_2d(H, N_max + 1, N_max, Dtype::Complex128);
}

std::tuple<VectorLike::Ptr, float64, std::vector<std::vector<float64>>, std::vector<int64>>
GMRES::run()
{
    if (total_error[0][0] < res) {
        return { x, total_error[0][0], total_error, total_iters };
    }
    for (int64 /*cycle*/ _ = 0; _ < restart; ++_) {
        bool converged = false;
        int64 k = 0;
        int64 performed = 0;
        for (; k < N_max; ++k) {
            arnoldi(k);
            apply_givens_rotation(k);
            e1[static_cast<std::size_t>(k + 1)] =
              -sine[static_cast<std::size_t>(k)] * e1[static_cast<std::size_t>(k)];
            e1[static_cast<std::size_t>(k)] =
              cosine[static_cast<std::size_t>(k)] * e1[static_cast<std::size_t>(k)];
            // The residual is the last element of the beta vector (see Wikipedia).
            auto denom = b_norm != 0.0 ? b_norm : 1.0;
            auto error = std::abs(e1[static_cast<std::size_t>(k + 1)]) / denom;
            total_error.back().push_back(error);
            performed = k + 1;
            if (error < res && k >= N_min) {
                converged = true;
                break;
            }
        }
        total_iters.push_back(performed);
        backsolve(performed);
        for (int64 i = 0; i < performed; ++i) {
            x =
              add_vec(std::move(x),
                      scaled_num(qs[static_cast<std::size_t>(i)], y[static_cast<std::size_t>(i)]));
        }
        if (!converged) {
            reset();
        } else {
            break;
        }
    }

    auto rel = abs_number(norm(VectorLikeCPtr(sub_vec(A->matvec(x), b))));
    if (b_norm != 0.0) {
        rel = rel / b_norm;
    }
    return { x, rel, total_error, total_iters };
}

void
GMRES::arnoldi(int64 k)
{
    // Iterative build orthogonal Krylov subspace and Hessenberg matrix.
    auto q = A->matvec(qs.back());
    for (int64 i = 0; i < k + 1; ++i) {
        auto hik = to_number(inner(VectorLikeCPtr(qs[static_cast<std::size_t>(i)]), q));
        H[idx2(i, k, N_max)] = hik;
        q = qs[static_cast<std::size_t>(i)]->axpy(as_scalar(*q, -hik), q);
    }
    auto h_next = abs_number(norm(VectorLikeCPtr(q)));
    H[idx2(k + 1, k, N_max)] = h_next;
    if (h_next > 0) { // avoid warning if norm(q)==0, error=0 in that case
        q = scalar_multiply(as_scalar(*q, 1.0 / h_next), q);
    }
    qs.push_back(std::move(q));
}

void
GMRES::apply_givens_rotation(int64 k)
{
    // Apply rotation to H so that it becomes upper triangular.
    for (int64 i = 0; i < k; ++i) {
        auto temp = cosine[static_cast<std::size_t>(i)] * H[idx2(i, k, N_max)] +
                    sine[static_cast<std::size_t>(i)] * H[idx2(i + 1, k, N_max)];
        H[idx2(i + 1, k, N_max)] = -sine[static_cast<std::size_t>(i)] * H[idx2(i, k, N_max)] +
                                   cosine[static_cast<std::size_t>(i)] * H[idx2(i + 1, k, N_max)];
        H[idx2(i, k, N_max)] = temp;
    }

    givens_rotation(k);
    H[idx2(k, k, N_max)] = cosine[static_cast<std::size_t>(k)] * H[idx2(k, k, N_max)] +
                           sine[static_cast<std::size_t>(k)] * H[idx2(k + 1, k, N_max)];
    H[idx2(k + 1, k, N_max)] = 0;
}

void
GMRES::givens_rotation(int64 k)
{
    // Find cosine and sine such that the element below the diagonal of kth column of H is removed.
    auto v1 = H[idx2(k, k, N_max)];
    auto v2 = H[idx2(k + 1, k, N_max)];
    auto t = std::sqrt(v1 * v1 + v2 * v2);
    cosine[static_cast<std::size_t>(k)] = v1 / t;
    sine[static_cast<std::size_t>(k)] = v2 / t;
}

void
GMRES::backsolve(int64 k)
{
    // H is now upper triangular; backsolve to find y exactly.
    y.assign(static_cast<std::size_t>(k), complex128(0.));
    for (int64 i = k - 1; i >= 0; --i) {
        y[static_cast<std::size_t>(i)] = e1[static_cast<std::size_t>(i)];
        for (int64 j = i + 1; j < k; ++j) {
            y[static_cast<std::size_t>(i)] -=
              H[idx2(i, j, N_max)] * y[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] /= H[idx2(i, i, N_max)];
    }
}

void
GMRES::reset()
{
    // Restart GMRES using current x as initial guess.
    auto r = sub_vec(b->clone(), A->matvec(x));
    rs.push_back(r);
    r_norm = abs_number(norm(VectorLikeCPtr(r)));
    auto denom = b_norm != 0.0 ? b_norm : 1.0;
    total_error.push_back({ r_norm / denom });
    if (r_norm > 0) {
        qs = { scalar_multiply(as_scalar(*r, 1.0 / r_norm), r) };
    } else {
        qs = { std::move(r) };
    }
    _init_hessenberg();
}

Arnoldi::Arnoldi(LinearOperator::Ptr H_, VectorLike::Ptr psi0_, py::object options_)
  : KrylovBased(std::move(H_), std::move(psi0_), options_)
{
    E_tol = dict_get<float64>(options, "E_tol", std::numeric_limits<float64>::infinity());
    which = dict_get<std::string>(options, "which", std::string("LM"));
    num_ev = dict_get<int64>(options, "num_ev", int64(1));
}

std::tuple<std::vector<complex128>, std::vector<VectorLike::Ptr>, int64>
Arnoldi::run()
{
    if (N_cache < N_max) {
        throw std::runtime_error("Arnoldi requires N_cache >= N_max");
    }
    auto N = _build_krylov();
    std::vector<complex128> E0(static_cast<std::size_t>(num_ev));
    for (int64 i = 0; i < num_ev; ++i) {
        E0[static_cast<std::size_t>(i)] = Es_at(N - 1, i);
    }
    if (E_shift.has_value()) {
        for (auto& e : E0) {
            e -= *E_shift;
        }
    }
    if (N == 1) {
        return { std::move(E0), { psi0 }, N };
    }
    return { std::move(E0), _calc_result_full_multi(N), N };
}

int64
Arnoldi::_build_krylov()
{
    auto w = psi0;
    auto w_norm = abs_number(norm(VectorLikeCPtr(w)));
    psi0 = scaled_num(w, 1.0 / w_norm);
    int64 k = 0;
    int64 performed = 0;
    for (; k < N_max; ++k) {
        w = scalar_multiply(as_scalar(*w, 1.0 / w_norm), w);
        _to_cache(w);
        w = H->matvec(w);
        for (int64 i = 0; i < static_cast<int64>(_cache.size()); ++i) {
            auto ov = inner(VectorLikeCPtr(_cache[static_cast<std::size_t>(i)]), w);
            set_h_krylov(i, k, to_number(ov));
            w = _cache[static_cast<std::size_t>(i)]->axpy(-ov, w);
        }
        w_norm = abs_number(norm(VectorLikeCPtr(w)));
        set_h_krylov(k + 1, k, w_norm);
        _calc_result_krylov(k);
        performed = k + 1;
        if (w_norm < _cutoff || (k + 1 >= N_min && _converged(k))) {
            break;
        }
    }
    return performed;
}

void
Arnoldi::_calc_result_krylov(int64 k)
{
    if (k == 0) {
        set_Es(0, 0, h_krylov(0, 0));
        _result_krylov = { 1. };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto h = extract_submatrix(_h_krylov, h_stride(), n, /*as_real=*/false);
    auto ev = dense_eig(std::move(h), n);
    auto const& E_kr = ev.values;
    auto const& v_kr = ev.vectors;
    auto sort = argsort_which(E_kr, which);
    for (int64 j = 0; j < n; ++j) {
        set_Es(k, j, E_kr[static_cast<std::size_t>(sort[static_cast<std::size_t>(j)])]);
    }
    _result_krylov.assign(static_cast<std::size_t>(n * n), complex128(0.));
    _result_krylov_rows = n;
    _result_krylov_cols = n;
    for (int64 col = 0; col < n; ++col) {
        auto src = sort[static_cast<std::size_t>(col)];
        for (int64 row = 0; row < n; ++row) {
            _result_krylov[idx2(row, col, n)] = v_kr[idx2(row, src, n)];
        }
    }
}

std::vector<VectorLike::Ptr>
Arnoldi::_calc_result_full_multi(int64 N)
{
    std::vector<VectorLike::Ptr> psis;
    auto n_ev = std::min(N, num_ev);
    psis.reserve(static_cast<std::size_t>(n_ev));
    for (int64 i = 0; i < n_ev; ++i) {
        std::vector<complex128> vf_raw(static_cast<std::size_t>(N));
        for (int64 row = 0; row < N; ++row) {
            vf_raw[static_cast<std::size_t>(row)] =
              _result_krylov[idx2(row, i, _result_krylov_cols)];
        }
        // try to convert to real:
        // e.g. the dominant eigenvectors of the MPS transfermatrix should be equivalent to
        // the power method, which will be purely real for H.dtype=float, even if there might
        // be other eigenvectors which are complex
        auto vf = real_if_close_vec(vf_raw);
        if (!(N == static_cast<int64>(vf.size()) && N > 1)) {
            throw std::runtime_error("Arnoldi._calc_result_full: expected N == len(vf) > 1");
        }
        auto const& krylov_basis = _cache;
        if (static_cast<int64>(krylov_basis.size()) < N) {
            throw std::runtime_error("Arnoldi._calc_result_full: Krylov basis shorter than N");
        }
        auto psi = scaled_num(krylov_basis[0], vf[0]);
        for (int64 k = 1; k < N; ++k) {
            psi = add_vec(std::move(psi),
                          scaled_num(krylov_basis[static_cast<std::size_t>(k)],
                                     vf[static_cast<std::size_t>(k)]));
        }

        auto psi_norm = abs_number(norm(VectorLikeCPtr(psi)));
        if (std::abs(1.0 - psi_norm) > 1.0e-5) {
            // One reason can be that `H` is not Hermitian
            // Otherwise, the matrix (even if small) might be ill conditioned.
            // If you get this warning, you can try to set the parameters
            // `reortho`=True and `N_cache` >= `N_max`
            krylov_log_warning(
              std::format("poorly conditioned H matrix in Arnoldi! |psi| = {}", psi_norm));
        }
        psis.push_back(scalar_multiply(as_scalar(*psi, 1.0 / psi_norm), psi));
    }
    return psis;
}

void
Arnoldi::_to_cache(VectorLike::Ptr psi)
{
    _cache.push_back(std::move(psi));
    if (static_cast<int64>(_cache.size()) > N_cache) {
        throw std::runtime_error("Arnoldi cache exceeded N_cache");
    }
}

bool
Arnoldi::_converged(int64 k)
{
    auto v0k = _result_krylov[idx2(k, 0, _result_krylov_cols)];
    auto RitzRes = std::abs(v0k) * std::abs(h_krylov(k + 1, k));
    float64 min_diff = std::numeric_limits<float64>::infinity();
    for (int64 i = 0; i < num_ev; ++i) {
        float64 local = std::numeric_limits<float64>::infinity();
        for (int64 j = i + 1; j < N_max; ++j) {
            local = std::min(local, std::abs(Es_at(k, j) - Es_at(k, i)));
        }
        min_diff = std::min(min_diff, local);
    }
    auto gap = std::max(min_diff, min_gap);
    auto P_err = (RitzRes / gap) * (RitzRes / gap);
    auto Delta_E0 = Es_at(k - 1, 0) - Es_at(k, 0);
    return P_err < P_tol && Delta_E0.real() < E_tol;
}

ArnoldiEvolution::ArnoldiEvolution(LinearOperator::Ptr H_,
                                   VectorLike::Ptr psi0_,
                                   py::object options_)
  : Arnoldi(std::move(H_), psi0_, options_)
{
    _result_norm = 1.0;
    delta = std::nullopt;
    // Arnoldi._build_krylov does not set _psi0_norm; do it here.
    _psi0_norm = abs_number(norm(VectorLikeCPtr(psi0_)));
}

std::tuple<VectorLike::Ptr, int64>
ArnoldiEvolution::run(complex128 delta_, std::optional<bool> normalize)
{
    if (N_cache < N_max) {
        throw std::runtime_error("ArnoldiEvolution requires N_cache >= N_max");
    }
    delta = delta_;
    // Arnoldi._to_cache does not pop old entries, so we must clear state between calls.
    _reset_krylov_state();
    auto N = _build_krylov();
    if (N > 1) {
        auto last = _result_krylov[idx2(N - 1, 0, _result_krylov_cols)];
        krylov_log_debug(
          std::format("ArnoldiEvolution N={}, |result[-1]|={:.3e}", N, std::abs(last)));
    } else {
        krylov_log_debug(
          std::format("ArnoldiEvolution N=1, |h[0,0]|={:.3e}", std::abs(h_krylov(0, 0))));
    }
    VectorLike::Ptr result_full;
    if (N == 1) {
        result_full = scaled_num(psi0, _result_krylov[0]);
    } else {
        result_full = _calc_result_full_evolution(N);
    }
    bool do_normalize = normalize.value_or(false);
    if (do_normalize) {
        return { std::move(result_full), N };
    }
    auto scale = (_psi0_norm.value_or(1.0)) * _result_norm;
    return { scaled_num(result_full, scale), N };
}

void
ArnoldiEvolution::_calc_result_krylov(int64 k)
{
    auto dlt = *delta;
    if (k == 0) {
        auto exp_dE = std::exp(dlt * h_krylov(0, 0));
        _result_norm = std::abs(exp_dE);
        _result_krylov = { exp_dE / _result_norm };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto h = extract_submatrix(_h_krylov, h_stride(), n, /*as_real=*/false);
    auto ev = dense_eig(h, n);
    // V^{-1} e0 = first column of V^{-1}; use solve for numerical stability
    std::vector<complex128> e0(static_cast<std::size_t>(n), complex128(0.));
    e0[0] = 1.0;
    auto coeff = dense_solve(ev.vectors, std::move(e0), n);
    std::vector<complex128> exp_dH_e0(static_cast<std::size_t>(n), complex128(0.));
    for (int64 j = 0; j < n; ++j) {
        complex128 scale = std::exp(ev.values[static_cast<std::size_t>(j)] * dlt) *
                           coeff[static_cast<std::size_t>(j)];
        for (int64 i = 0; i < n; ++i) {
            exp_dH_e0[static_cast<std::size_t>(i)] += ev.vectors[idx2(i, j, n)] * scale;
        }
    }
    _result_norm = dense_norm(exp_dH_e0);
    for (auto& z : exp_dH_e0) {
        z /= _result_norm;
    }
    _result_krylov = std::move(exp_dH_e0);
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
}

bool
ArnoldiEvolution::_converged(int64 k)
{
    return std::abs(_result_krylov[static_cast<std::size_t>(k)]) < P_tol;
}

VectorLike::Ptr
ArnoldiEvolution::_calc_result_full_evolution(int64 N)
{
    auto const& cache = _cache;
    if (static_cast<int64>(cache.size()) < N) {
        throw std::runtime_error("ArnoldiEvolution: Krylov basis shorter than N");
    }
    auto psif = scaled_num(cache[0], _result_krylov[0]);
    for (int64 k = 1; k < N; ++k) {
        psif = add_vec(std::move(psif),
                       scaled_num(cache[static_cast<std::size_t>(k)],
                                  _result_krylov[static_cast<std::size_t>(k)]));
    }
    auto psif_norm = abs_number(norm(VectorLikeCPtr(psif)));
    if (std::abs(1.0 - psif_norm) > 1.0e-5) {
        krylov_log_warning(
          std::format("poorly conditioned H in ArnoldiEvolution! |psi|={}", psif_norm));
    }
    return scalar_multiply(as_scalar(*psif, 1.0 / psif_norm), psif);
}

LanczosGroundState::LanczosGroundState(LinearOperator::Ptr H_,
                                       VectorLike::Ptr psi0_,
                                       py::object options_)
  : KrylovBased(std::move(H_), std::move(psi0_), options_, Dtype::Float64, Dtype::Float64)
{
    E_tol = dict_get<float64>(options, "E_tol", std::numeric_limits<float64>::infinity());
    N_cache = dict_get<int64>(options, "N_cache", N_max);
    if (N_cache < 2) {
        throw std::invalid_argument("Need to cache at least two vectors.");
    }
}

std::tuple<float64, VectorLike::Ptr, int64>
LanczosGroundState::run()
{
    auto N = _build_krylov();
    auto E0 = Es_at(N - 1, 0).real();
    if (N > 1) {
        krylov_log_debug(
          std::format("Lanczos N={}, gap={:.3e}, DeltaE0={:.3e}, _result_krylov[-1]={:.3e}",
                      N,
                      Es_at(N - 1, 1).real() - E0,
                      Es_at(N - 2, 0).real() - E0,
                      _result_krylov.back().real()));
    } else {
        krylov_log_debug(std::format("Lanczos N={}, first alpha={:.3e}, beta={:.3e}",
                                     N,
                                     h_krylov(0, 0).real(),
                                     h_krylov(0, 1).real()));
    }
    if (E_shift.has_value()) {
        E0 -= *E_shift;
    }
    if (N == 1) {
        return { E0, psi0, N };
    }
    return { E0, _calc_result_full(N), N };
}

int64
LanczosGroundState::_build_krylov()
{
    auto w = psi0;
    auto beta = abs_number(norm(VectorLikeCPtr(w)));
    if (beta < _cutoff) {
        throw std::invalid_argument(std::format("Norm of self.psi0 too small: {}", beta));
    }
    psi0 = scaled_num(w, 1.0 / beta);
    if (!_psi0_norm.has_value()) {
        // this is only needed for normalization in LanczosEvolution
        _psi0_norm = beta;
    }
    int64 k = 0;
    int64 performed = 0;
    for (; k < N_max; ++k) {
        w = scalar_multiply(as_scalar(*w, 1.0 / beta), w);
        _to_cache(w);
        w = H->matvec(w);
        auto alpha = inner(w, VectorLikeCPtr(_cache.back())).real().as_float64();
        set_h_krylov(k, k, alpha);
        _calc_result_krylov(k);
        w = _cache.back()->axpy(as_scalar(*w, -alpha), w);
        if (reortho) {
            for (std::size_t i = 0; i + 1 < _cache.size(); ++i) {
                auto ov = inner(VectorLikeCPtr(_cache[i]), w);
                w = _cache[i]->axpy(-ov, w);
            }
        } else if (k > 0) {
            w = _cache[_cache.size() - 2]->axpy(as_scalar(*w, -beta), w);
        }
        beta = abs_number(norm(VectorLikeCPtr(w)));
        set_h_krylov(k, k + 1, beta);
        set_h_krylov(k + 1, k, beta); // needed for the next step and convergence criteria
        performed = k + 1;
        if (std::abs(beta) < _cutoff || (k + 1 >= N_min && _converged(k))) {
            break;
        }
    }
    return performed;
}

bool
LanczosGroundState::_converged(int64 k)
{
    auto v0k = _result_krylov[static_cast<std::size_t>(k)];
    auto RitzRes = std::abs(v0k) * std::abs(h_krylov(k, k + 1));
    auto gap = std::max(Es_at(k, 1).real() - Es_at(k, 0).real(), min_gap);
    auto P_err = (RitzRes / gap) * (RitzRes / gap);
    auto Delta_E0 = Es_at(k - 1, 0).real() - Es_at(k, 0).real();
    return P_err < P_tol && Delta_E0 < E_tol;
}

VectorLike::Ptr
LanczosGroundState::_rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max_rebuild)
{
    auto const& vf = _result_krylov;
    auto w = psi0;
    float64 beta = 0.;
    for (int64 k = 0; k < N_max_rebuild; ++k) {
        _to_cache(w);
        w = H->matvec(w);
        auto alpha = h_krylov(k, k).real();
        w = _cache.back()->axpy(as_scalar(*w, -alpha), w);
        if (reortho) {
            for (std::size_t i = 0; i + 1 < _cache.size(); ++i) {
                auto ov = inner(VectorLikeCPtr(_cache[i]), w);
                w = _cache[i]->axpy(-ov, w);
            }
        } else if (k > 0) {
            w = _cache[_cache.size() - 2]->axpy(as_scalar(*w, -beta), w);
        }
        beta = h_krylov(k, k + 1).real(); // = norm(w)
        w = scalar_multiply(as_scalar(*w, 1.0 / beta), w);
        psif = add_vec(std::move(psif), scaled_num(w, vf[static_cast<std::size_t>(k + 1)]));
    }
    return psif;
}

void
LanczosGroundState::_calc_result_krylov(int64 k)
{
    if (k == 0) {
        set_Es(0, 0, h_krylov(0, 0));
        _result_krylov = { 1. };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto h = extract_submatrix(_h_krylov, h_stride(), n, /*as_real=*/true);
    auto ev = dense_eigh(h, n);
    for (int64 j = 0; j < n; ++j) {
        set_Es(k, j, ev.values[static_cast<std::size_t>(j)]);
    }
    _result_krylov.resize(static_cast<std::size_t>(n));
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
    for (int64 row = 0; row < n; ++row) {
        _result_krylov[static_cast<std::size_t>(row)] = ev.vectors[idx2(row, 0, n)];
    }
}

LanczosEvolution::LanczosEvolution(LinearOperator::Ptr H_,
                                   VectorLike::Ptr psi0_,
                                   py::object options_)
  : LanczosGroundState(std::move(H_), std::move(psi0_), std::move(options_))
{
    _result_norm = 1.0;
    delta = std::nullopt;
}

std::tuple<VectorLike::Ptr, int64>
LanczosEvolution::run(complex128 delta_, std::optional<bool> normalize)
{
    delta = delta_;
    _reset_krylov_state();
    auto N = _build_krylov();
    if (N > 1) {
        krylov_log_debug(std::format(
          "Lanczos N={}, |result_krylov[-1]|={:.3e}", N, std::abs(_result_krylov.back())));
    } else {
        krylov_log_debug(std::format("Lanczos N={}, first alpha={:.3e}, beta={:.3e}",
                                     N,
                                     h_krylov(0, 0).real(),
                                     h_krylov(0, 1).real()));
    }
    VectorLike::Ptr result_full;
    if (N == 1) {
        result_full = scaled_num(psi0, _result_krylov[0]); // _result_krylov[0] is only a phase
    } else {
        result_full = _calc_result_full(N);
    }
    // result_full is normalized at this point
    bool do_normalize = normalize.value_or(delta_.real() == 0.0);
    if (do_normalize) {
        return { std::move(result_full), N };
    }
    auto scale = (_psi0_norm.value_or(1.0)) * _result_norm;
    return { scaled_num(result_full, scale), N };
}

void
LanczosEvolution::_calc_result_krylov(int64 k)
{
    // self._result_krylov should be a normalized vector.
    auto dlt = *delta;
    if (k == 0) {
        auto E = h_krylov(0, 0);
        auto exp_dE = std::exp(dlt * E);
        _result_norm = std::abs(exp_dE);
        _result_krylov = { exp_dE / _result_norm };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto h = extract_submatrix(_h_krylov, h_stride(), n, /*as_real=*/true);
    auto ev = dense_eigh(h, n);
    // exp(delta * H) e0 = V * diag(exp(delta * E)) * V^H e0
    // V^H e0 = conj(first row of V) for real orthonormal V: conj(V[0, :])
    std::vector<complex128> exp_dH_e0(static_cast<std::size_t>(n), complex128(0.));
    for (int64 j = 0; j < n; ++j) {
        complex128 coeff = std::conj(ev.vectors[idx2(0, j, n)]);
        complex128 scale = std::exp(ev.values[static_cast<std::size_t>(j)] * dlt) * coeff;
        for (int64 i = 0; i < n; ++i) {
            exp_dH_e0[static_cast<std::size_t>(i)] += ev.vectors[idx2(i, j, n)] * scale;
        }
    }
    _result_norm = dense_norm(exp_dH_e0);
    for (auto& z : exp_dH_e0) {
        z /= _result_norm;
    }
    _result_krylov = std::move(exp_dH_e0);
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
}

bool
LanczosEvolution::_converged(int64 k)
{
    return std::abs(_result_krylov[static_cast<std::size_t>(k)]) < P_tol;
}

std::tuple<float64, VectorLike::Ptr, int64>
lanczos(LinearOperator::Ptr H, VectorLike::Ptr psi, py::object options)
{
    return LanczosGroundState(std::move(H), std::move(psi), std::move(options)).run();
}

} // namespace lasyten
