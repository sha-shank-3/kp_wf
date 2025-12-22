#pragma once
// =============================================================================
// HW1F Library - Common Utilities and Type Definitions
// =============================================================================

// MSVC compatibility: enable math constants
#define _USE_MATH_DEFINES
#include <cmath>

// Define math constants if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <random>
#include <chrono>
#include <functional>

// XAD headers
#include <XAD/XAD.hpp>

namespace hw1f {

// =============================================================================
// AD Type Definitions
// =============================================================================

using ADouble = xad::fwd_adj<double>;  // Forward mode AD type
using AADouble = xad::adj<double>::active_type;  // Adjoint (reverse) mode AD type
using tape_type = xad::adj<double>::tape_type;

// =============================================================================
// Utility Functions
// =============================================================================

// Get the underlying value from an AD type or double
template<typename T>
inline double value(const T& x) {
    if constexpr (std::is_same_v<T, double>) {
        return x;
    } else {
        return xad::value(x);
    }
}

// Safe exponential to avoid overflow
template<typename T>
inline T safe_exp(const T& x) {
    using std::exp;
    double v = value(x);
    if (v > 700.0) return T(std::exp(700.0));
    if (v < -700.0) return T(0.0);
    return exp(x);
}

// Linear interpolation
template<typename T>
inline T lerp(const T& a, const T& b, double t) {
    return a * (1.0 - t) + b * t;
}

// Find index for interpolation (returns lower index)
inline size_t findIndex(double x, const std::vector<double>& nodes) {
    if (nodes.empty()) throw std::runtime_error("Empty node vector");
    if (x <= nodes.front()) return 0;
    if (x >= nodes.back()) return nodes.size() - 2;
    
    auto it = std::lower_bound(nodes.begin(), nodes.end(), x);
    size_t idx = std::distance(nodes.begin(), it);
    return (idx == 0) ? 0 : idx - 1;
}

// =============================================================================
// Timer for Benchmarking
// =============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// Random Number Generator
// =============================================================================

class RNG {
public:
    explicit RNG(unsigned seed = 12345) : gen_(seed), dist_(0.0, 1.0) {}
    
    double uniform() { return dist_(gen_); }
    
    double normal() {
        // Box-Muller transform
        double u1 = uniform();
        double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    
    std::vector<double> normalVector(size_t n) {
        std::vector<double> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = normal();
        }
        return result;
    }
    
    // Generate matrix of random normals (paths x steps)
    std::vector<std::vector<double>> normalMatrix(size_t paths, size_t steps) {
        std::vector<std::vector<double>> result(paths, std::vector<double>(steps));
        for (size_t i = 0; i < paths; ++i) {
            for (size_t j = 0; j < steps; ++j) {
                result[i][j] = normal();
            }
        }
        return result;
    }
    
    void setSeed(unsigned seed) { gen_.seed(seed); }
    
private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
};

// =============================================================================
// Small Dense Matrix/Vector Operations
// =============================================================================

// Solve A * x = b using Cholesky decomposition (A must be SPD)
inline std::vector<double> solveCholesky(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b
) {
    size_t n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n) {
        throw std::runtime_error("Invalid dimensions for Cholesky solve");
    }
    
    // Cholesky decomposition: A = L * L^T
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag = A[i][i] - sum;
                if (diag <= 0) {
                    throw std::runtime_error("Matrix is not positive definite");
                }
                L[i][j] = std::sqrt(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    
    // Forward substitution: L * y = b
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    
    // Backward substitution: L^T * x = y
    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }
    
    return x;
}

// Cholesky decomposition only (returns L where A = L * L^T)
inline std::vector<std::vector<double>> choleskyDecompose(
    const std::vector<std::vector<double>>& A
) {
    size_t n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag = A[i][i] - sum;
                if (diag <= 0) {
                    throw std::runtime_error("Matrix is not positive definite");
                }
                L[i][j] = std::sqrt(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    
    return L;
}

// Solve L * L^T * x = b given precomputed L
inline std::vector<double> solveCholeskySplit(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& b
) {
    size_t n = L.size();
    
    // Forward substitution: L * y = b
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    
    // Backward substitution: L^T * x = y
    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }
    
    return x;
}

// Matrix-vector multiplication
inline std::vector<double> matVecMult(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& x
) {
    size_t m = A.size();
    size_t n = x.size();
    std::vector<double> result(m, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Transpose matrix
inline std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& A
) {
    if (A.empty()) return {};
    size_t m = A.size();
    size_t n = A[0].size();
    std::vector<std::vector<double>> AT(n, std::vector<double>(m));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

// Dot product
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vector norm
inline double norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

// =============================================================================
// Standard Normal CDF and Inverse
// =============================================================================

inline double normalCDF(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double normalPDF(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Rational approximation for inverse normal CDF
inline double normalInvCDF(double p) {
    if (p <= 0.0) return -1e10;
    if (p >= 1.0) return 1e10;
    
    // Coefficients for rational approximation
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    
    double q, r;
    if (p < 0.02425) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 0.97575) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

} // namespace hw1f
