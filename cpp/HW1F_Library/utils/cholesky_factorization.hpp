#pragma once
// =============================================================================
// HW1F Library - Cholesky Factorization with Reuse
// 
// Separates factorization from solve for efficiency when solving 
// multiple right-hand sides with the same matrix (e.g., in IFT).
//
// Features:
//   - Optional regularization (eps * I) for numerical stability
//   - Reuse factorization L to solve multiple systems
//   - Support for symmetric positive definite matrices
//
// =============================================================================

#include <vector>
#include <cmath>
#include <stdexcept>

namespace hw1f {

// =============================================================================
// Cholesky Factorization Class
// =============================================================================

class CholeskyFactorization {
public:
    // Default constructor (empty factorization)
    CholeskyFactorization() : n_(0), factored_(false), eps_reg_(0.0) {}
    
    // Factor matrix A with optional regularization: A_reg = A + eps * I
    // Throws if matrix is not positive definite (after regularization)
    void factor(
        const std::vector<std::vector<double>>& A,
        double eps_reg = 0.0
    ) {
        n_ = A.size();
        eps_reg_ = eps_reg;
        
        if (n_ == 0) {
            throw std::runtime_error("Cannot factor empty matrix");
        }
        if (A[0].size() != n_) {
            throw std::runtime_error("Matrix must be square for Cholesky factorization");
        }
        
        // Initialize L
        L_.assign(n_, std::vector<double>(n_, 0.0));
        
        // Cholesky decomposition: A = L * L^T
        for (size_t i = 0; i < n_; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L_[i][k] * L_[j][k];
                }
                
                if (i == j) {
                    // Apply regularization to diagonal
                    double diag = A[i][i] + eps_reg_ - sum;
                    if (diag <= 0) {
                        throw std::runtime_error(
                            "Matrix is not positive definite at index " + std::to_string(i) +
                            " (diag = " + std::to_string(diag) + ")"
                        );
                    }
                    L_[i][j] = std::sqrt(diag);
                } else {
                    if (std::abs(L_[j][j]) < 1e-14) {
                        throw std::runtime_error("Zero diagonal in Cholesky factorization");
                    }
                    L_[i][j] = (A[i][j] - sum) / L_[j][j];
                }
            }
        }
        
        factored_ = true;
    }
    
    // Solve L * L^T * x = b using pre-computed factorization
    std::vector<double> solve(const std::vector<double>& b) const {
        if (!factored_) {
            throw std::runtime_error("Must call factor() before solve()");
        }
        if (b.size() != n_) {
            throw std::runtime_error(
                "RHS has size " + std::to_string(b.size()) + 
                ", expected " + std::to_string(n_)
            );
        }
        
        // Forward substitution: L * y = b
        std::vector<double> y(n_);
        for (size_t i = 0; i < n_; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += L_[i][j] * y[j];
            }
            y[i] = (b[i] - sum) / L_[i][i];
        }
        
        // Backward substitution: L^T * x = y
        std::vector<double> x(n_);
        for (int i = static_cast<int>(n_) - 1; i >= 0; --i) {
            double sum = 0.0;
            for (size_t j = i + 1; j < n_; ++j) {
                sum += L_[j][i] * x[j];
            }
            x[i] = (y[i] - sum) / L_[i][i];
        }
        
        return x;
    }
    
    // Get the factorization L (read-only)
    const std::vector<std::vector<double>>& L() const { return L_; }
    
    // Get matrix dimension
    size_t size() const { return n_; }
    
    // Check if factored
    bool isFactored() const { return factored_; }
    
    // Get regularization used
    double regularization() const { return eps_reg_; }
    
    // Compute determinant: det(A) = det(L)^2 = (prod L_ii)^2
    double logDeterminant() const {
        if (!factored_) {
            throw std::runtime_error("Must call factor() before logDeterminant()");
        }
        double logDet = 0.0;
        for (size_t i = 0; i < n_; ++i) {
            logDet += 2.0 * std::log(L_[i][i]);
        }
        return logDet;
    }
    
    // Compute inverse of original matrix using factorization
    std::vector<std::vector<double>> inverse() const {
        if (!factored_) {
            throw std::runtime_error("Must call factor() before inverse()");
        }
        
        std::vector<std::vector<double>> inv(n_, std::vector<double>(n_, 0.0));
        
        // Solve A * inv_col_i = e_i for each column
        for (size_t i = 0; i < n_; ++i) {
            std::vector<double> e(n_, 0.0);
            e[i] = 1.0;
            std::vector<double> col = solve(e);
            for (size_t j = 0; j < n_; ++j) {
                inv[j][i] = col[j];
            }
        }
        
        return inv;
    }

private:
    size_t n_;
    std::vector<std::vector<double>> L_;
    bool factored_;
    double eps_reg_;
};

// =============================================================================
// Convenience Function: Factor and Solve in One Call
// (For when factorization reuse is not needed)
// =============================================================================

inline std::vector<double> solveCholeskyReg(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    double eps_reg = 0.0
) {
    CholeskyFactorization chol;
    chol.factor(A, eps_reg);
    return chol.solve(b);
}

} // namespace hw1f
