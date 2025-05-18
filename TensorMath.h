// TensorMath.h - v0.2.0
// Mathematical operations for tensors

#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include "Tensor.h"
#include "TensorOps.h"
#include <cmath>
#include <limits>
#include <functional>

namespace tensor {
    namespace math {

        // Element-wise operations with automatic type promotion
        template<typename T, typename UnaryOp>
        Tensor<T> unary_op(const Tensor<T>& t, UnaryOp op) {
            Tensor<T> result(t.shape());
            const T* src = t.data();
            T* dst = result.data();

            for (size_t i = 0; i < t.size(); ++i) {
                dst[i] = op(src[i]);
            }

            return result;
        }

        // Exponential
        template<typename T>
        Tensor<T> exp(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::exp(x); });
        }

        // Natural logarithm
        template<typename T>
        Tensor<T> log(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x <= T(0)) throw std::domain_error("Log of non-positive number");
                return std::log(x);
                });
        }

        // Base-10 logarithm
        template<typename T>
        Tensor<T> log10(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x <= T(0)) throw std::domain_error("Log10 of non-positive number");
                return std::log10(x);
                });
        }

        // Base-2 logarithm
        template<typename T>
        Tensor<T> log2(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x <= T(0)) throw std::domain_error("Log2 of non-positive number");
                return std::log2(x);
                });
        }

        // Square root
        template<typename T>
        Tensor<T> sqrt(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x < T(0)) throw std::domain_error("Square root of negative number");
                return std::sqrt(x);
                });
        }

        // Cube root
        template<typename T>
        Tensor<T> cbrt(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::cbrt(x); });
        }

        // Absolute value
        template<typename T>
        Tensor<T> abs(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::abs(x); });
        }

        // Sine
        template<typename T>
        Tensor<T> sin(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::sin(x); });
        }

        // Cosine
        template<typename T>
        Tensor<T> cos(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::cos(x); });
        }

        // Tangent
        template<typename T>
        Tensor<T> tan(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::tan(x); });
        }

        // Hyperbolic sine
        template<typename T>
        Tensor<T> sinh(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::sinh(x); });
        }

        // Hyperbolic cosine
        template<typename T>
        Tensor<T> cosh(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::cosh(x); });
        }

        // Hyperbolic tangent
        template<typename T>
        Tensor<T> tanh(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::tanh(x); });
        }

        // Inverse sine
        template<typename T>
        Tensor<T> asin(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x < T(-1) || x > T(1)) throw std::domain_error("Asin domain error");
                return std::asin(x);
                });
        }

        // Inverse cosine
        template<typename T>
        Tensor<T> acos(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x < T(-1) || x > T(1)) throw std::domain_error("Acos domain error");
                return std::acos(x);
                });
        }

        // Inverse tangent
        template<typename T>
        Tensor<T> atan(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::atan(x); });
        }

        // Inverse tangent of y/x
        template<typename T>
        Tensor<T> atan2(const Tensor<T>& y, const Tensor<T>& x) {
            return element_wise_op(y, x, [](T y_val, T x_val) -> T {
                return std::atan2(y_val, x_val);
                });
        }

        // Error function
        template<typename T>
        Tensor<T> erf(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::erf(x); });
        }

        // Complementary error function
        template<typename T>
        Tensor<T> erfc(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::erfc(x); });
        }

        // Gamma function
        template<typename T>
        Tensor<T> tgamma(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::tgamma(x); });
        }

        // Log-gamma function
        template<typename T>
        Tensor<T> lgamma(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::lgamma(x); });
        }

        // Ceiling
        template<typename T>
        Tensor<T> ceil(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::ceil(x); });
        }

        // Floor
        template<typename T>
        Tensor<T> floor(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::floor(x); });
        }

        // Round to nearest integer
        template<typename T>
        Tensor<T> round(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::round(x); });
        }

        // Truncate
        template<typename T>
        Tensor<T> trunc(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::trunc(x); });
        }

        // Sigmoid function
        template<typename T>
        Tensor<T> sigmoid(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return T(1) / (T(1) + std::exp(-x)); });
        }

        // ReLU function
        template<typename T>
        Tensor<T> relu(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::max(T(0), x); });
        }

        // Leaky ReLU
        template<typename T>
        Tensor<T> leaky_relu(const Tensor<T>& t, T alpha = T(0.01)) {
            return unary_op(t, [alpha](T x) -> T {
                return x >= T(0) ? x : alpha * x;
                });
        }

        // Softplus
        template<typename T>
        Tensor<T> softplus(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T { return std::log(T(1) + std::exp(x)); });
        }

        // Sign function
        template<typename T>
        Tensor<T> sign(const Tensor<T>& t) {
            return unary_op(t, [](T x) -> T {
                if (x > T(0)) return T(1);
                if (x < T(0)) return T(-1);
                return T(0);
                });
        }

        // Clip values to range [min_val, max_val]
        template<typename T>
        Tensor<T> clip(const Tensor<T>& t, T min_val, T max_val) {
            if (min_val > max_val) {
                throw std::invalid_argument("min_val must be less than or equal to max_val");
            }

            return unary_op(t, [min_val, max_val](T x) -> T {
                return std::max(min_val, std::min(x, max_val));
                });
        }

        // Calculate L1 norm (sum of absolute values)
        template<typename T>
        T l1_norm(const Tensor<T>& t) {
            T sum = T(0);
            for (size_t i = 0; i < t.size(); ++i) {
                sum += std::abs(t.data()[i]);
            }
            return sum;
        }

        // Calculate L2 norm (Euclidean norm)
        template<typename T>
        T l2_norm(const Tensor<T>& t) {
            T sum_squares = T(0);
            for (size_t i = 0; i < t.size(); ++i) {
                T val = t.data()[i];
                sum_squares += val * val;
            }
            return std::sqrt(sum_squares);
        }

        // Calculate infinity norm (max absolute value)
        template<typename T>
        T inf_norm(const Tensor<T>& t) {
            T max_abs = T(0);
            for (size_t i = 0; i < t.size(); ++i) {
                max_abs = std::max(max_abs, std::abs(t.data()[i]));
            }
            return max_abs;
        }

        // Normalize tensor (L2 normalization)
        template<typename T>
        Tensor<T> normalize(const Tensor<T>& t) {
            T norm = l2_norm(t);
            if (norm <= std::numeric_limits<T>::epsilon()) {
                throw std::domain_error("Cannot normalize tensor with zero norm");
            }

            return t * (T(1) / norm);
        }

    } // namespace math
} // namespace tensor

#endif // TENSOR_MATH_H