// EnhancedMath.h - Advanced mathematical operations with SIMD optimizations

#ifndef ENHANCED_MATH_H
#define ENHANCED_MATH_H

#include "TensorCore.h"
#include <cmath>
#include <functional>
#include <thread>

namespace tensor {
    namespace math {

        // Parallel execution control
        struct ParallelOptions {
            size_t num_threads = 0; // 0 means auto-detect
            size_t min_elements_per_thread = 1000;
            bool force_sequential = false;
        };

        // Enhanced element-wise operations with SIMD support
        template<typename T, typename UnaryOp>
        TensorCore<T> unary_op(const TensorCore<T>& t, UnaryOp op,
            const ParallelOptions& options = {});

        template<typename T, typename BinaryOp>
        TensorCore<T> binary_op(const TensorCore<T>& a, const TensorCore<T>& b,
            BinaryOp op, const ParallelOptions& options = {});

        // Optimized common operations
        template<typename T>
        TensorCore<T> exp(const TensorCore<T>& t, const ParallelOptions& options = {});

        template<typename T>
        TensorCore<T> log(const TensorCore<T>& t, const ParallelOptions& options = {});

        template<typename T>
        TensorCore<T> sigmoid(const TensorCore<T>& t, const ParallelOptions& options = {});

        template<typename T>
        TensorCore<T> tanh(const TensorCore<T>& t, const ParallelOptions& options = {});

        // Batch matrix operations
        template<typename T>
        TensorCore<T> matmul(const TensorCore<T>& a, const TensorCore<T>& b,
            const ParallelOptions& options = {});

        template<typename T>
        TensorCore<T> batch_matmul(const TensorCore<T>& a, const TensorCore<T>& b,
            const ParallelOptions& options = {});

        // Advanced operations for deep learning
        template<typename T>
        TensorCore<T> softmax(const TensorCore<T>& t, int axis = -1,
            const ParallelOptions& options = {});

        template<typename T>
        TensorCore<T> layer_norm(const TensorCore<T>& t, T epsilon = T(1e-5),
            const ParallelOptions& options = {});

        // Device-specific implementations
        namespace cpu {
            // CPU-optimized implementations
        }

        namespace cuda {
            // CUDA-optimized implementations
        }

    } // namespace math
} // namespace tensor

#endif // ENHANCED_MATH_H