// TensorOps.h - v2.0.0
// Basic tensor operations

#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "Tensor.h"
#include "TensorBroadcast.h"
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <cmath>

namespace tensor {

    //==============================================================================
    // ELEMENT-WISE OPERATIONS
    //==============================================================================

    /**
     * @brief Apply a binary operation element-wise to two tensors
     */
    template<typename T, typename BinaryOp>
    Tensor<T> element_wise_op(const Tensor<T>& a, const Tensor<T>& b, BinaryOp op) {
        // Check if shapes are identical
        if (a.shape() == b.shape()) {
            Tensor<T> result(a.shape());
            for (size_t i = 0; i < a.size(); ++i) {
                result.data()[i] = op(a.data()[i], b.data()[i]);
            }
            return result;
        }

        // Try broadcasting
        std::vector<size_t> broadcast_shape;
        try {
            broadcast_shape = TensorBroadcaster::get_broadcast_shape(a.shape(), b.shape());
        }
        catch (const std::invalid_argument& e) {
            throw std::invalid_argument("Cannot broadcast tensors for element-wise operation: " +
                std::string(e.what()));
        }

        Tensor<T> result(broadcast_shape);

        // Iterate through all indices in the broadcast shape
        std::vector<size_t> idx(broadcast_shape.size(), 0);
        bool done = false;

        while (!done) {
            // Map indices to input tensors
            std::vector<size_t> a_idx = TensorBroadcaster::map_indices(idx, a.shape());
            std::vector<size_t> b_idx = TensorBroadcaster::map_indices(idx, b.shape());

            // Apply operation
            result.at(idx) = op(a.at(a_idx), b.at(b_idx));

            // Increment indices
            for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                idx[i]++;
                if (idx[i] < broadcast_shape[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    done = true;
                }
            }
        }

        return result;
    }

    /**
     * @brief Apply a unary operation element-wise to a tensor
     */
    template<typename T, typename UnaryOp>
    Tensor<T> element_wise_op(const Tensor<T>& a, UnaryOp op) {
        Tensor<T> result(a.shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data()[i] = op(a.data()[i]);
        }
        return result;
    }

    /**
     * @brief Apply a binary operation with a scalar to a tensor
     */
    template<typename T, typename BinaryOp>
    Tensor<T> element_wise_op_scalar(const Tensor<T>& a, T scalar, BinaryOp op) {
        Tensor<T> result(a.shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data()[i] = op(a.data()[i], scalar);
        }
        return result;
    }

    //==============================================================================
    // COMMON ELEMENT-WISE OPERATIONS
    //==============================================================================

    /**
     * @brief Add two tensors element-wise
     */
    template<typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        return element_wise_op(a, b, std::plus<T>());
    }

    /**
     * @brief Subtract one tensor from another element-wise
     */
    template<typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
        return element_wise_op(a, b, std::minus<T>());
    }

    /**
     * @brief Multiply two tensors element-wise
     */
    template<typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        return element_wise_op(a, b, std::multiplies<T>());
    }

    /**
     * @brief Divide one tensor by another element-wise
     */
    template<typename T>
    Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b) {
        return element_wise_op(a, b, [](T x, T y) {
            if (y == T(0)) throw std::domain_error("Division by zero");
            return x / y;
            });
    }

    /**
     * @brief Raise tensor elements to power of another tensor's elements
     */
    template<typename T>
    Tensor<T> power(const Tensor<T>& a, const Tensor<T>& b) {
        return element_wise_op(a, b, [](T x, T y) {
            return std::pow(x, y);
            });
    }

    //==============================================================================
    // SCALAR OPERATIONS
    //==============================================================================

    /**
     * @brief Add scalar to tensor
     */
    template<typename T>
    Tensor<T> add(const Tensor<T>& a, T scalar) {
        return element_wise_op_scalar(a, scalar, std::plus<T>());
    }

    /**
     * @brief Subtract scalar from tensor
     */
    template<typename T>
    Tensor<T> subtract(const Tensor<T>& a, T scalar) {
        return element_wise_op_scalar(a, scalar, std::minus<T>());
    }

    /**
     * @brief Multiply tensor by scalar
     */
    template<typename T>
    Tensor<T> multiply(const Tensor<T>& a, T scalar) {
        return element_wise_op_scalar(a, scalar, std::multiplies<T>());
    }

    /**
     * @brief Divide tensor by scalar
     */
    template<typename T>
    Tensor<T> divide(const Tensor<T>& a, T scalar) {
        if (scalar == T(0)) throw std::domain_error("Division by zero");
        return element_wise_op_scalar(a, scalar, std::divides<T>());
    }

    /**
     * @brief Raise tensor elements to power of scalar
     */
    template<typename T>
    Tensor<T> power(const Tensor<T>& a, T exponent) {
        return element_wise_op_scalar(a, exponent, [](T x, T y) {
            return std::pow(x, y);
            });
    }

    //==============================================================================
    // MATRIX OPERATIONS
    //==============================================================================

    /**
     * @brief Matrix multiplication
     */
    template<typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
        // Check dimensions
        if (a.ndim() < 2 || b.ndim() < 2) {
            throw std::invalid_argument("Both tensors must have at least 2 dimensions for matmul");
        }

        // Check inner dimensions match
        const auto& a_shape = a.shape();
        const auto& b_shape = b.shape();

        size_t a_inner_dim = a_shape[a_shape.size() - 1];
        size_t b_inner_dim = b_shape[b_shape.size() - 2];

        if (a_inner_dim != b_inner_dim) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication: " +
                std::to_string(a_inner_dim) + " vs " +
                std::to_string(b_inner_dim));
        }

        // Calculate batch dimensions
        size_t a_batch_dims = a_shape.size() - 2;
        size_t b_batch_dims = b_shape.size() - 2;
        size_t max_batch_dims = std::max(a_batch_dims, b_batch_dims);

        // Build broadcast batch shape
        std::vector<size_t> batch_shape;
        for (size_t i = 0; i < max_batch_dims; ++i) {
            size_t a_dim = (i < a_batch_dims) ? a_shape[i] : 1;
            size_t b_dim = (i < b_batch_dims) ? b_shape[i] : 1;

            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw std::invalid_argument("Batch dimensions must be broadcastable");
            }

            batch_shape.push_back(std::max(a_dim, b_dim));
        }

        // Create result shape: batch_dims + [a's rows, b's cols]
        std::vector<size_t> result_shape = batch_shape;
        result_shape.push_back(a_shape[a_shape.size() - 2]);  // a's rows
        result_shape.push_back(b_shape[b_shape.size() - 1]);  // b's cols

        Tensor<T> result(result_shape);

        // Handle each batch element
        std::vector<size_t> batch_idx(max_batch_dims, 0);
        bool batch_done = false;

        while (!batch_done) {
            // Map batch indices to a and b
            std::vector<size_t> a_batch_idx, b_batch_idx;

            for (size_t i = 0; i < max_batch_dims; ++i) {
                if (i < a_batch_dims) {
                    a_batch_idx.push_back((a_shape[i] == 1) ? 0 : batch_idx[i]);
                }
                if (i < b_batch_dims) {
                    b_batch_idx.push_back((b_shape[i] == 1) ? 0 : batch_idx[i]);
                }
            }

            // Perform matrix multiplication for this batch
            for (size_t i = 0; i < result_shape[result_shape.size() - 2]; ++i) {
                for (size_t j = 0; j < result_shape[result_shape.size() - 1]; ++j) {
                    T sum = T(0);

                    for (size_t k = 0; k < a_inner_dim; ++k) {
                        // Create full indices for a and b
                        std::vector<size_t> a_idx = a_batch_idx;
                        a_idx.push_back(i);
                        a_idx.push_back(k);

                        std::vector<size_t> b_idx = b_batch_idx;
                        b_idx.push_back(k);
                        b_idx.push_back(j);

                        sum += a.at(a_idx) * b.at(b_idx);
                    }

                    // Store result
                    std::vector<size_t> result_idx = batch_idx;
                    result_idx.push_back(i);
                    result_idx.push_back(j);
                    result.at(result_idx) = sum;
                }
            }

            // Increment batch indices
            for (int i = static_cast<int>(batch_idx.size()) - 1; i >= 0; --i) {
                batch_idx[i]++;
                if (batch_idx[i] < batch_shape[i]) {
                    break;
                }
                batch_idx[i] = 0;
                if (i == 0) {
                    batch_done = true;
                }
            }
        }

        return result;
    }

    /**
     * @brief Compute dot product of vectors (1D tensors)
     */
    template<typename T>
    T dot(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.ndim() != 1 || b.ndim() != 1) {
            throw std::invalid_argument("Both tensors must be 1D for dot product");
        }

        if (a.shape()[0] != b.shape()[0]) {
            throw std::invalid_argument("Vectors must have the same length for dot product");
        }

        T result = T(0);
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            result += a.at(i) * b.at(i);
        }

        return result;
    }

    //==============================================================================
    // TENSOR TRANSPOSITION
    //==============================================================================

    /**
     * @brief Transpose a tensor by permuting its dimensions
     */
    template<typename T>
    Tensor<T> transpose(const Tensor<T>& tensor, const std::vector<size_t>& dims) {
        // Validate dimensions
        if (dims.size() != tensor.ndim()) {
            throw std::invalid_argument("Transpose dimensions must match tensor rank");
        }

        // Check for duplicates and bounds
        std::vector<bool> used(dims.size(), false);
        for (size_t dim : dims) {
            if (dim >= dims.size() || used[dim]) {
                throw std::invalid_argument("Invalid dimensions for transpose");
            }
            used[dim] = true;
        }

        // Create new shape
        std::vector<size_t> new_shape(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            new_shape[i] = tensor.shape()[dims[i]];
        }

        Tensor<T> result(new_shape);

        // Copy data with transposed indices
        std::vector<size_t> idx(tensor.ndim(), 0);
        bool done = false;

        while (!done) {
            // Map original indices to transposed indices
            std::vector<size_t> result_idx(dims.size());
            for (size_t i = 0; i < dims.size(); ++i) {
                result_idx[i] = idx[dims[i]];
            }

            // Copy value
            result.at(result_idx) = tensor.at(idx);

            // Increment indices
            for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                idx[i]++;
                if (idx[i] < tensor.shape()[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    done = true;
                }
            }
        }

        return result;
    }

    /**
     * @brief Standard 2D matrix transpose
     */
    template<typename T>
    Tensor<T> transpose(const Tensor<T>& tensor) {
        if (tensor.ndim() != 2) {
            throw std::invalid_argument("Default transpose is for 2D tensors only");
        }

        return transpose(tensor, { 1, 0 });
    }

} // namespace tensor

#endif // TENSOR_OPS_H