// TensorReshape.h - v2.0.0
// Reshape operations for tensors

#ifndef TENSOR_RESHAPE_H
#define TENSOR_RESHAPE_H

#include "Tensor.h"
#include "TensorView.h"
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tensor {

    /**
     * @brief Class for tensor reshaping operations
     */
    class TensorReshaper {
    public:
        /**
         * @brief Reshape tensor to new shape
         *
         * @param input Input tensor
         * @param new_shape New shape vector
         * @return Reshaped tensor
         * @throws std::invalid_argument if total size changes
         */
        template<typename T>
        static Tensor<T> reshape(const Tensor<T>& input, const std::vector<size_t>& new_shape) {
            // Calculate final shape with dimension inference
            std::vector<size_t> final_shape = infer_shape(input.shape(), new_shape);

            // Verify sizes match
            size_t new_size = std::accumulate(final_shape.begin(), final_shape.end(),
                size_t(1), std::multiplies<size_t>());
            if (input.size() != new_size) {
                throw std::invalid_argument("Reshape: total size must remain the same");
            }

            // Create new tensor with desired shape
            Tensor<T> output(final_shape);

            // Copy data (assuming both use same memory layout)
            std::copy(input.data(), input.data() + input.size(), output.data());

            return output;
        }

        /**
         * @brief Reshape tensor using initializer list
         */
        template<typename T>
        static Tensor<T> reshape(const Tensor<T>& input, std::initializer_list<size_t> dims) {
            return reshape(input, std::vector<size_t>(dims));
        }

        /**
         * @brief Create a view of tensor with new shape
         *
         * @param input Input tensor
         * @param new_shape New shape vector
         * @return View of tensor with new shape
         * @throws std::invalid_argument if total size changes
         */
        template<typename T>
        static TensorView<T> reshape_view(Tensor<T>& input, const std::vector<size_t>& new_shape) {
            // Calculate final shape with dimension inference
            std::vector<size_t> final_shape = infer_shape(input.shape(), new_shape);

            // Verify sizes match
            size_t new_size = std::accumulate(final_shape.begin(), final_shape.end(),
                size_t(1), std::multiplies<size_t>());
            if (input.size() != new_size) {
                throw std::invalid_argument("Reshape: total size must remain the same");
            }

            // Create view with new shape
            return TensorView<T>(input, final_shape, 0);
        }

        /**
         * @brief Flatten tensor to 1D
         */
        template<typename T>
        static Tensor<T> flatten(const Tensor<T>& input) {
            return reshape(input, { input.size() });
        }

        /**
         * @brief Create a flattened view of tensor
         */
        template<typename T>
        static TensorView<T> flatten_view(Tensor<T>& input) {
            return reshape_view(input, { input.size() });
        }

        /**
         * @brief Flatten specific dimensions in a tensor
         *
         * @param input Input tensor
         * @param start_dim First dimension to flatten
         * @param end_dim Last dimension to flatten (inclusive)
         * @return Tensor with specified dimensions flattened
         */
        template<typename T>
        static Tensor<T> flatten(const Tensor<T>& input, size_t start_dim, size_t end_dim) {
            if (start_dim >= input.ndim() || end_dim >= input.ndim() || start_dim > end_dim) {
                throw std::invalid_argument("Invalid start or end dimension for flatten");
            }

            std::vector<size_t> new_shape;

            // Add dimensions before start_dim
            for (size_t i = 0; i < start_dim; ++i) {
                new_shape.push_back(input.shape()[i]);
            }

            // Compute flattened size
            size_t flat_size = 1;
            for (size_t i = start_dim; i <= end_dim; ++i) {
                flat_size *= input.shape()[i];
            }
            new_shape.push_back(flat_size);

            // Add dimensions after end_dim
            for (size_t i = end_dim + 1; i < input.ndim(); ++i) {
                new_shape.push_back(input.shape()[i]);
            }

            return reshape(input, new_shape);
        }

        /**
         * @brief Remove dimensions of size 1
         */
        template<typename T>
        static Tensor<T> squeeze(const Tensor<T>& input) {
            std::vector<size_t> new_shape;

            for (size_t i = 0; i < input.ndim(); ++i) {
                if (input.shape()[i] != 1) {
                    new_shape.push_back(input.shape()[i]);
                }
            }

            // If all dimensions were 1, keep a single dimension
            if (new_shape.empty()) {
                new_shape.push_back(1);
            }

            return reshape(input, new_shape);
        }

        /**
         * @brief Remove a specific dimension of size 1
         */
        template<typename T>
        static Tensor<T> squeeze(const Tensor<T>& input, size_t dim) {
            if (dim >= input.ndim()) {
                throw std::out_of_range("Dimension out of range for squeeze");
            }

            if (input.shape()[dim] != 1) {
                throw std::invalid_argument("Can only squeeze dimensions of size 1");
            }

            std::vector<size_t> new_shape;

            for (size_t i = 0; i < input.ndim(); ++i) {
                if (i != dim) {
                    new_shape.push_back(input.shape()[i]);
                }
            }

            return reshape(input, new_shape);
        }

        /**
         * @brief Add a dimension of size 1
         */
        template<typename T>
        static Tensor<T> unsqueeze(const Tensor<T>& input, size_t dim) {
            if (dim > input.ndim()) {
                throw std::out_of_range("Dimension out of range for unsqueeze");
            }

            std::vector<size_t> new_shape = input.shape();
            new_shape.insert(new_shape.begin() + dim, 1);

            return reshape(input, new_shape);
        }

        /**
         * @brief Create a view with a different shape (alias for reshape_view)
         */
        template<typename T>
        static TensorView<T> view(Tensor<T>& input, const std::vector<size_t>& new_shape) {
            return reshape_view(input, new_shape);
        }

        /**
         * @brief Expand dimensions to match target shape (broadcast)
         */
        template<typename T>
        static Tensor<T> expand(const Tensor<T>& input, const std::vector<size_t>& target_shape) {
            // Check if dimensions are compatible for broadcasting
            if (!is_expandable(input.shape(), target_shape)) {
                throw std::invalid_argument("Cannot expand tensor to target shape");
            }

            // Create output tensor
            Tensor<T> output(target_shape);

            // Copy data with broadcasting
            std::vector<size_t> out_idx(target_shape.size(), 0);
            bool done = false;

            while (!done) {
                // Map input indices
                std::vector<size_t> in_idx = map_broadcast_indices(out_idx, input.shape(), target_shape);

                // Copy value
                output.at(out_idx) = input.at(in_idx);

                // Increment output indices
                for (int i = static_cast<int>(out_idx.size()) - 1; i >= 0; --i) {
                    out_idx[i]++;
                    if (out_idx[i] < target_shape[i]) {
                        break;
                    }
                    out_idx[i] = 0;
                    if (i == 0) {
                        done = true;
                    }
                }
            }

            return output;
        }

    private:
        /**
         * @brief Infer shape when a dimension is specified as -1
         */
        static std::vector<size_t> infer_shape(
            const std::vector<size_t>& original_shape,
            const std::vector<size_t>& new_shape) {

            // Check if there's a dimension to infer (-1)
            int infer_dim = -1;
            size_t product = 1;

            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (new_shape[i] == static_cast<size_t>(-1)) {  // Detect -1 (will wrap to max size_t)
                    if (infer_dim != -1) {
                        throw std::invalid_argument("Can only specify one dimension as -1");
                    }
                    infer_dim = static_cast<int>(i);
                }
                else {
                    product *= new_shape[i];
                }
            }

            // If no inference needed, return the shape as is
            if (infer_dim == -1) {
                return new_shape;
            }

            // Compute the inferred dimension
            size_t original_size = std::accumulate(original_shape.begin(), original_shape.end(),
                size_t(1), std::multiplies<size_t>());
            if (original_size % product != 0) {
                throw std::invalid_argument("Cannot infer size for dimension");
            }

            size_t inferred_size = original_size / product;

            // Create the final shape with inferred dimension
            std::vector<size_t> final_shape = new_shape;
            final_shape[infer_dim] = inferred_size;

            return final_shape;
        }

        /**
         * @brief Check if a shape can be expanded to target shape
         */
        static bool is_expandable(
            const std::vector<size_t>& shape,
            const std::vector<size_t>& target_shape) {

            // Target must have at least as many dimensions
            if (target_shape.size() < shape.size()) {
                return false;
            }

            // Check dimensions from right to left
            size_t offset = target_shape.size() - shape.size();

            for (size_t i = 0; i < shape.size(); ++i) {
                // Each dimension must be either same size or 1 (for broadcasting)
                if (shape[i] != target_shape[i + offset] && shape[i] != 1) {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Map output indices to input indices for broadcasting
         */
        static std::vector<size_t> map_broadcast_indices(
            const std::vector<size_t>& out_idx,
            const std::vector<size_t>& in_shape,
            const std::vector<size_t>& out_shape) {

            std::vector<size_t> in_idx(in_shape.size());

            // Start from the right (last dimension)
            size_t offset = out_shape.size() - in_shape.size();

            for (size_t i = 0; i < in_shape.size(); ++i) {
                // Map index, modulo for broadcast dimensions
                if (in_shape[i] == 1) {
                    in_idx[i] = 0;  // Broadcasting dimension
                }
                else {
                    in_idx[i] = out_idx[i + offset];
                }
            }

            return in_idx;
        }
    };

} // namespace tensor

#endif // TENSOR_RESHAPE_H