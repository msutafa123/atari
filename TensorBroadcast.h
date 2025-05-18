// TensorBroadcast.h - v2.0.0
// Support for broadcasting tensors of different shapes

#ifndef TENSOR_BROADCAST_H
#define TENSOR_BROADCAST_H

#include "Tensor.h"
#include "TensorView.h"
#include <algorithm>
#include <stdexcept>

namespace tensor {

    /**
     * @brief Class for handling tensor broadcasting operations
     */
    class TensorBroadcaster {
    public:
        /**
         * @brief Compute broadcasted shape from two input shapes
         */
        static std::vector<size_t> get_broadcast_shape(
            const std::vector<size_t>& a,
            const std::vector<size_t>& b) {

            // Get dimensions
            size_t a_ndim = a.size();
            size_t b_ndim = b.size();

            // Result will have max number of dimensions
            size_t result_ndim = std::max(a_ndim, b_ndim);
            std::vector<size_t> result_dims(result_ndim);

            // Compute dimensions from right to left
            for (size_t i = 0; i < result_ndim; ++i) {
                size_t a_idx = i < a_ndim ? a_ndim - 1 - i : SIZE_MAX;
                size_t b_idx = i < b_ndim ? b_ndim - 1 - i : SIZE_MAX;

                size_t a_dim = a_idx != SIZE_MAX ? a[a_idx] : 1;
                size_t b_dim = b_idx != SIZE_MAX ? b[b_idx] : 1;

                if (a_dim == b_dim || a_dim == 1 || b_dim == 1) {
                    result_dims[result_ndim - 1 - i] = std::max(a_dim, b_dim);
                }
                else {
                    throw std::invalid_argument("Shapes cannot be broadcast together");
                }
            }

            return result_dims;
        }

        /**
         * @brief Compute broadcasted shape from multiple input shapes
         */
        static std::vector<size_t> get_broadcast_shape(
            const std::vector<std::vector<size_t>>& shapes) {

            if (shapes.empty()) {
                throw std::invalid_argument("Cannot broadcast empty shape list");
            }

            std::vector<size_t> result = shapes[0];
            for (size_t i = 1; i < shapes.size(); ++i) {
                result = get_broadcast_shape(result, shapes[i]);
            }

            return result;
        }

        /**
         * @brief Check if two shapes are broadcast compatible
         */
        static bool is_broadcastable(
            const std::vector<size_t>& a,
            const std::vector<size_t>& b) {

            try {
                get_broadcast_shape(a, b);
                return true;
            }
            catch (const std::invalid_argument&) {
                return false;
            }
        }

        /**
         * @brief Check if multiple shapes are broadcast compatible
         */
        static bool is_broadcastable(
            const std::vector<std::vector<size_t>>& shapes) {

            try {
                get_broadcast_shape(shapes);
                return true;
            }
            catch (const std::invalid_argument&) {
                return false;
            }
        }

        /**
         * @brief Map an index from the broadcasted shape to an original shape
         */
        static std::vector<size_t> map_indices(
            const std::vector<size_t>& broadcasted_indices,
            const std::vector<size_t>& original_shape) {

            // Check if indices are valid for broadcast shape
            if (broadcasted_indices.size() < original_shape.size()) {
                throw std::invalid_argument("Broadcast indices must have at least as many dimensions as original shape");
            }

            std::vector<size_t> original_indices(original_shape.size());

            // Map from right to left
            size_t offset = broadcasted_indices.size() - original_shape.size();

            for (size_t i = 0; i < original_shape.size(); ++i) {
                // If original dimension is 1, index is always 0 (broadcasted)
                if (original_shape[i] == 1) {
                    original_indices[i] = 0;
                }
                else {
                    // Otherwise, use the corresponding broadcast index
                    original_indices[i] = broadcasted_indices[i + offset];
                }
            }

            return original_indices;
        }

        /**
         * @brief Broadcast tensor to target shape
         */
        template<typename T>
        static Tensor<T> broadcast_to(const Tensor<T>& tensor, const std::vector<size_t>& target_shape) {
            // Validate shapes
            if (!is_broadcastable(tensor.shape(), target_shape)) {
                throw std::invalid_argument("Cannot broadcast tensor to target shape");
            }

            // Create result tensor
            Tensor<T> result(target_shape);

            // Copy data with broadcasting
            std::vector<size_t> out_idx(target_shape.size(), 0);
            bool done = false;

            while (!done) {
                // Map output indices to input indices
                std::vector<size_t> in_idx = map_indices(out_idx, tensor.shape());

                // Copy value
                result.at(out_idx) = tensor.at(in_idx);

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

            return result;
        }
    };

} // namespace tensor

#endif // TENSOR_BROADCAST_H