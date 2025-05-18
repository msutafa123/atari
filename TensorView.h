// TensorView.h - v2.0.0
// Provides view into existing tensor without copying data

#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace tensor {

    /**
     * @brief Class providing a view into an existing tensor
     *
     * This class allows creating a view into a subset of a tensor without copying data.
     */
    template<typename T>
    class TensorView {
    public:
        /**
         * @brief Create view from tensor
         */
        explicit TensorView(Tensor<T>& tensor) :
            tensor_(&tensor),
            offset_(0),
            shape_(tensor.shape()) {
        }

        /**
         * @brief Create view from tensor with specific shape and offset
         */
        TensorView(Tensor<T>& tensor, const std::vector<size_t>& shape, size_t offset = 0) :
            tensor_(&tensor),
            offset_(offset),
            shape_(shape) {

            // Validate view parameters
            size_t view_size = std::accumulate(shape.begin(), shape.end(),
                size_t(1), std::multiplies<size_t>());
            if (offset + view_size > tensor.size()) {
                throw std::out_of_range("View extends beyond tensor boundaries");
            }
        }

        /**
         * @brief Access element at indices
         */
        T& at(const std::vector<size_t>& indices) {
            validate_indices(indices);
            size_t idx = offset_ + calculate_offset(indices);
            return tensor_->data()[idx];
        }

        /**
         * @brief Access element at indices (const version)
         */
        const T& at(const std::vector<size_t>& indices) const {
            validate_indices(indices);
            size_t idx = offset_ + calculate_offset(indices);
            return tensor_->data()[idx];
        }

        /**
         * @brief Convenient access for 1D tensors
         */
        T& at(size_t i) {
            return at({ i });
        }

        /**
         * @brief Convenient access for 1D tensors (const version)
         */
        const T& at(size_t i) const {
            return at({ i });
        }

        /**
         * @brief Convenient access for 2D tensors
         */
        T& at(size_t i, size_t j) {
            return at({ i, j });
        }

        /**
         * @brief Convenient access for 2D tensors (const version)
         */
        const T& at(size_t i, size_t j) const {
            return at({ i, j });
        }

        /**
         * @brief Convenient access for 3D tensors
         */
        T& at(size_t i, size_t j, size_t k) {
            return at({ i, j, k });
        }

        /**
         * @brief Convenient access for 3D tensors (const version)
         */
        const T& at(size_t i, size_t j, size_t k) const {
            return at({ i, j, k });
        }

        /**
         * @brief Get shape of view
         */
        const std::vector<size_t>& shape() const {
            return shape_;
        }

        /**
         * @brief Get number of dimensions
         */
        size_t ndim() const {
            return shape_.size();
        }

        /**
         * @brief Get total number of elements in view
         */
        size_t size() const {
            return std::accumulate(shape_.begin(), shape_.end(),
                size_t(1), std::multiplies<size_t>());
        }

        /**
         * @brief Get the parent tensor
         */
        Tensor<T>& tensor() {
            return *tensor_;
        }

        /**
         * @brief Get the parent tensor (const version)
         */
        const Tensor<T>& tensor() const {
            return *tensor_;
        }

        /**
         * @brief Fill view with value
         */
        void fill(T value) {
            // Handle each element using indices
            std::vector<size_t> idx(shape_.size(), 0);
            bool done = false;

            while (!done) {
                at(idx) = value;

                // Increment indices
                for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < shape_[i]) {
                        break;
                    }
                    idx[i] = 0;
                    if (i == 0) {
                        done = true;
                    }
                }
            }
        }

        /**
         * @brief Create a new tensor from this view (copies data)
         */
        Tensor<T> to_tensor() const {
            Tensor<T> result(shape_);

            // Copy data from view to new tensor
            std::vector<size_t> idx(shape_.size(), 0);
            bool done = false;

            while (!done) {
                result.at(idx) = at(idx);

                // Increment indices
                for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < shape_[i]) {
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
         * @brief Create a subview (a view of this view)
         */
        TensorView slice(size_t dim, size_t start, size_t end) const {
            if (dim >= shape_.size()) {
                throw std::out_of_range("Dimension out of range");
            }

            if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
                throw std::invalid_argument("Invalid slice range");
            }

            // Calculate new shape
            std::vector<size_t> new_shape = shape_;
            new_shape[dim] = end - start;

            // Calculate new offset
            size_t stride = 1;
            for (size_t i = dim + 1; i < shape_.size(); ++i) {
                stride *= shape_[i];
            }
            size_t new_offset = offset_ + start * stride;

            return TensorView(*tensor_, new_shape, new_offset);
        }

    private:
        Tensor<T>* tensor_;         // Pointer to parent tensor
        size_t offset_;             // Offset into parent tensor data
        std::vector<size_t> shape_; // Shape of view

        /**
         * @brief Validate indices are within bounds
         */
        void validate_indices(const std::vector<size_t>& indices) const {
            if (indices.size() != shape_.size()) {
                throw std::invalid_argument("Number of indices must match number of dimensions");
            }

            for (size_t i = 0; i < indices.size(); i++) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range("Index out of bounds");
                }
            }
        }

        /**
         * @brief Calculate offset from indices
         */
        size_t calculate_offset(const std::vector<size_t>& indices) const {
            size_t offset = 0;
            size_t stride = 1;

            // Calculate row-major (C-style) offset
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                offset += indices[i] * stride;
                stride *= shape_[i];
            }

            return offset;
        }
    };

} // namespace tensor

#endif // TENSOR_VIEW_H