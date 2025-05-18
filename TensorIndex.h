// TensorIndex.h - v0.1.0
// Advanced indexing for tensors

#ifndef TENSOR_INDEX_H
#define TENSOR_INDEX_H

#include "Tensor.h"
#include "TensorView.h"

namespace tensor {

    template<typename T>
    class TensorIndex {
    public:
        // Constructor
        explicit TensorIndex(Tensor<T>& tensor) : tensor_(tensor) {}

        // Slice tensor along dimension
        TensorView<T> slice(size_t dim, size_t start, size_t end) {
            if (dim >= tensor_.shape().ndim()) {
                throw std::out_of_range("Dimension out of range");
            }

            const auto& shape = tensor_.shape();
            const auto& strides = tensor_.strides();

            if (start >= shape.dim(dim) || end > shape.dim(dim) || start >= end) {
                throw std::invalid_argument("Invalid slice range");
            }

            // Create new shape with reduced dimension
            std::vector<size_t> new_dims = shape.dims();
            new_dims[dim] = end - start;
            TensorShape new_shape(new_dims);

            // Calculate offset
            size_t offset = start * strides.stride(dim);

            return TensorView<T>(tensor_, new_shape, strides, offset);
        }

        // More advanced indexing methods will be added...

    private:
        Tensor<T>& tensor_;
    };

} // namespace tensor

#endif // TENSOR_INDEX_H