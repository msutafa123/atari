// TensorShape.h - v0.2.0
// Represents the shape of a tensor

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <vector>
#include <numeric>
#include <cstddef>
#include <string>
#include <sstream>
#include <algorithm>

namespace tensor {

    class TensorShape {
    public:
        // Create empty shape
        TensorShape() = default;

        // Create shape from dimensions
        explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims) {}

        // Create shape from initializer list
        TensorShape(std::initializer_list<size_t> dims) : dims_(dims) {}

        // Get dimensions
        const std::vector<size_t>& dims() const { return dims_; }

        // Get dimension at index
        size_t dim(size_t index) const { return index < dims_.size() ? dims_[index] : 1; }

        // Get number of dimensions
        size_t ndim() const { return dims_.size(); }

        // Get total number of elements
        size_t size() const {
            if (dims_.empty()) return 0;
            return std::accumulate(dims_.begin(), dims_.end(),
                size_t(1), std::multiplies<size_t>());
        }

        // Add a dimension of size 1 at position
        TensorShape unsqueeze(size_t dim) const {
            if (dim > dims_.size()) {
                throw std::out_of_range("Dimension out of range for unsqueeze");
            }

            std::vector<size_t> new_dims = dims_;
            new_dims.insert(new_dims.begin() + dim, 1);
            return TensorShape(new_dims);
        }

        // Remove dimensions of size 1
        TensorShape squeeze() const {
            std::vector<size_t> new_dims;
            for (size_t d : dims_) {
                if (d != 1) {
                    new_dims.push_back(d);
                }
            }
            return TensorShape(new_dims);
        }

        // Remove dimension of size 1 at position
        TensorShape squeeze(size_t dim) const {
            if (dim >= dims_.size()) {
                throw std::out_of_range("Dimension out of range for squeeze");
            }

            if (dims_[dim] != 1) {
                throw std::invalid_argument("Can only squeeze dimensions of size 1");
            }

            std::vector<size_t> new_dims = dims_;
            new_dims.erase(new_dims.begin() + dim);
            return TensorShape(new_dims);
        }

        // Check if two shapes are compatible for broadcasting
        bool is_broadcastable_with(const TensorShape& other) const {
            // Start from the right (last dimension)
            size_t i = dims_.size();
            size_t j = other.dims_.size();

            while (i > 0 && j > 0) {
                --i;
                --j;

                // Dimensions must be equal or one of them must be 1
                if (dims_[i] != other.dims_[j] && dims_[i] != 1 && other.dims_[j] != 1) {
                    return false;
                }
            }

            return true;
        }

        // Calculate broadcast shape with another shape
        TensorShape broadcast_with(const TensorShape& other) const {
            if (!is_broadcastable_with(other)) {
                throw std::invalid_argument("Shapes are not broadcastable");
            }

            // Result will have max number of dimensions
            size_t result_ndim = std::max(dims_.size(), other.dims_.size());
            std::vector<size_t> result_dims(result_ndim);

            // Fill from right to left
            for (size_t i = 0; i < result_ndim; ++i) {
                size_t this_dim = (i < dims_.size()) ? dims_[dims_.size() - 1 - i] : 1;
                size_t other_dim = (i < other.dims_.size()) ? other.dims_[other.dims_.size() - 1 - i] : 1;
                result_dims[result_ndim - 1 - i] = std::max(this_dim, other_dim);
            }

            return TensorShape(result_dims);
        }

        // Create a new shape with a different dimension
        TensorShape with_dim(size_t index, size_t dim_size) const {
            if (index >= dims_.size()) {
                throw std::out_of_range("Index out of range for with_dim");
            }

            std::vector<size_t> new_dims = dims_;
            new_dims[index] = dim_size;
            return TensorShape(new_dims);
        }

        // String representation
        std::string to_string() const {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < dims_.size(); ++i) {
                ss << dims_[i];
                if (i < dims_.size() - 1) ss << ", ";
            }
            ss << "]";
            return ss.str();
        }

        // Equality operator
        bool operator==(const TensorShape& other) const {
            return dims_ == other.dims_;
        }

        // Inequality operator
        bool operator!=(const TensorShape& other) const {
            return !(*this == other);
        }

    private:
        std::vector<size_t> dims_;
    };

} // namespace tensor

#endif // TENSOR_SHAPE_H