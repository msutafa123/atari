// TensorStrides.h - v0.3.0
// Optimized stride computation for tensor indexing

#ifndef TENSOR_STRIDES_H
#define TENSOR_STRIDES_H

#include "TensorShape.h"
#include <vector>
#include <numeric>
#include <cassert>
#include <functional>
#include <sstream>
#include <iostream>

namespace tensor {

    // Memory layout for tensor
    enum class MemoryLayout {
        ROW_MAJOR,    // C-style, last dimension is contiguous
        COLUMN_MAJOR  // Fortran-style, first dimension is contiguous
    };

    class TensorStrides {
    public:
        // Default constructor
        TensorStrides() = default;

        // Create strides from shape with specified layout
        explicit TensorStrides(const TensorShape& shape, MemoryLayout layout = MemoryLayout::ROW_MAJOR) :
            shape_dims_(shape.dims()),
            layout_(layout)
        {
            compute_strides();
        }

        // Get strides vector
        const std::vector<size_t>& strides() const {
            return strides_;
        }

        // Get stride at index
        size_t stride(size_t index) const {
            return index < strides_.size() ? strides_[index] : 0;
        }

        // Get memory layout
        MemoryLayout layout() const {
            return layout_;
        }

        // Check if strides are contiguous
        bool is_contiguous() const {
            if (strides_.empty()) return true;

            size_t expected_stride = 1;

            if (layout_ == MemoryLayout::ROW_MAJOR) {
                for (size_t i = strides_.size(); i > 0; --i) {
                    size_t idx = i - 1;
                    if (strides_[idx] != expected_stride) {
                        return false;
                    }
                    expected_stride *= shape_dims_[idx];
                }
            }
            else { // COLUMN_MAJOR
                for (size_t i = 0; i < strides_.size(); ++i) {
                    if (strides_[i] != expected_stride) {
                        return false;
                    }
                    expected_stride *= shape_dims_[i];
                }
            }

            return true;
        }

        // Compute linear index from multi-dimensional indices - Optimized for performance
        size_t get_offset(const std::vector<size_t>& indices) const {
            assert(indices.size() == strides_.size() && "Indices dimension mismatch");

            // Optimized loop unrolling for common dimensions (1-4)
            size_t offset = 0;
            size_t n = indices.size();

            if (n == 0) return 0;

            // Fast path for common dimensions using manual loop unrolling
            if (n == 1) {
                offset = indices[0] * strides_[0];
            }
            else if (n == 2) {
                offset = indices[0] * strides_[0] + indices[1] * strides_[1];
            }
            else if (n == 3) {
                offset = indices[0] * strides_[0] + indices[1] * strides_[1] + indices[2] * strides_[2];
            }
            else if (n == 4) {
                offset = indices[0] * strides_[0] + indices[1] * strides_[1] +
                    indices[2] * strides_[2] + indices[3] * strides_[3];
            }
            else {
                // General case for higher dimensions
                for (size_t i = 0; i < n; ++i) {
                    offset += indices[i] * strides_[i];
                }
            }

            return offset;
        }

        // Convert flat index to multidimensional indices
        std::vector<size_t> flat_to_indices(size_t flat_index) const {
            std::vector<size_t> indices(shape_dims_.size());

            for (size_t i = 0; i < shape_dims_.size(); ++i) {
                indices[i] = (flat_index / strides_[i]) % shape_dims_[i];
            }

            return indices;
        }

        // Pre-compute common index patterns for optimization
        void cache_common_offsets() {
            if (shape_dims_.size() != 2) return; // Only for 2D tensors for now

            size_t rows = shape_dims_[0];
            size_t cols = shape_dims_[1];

            // Pre-compute row offsets for 2D tensors
            row_offsets_.resize(rows);
            for (size_t i = 0; i < rows; ++i) {
                row_offsets_[i] = i * strides_[0];
            }

            has_cached_offsets_ = true;
        }

        // Get cached row offset (for 2D tensors)
        size_t get_row_offset(size_t row) const {
            if (has_cached_offsets_ && row < row_offsets_.size()) {
                return row_offsets_[row];
            }
            return row * strides_[0];
        }

        // Fast offset for 2D indexing
        size_t offset_2d(size_t i, size_t j) const {
            if (has_cached_offsets_) {
                return row_offsets_[i] + j * strides_[1];
            }
            return i * strides_[0] + j * strides_[1];
        }

        // Fast offset for 3D indexing
        size_t offset_3d(size_t i, size_t j, size_t k) const {
            return i * strides_[0] + j * strides_[1] + k * strides_[2];
        }

        // Fast offset for 4D indexing
        size_t offset_4d(size_t i, size_t j, size_t k, size_t l) const {
            return i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3];
        }

        // String representation
        std::string to_string() const {
            std::stringstream ss;
            ss << "TensorStrides(";

            for (size_t i = 0; i < strides_.size(); ++i) {
                ss << strides_[i];
                if (i < strides_.size() - 1) ss << ", ";
            }

            ss << ")";
            return ss.str();
        }

        // Debug output
        void debug_print() const {
            std::cout << "Shape: [";
            for (size_t i = 0; i < shape_dims_.size(); ++i) {
                std::cout << shape_dims_[i];
                if (i < shape_dims_.size() - 1) std::cout << ", ";
            }
            std::cout << "], Strides: [";

            for (size_t i = 0; i < strides_.size(); ++i) {
                std::cout << strides_[i];
                if (i < strides_.size() - 1) std::cout << ", ";
            }
            std::cout << "], Layout: "
                << (layout_ == MemoryLayout::ROW_MAJOR ? "ROW_MAJOR" : "COLUMN_MAJOR")
                << std::endl;
        }

    private:
        std::vector<size_t> shape_dims_;
        std::vector<size_t> strides_;
        MemoryLayout layout_ = MemoryLayout::ROW_MAJOR;

        // Cached offsets for faster access patterns
        std::vector<size_t> row_offsets_;
        bool has_cached_offsets_ = false;

        // Compute strides from shape and layout
        void compute_strides() {
            if (shape_dims_.empty()) {
                strides_.clear();
                return;
            }

            strides_.resize(shape_dims_.size());

            if (layout_ == MemoryLayout::ROW_MAJOR) {
                // C-style, last dimension is contiguous
                size_t stride = 1;
                for (size_t i = shape_dims_.size(); i > 0; --i) {
                    size_t idx = i - 1;
                    strides_[idx] = stride;
                    stride *= shape_dims_[idx];
                }
            }
            else {
                // Fortran-style, first dimension is contiguous
                size_t stride = 1;
                for (size_t i = 0; i < shape_dims_.size(); ++i) {
                    strides_[i] = stride;
                    stride *= shape_dims_[i];
                }
            }
        }
    };

} // namespace tensor

#endif // TENSOR_STRIDES_H