// TensorIterator.h - v0.2.0
// Iterator for traversing tensors

#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H

#include "Tensor.h"
#include <vector>
#include <iterator>
#include <functional>

namespace tensor {

    template<typename T>
    class TensorIterator {
    public:
        // Iterator traits
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        // Constructor
        explicit TensorIterator(Tensor<T>& tensor) :
            tensor_(&tensor),
            current_index_(std::vector<size_t>(tensor.shape().ndim(), 0)),
            end_(tensor.size() == 0) {
        }

        // Create end iterator
        static TensorIterator end(Tensor<T>& tensor) {
            TensorIterator it(tensor);
            it.end_ = true;
            return it;
        }

        // Dereference operator
        reference operator*() {
            return tensor_->at(current_index_);
        }

        // Arrow operator
        pointer operator->() {
            return &(tensor_->at(current_index_));
        }

        // Pre-increment
        TensorIterator& operator++() {
            next();
            return *this;
        }

        // Post-increment
        TensorIterator operator++(int) {
            TensorIterator tmp = *this;
            next();
            return tmp;
        }

        // Equality comparison
        bool operator==(const TensorIterator& other) const {
            // If either is at end, just compare end status
            if (end_ || other.end_) {
                return end_ == other.end_;
            }

            // Otherwise, compare tensors and current indices
            return tensor_ == other.tensor_ && current_index_ == other.current_index_;
        }

        // Inequality comparison
        bool operator!=(const TensorIterator& other) const {
            return !(*this == other);
        }

        // Check if iterator has reached the end
        bool end() const { return end_; }

        // Move to next element
        void next() {
            if (end_) return;

            const auto& shape = tensor_->shape();

            // Increment indices like a multi-dimensional counter
            // DÜZELTME: size_t için güvenli döngü
            for (size_t i = current_index_.size(); i > 0; --i) {
                size_t dim = i - 1;
                current_index_[dim]++;
                if (current_index_[dim] < shape.dim(dim)) {
                    return; // Successfully incremented
                }

                // This dimension overflowed, reset it and try to increment next dimension
                current_index_[dim] = 0;
            }

            // If we get here, we've exhausted all indices
            end_ = true;
        }

        // Get current element
        T& value() {
            return tensor_->at(current_index_);
        }

        const T& value() const {
            return tensor_->at(current_index_);
        }

        // Get current indices
        const std::vector<size_t>& indices() const {
            return current_index_;
        }

        // Get current flat index
        size_t flat_index() const {
            size_t idx = 0;
            const auto& strides = tensor_->strides();

            for (size_t i = 0; i < current_index_.size(); ++i) {
                idx += current_index_[i] * strides.stride(i);
            }

            return idx;
        }

        // Create a range for use with for-range loop
        class TensorRange {
        public:
            TensorRange(Tensor<T>& tensor) : tensor_(tensor) {}

            TensorIterator<T> begin() {
                return TensorIterator<T>(tensor_);
            }

            TensorIterator<T> end() {
                return TensorIterator<T>::end(tensor_);
            }

        private:
            Tensor<T>& tensor_;
        };

    private:
        Tensor<T>* tensor_;
        std::vector<size_t> current_index_;
        bool end_;
    };

    // Helper function to get iterator range for tensor
    template<typename T>
    typename TensorIterator<T>::TensorRange iter(Tensor<T>& tensor) {
        return typename TensorIterator<T>::TensorRange(tensor);
    }

    // For-each function to apply a function to each element
    template<typename T, typename Func>
    void for_each(Tensor<T>& tensor, Func f) {
        for (auto it = TensorIterator<T>(tensor); !it.end(); ++it) {
            f(it.indices(), *it);
        }
    }

    // Map function to create a new tensor by applying a function to each element
    template<typename T, typename R, typename Func>
    Tensor<R> map(const Tensor<T>& tensor, Func f) {
        Tensor<R> result(tensor.shape());

        for (auto it = TensorIterator<T>(const_cast<Tensor<T>&>(tensor)); !it.end(); ++it) {
            result.at(it.indices()) = f(it.value());
        }

        return result;
    }

    // Zip iteration over multiple tensors (single pair version)
    template<typename T1, typename T2, typename Func>
    void zip_for_each(Tensor<T1>& tensor1, Tensor<T2>& tensor2, Func f) {
        if (tensor1.shape() != tensor2.shape()) {
            throw std::invalid_argument("All tensors must have the same shape for zip_for_each");
        }

        auto it1 = TensorIterator<T1>(tensor1);
        auto it2 = TensorIterator<T2>(tensor2);

        while (!it1.end()) {
            f(it1.indices(), it1.value(), it2.value());
            ++it1;
            ++it2;
        }
    }

} // namespace tensor

#endif // TENSOR_ITERATOR_H