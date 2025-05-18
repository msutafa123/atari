// TensorStorage.h - v0.3.0
// Modern memory management for tensors

#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace tensor {

    template<typename T>
    class TensorStorage {
    public:
        // Default constructor
        TensorStorage() : size_(0), data_(nullptr) {}

        // Create storage with size (zeros-initialized)
        explicit TensorStorage(size_t size) : size_(size) {
            if (size > 0) {
                // Use shared_ptr with custom deleter for automatic memory management
                data_ = std::shared_ptr<T[]>(new T[size], std::default_delete<T[]>());

                // Initialize all memory to zero
                std::fill(data_.get(), data_.get() + size, T(0));
            }
        }

        // Create storage with size and initial value
        TensorStorage(size_t size, T initial_value) : size_(size) {
            if (size > 0) {
                data_ = std::shared_ptr<T[]>(new T[size], std::default_delete<T[]>());
                std::fill(data_.get(), data_.get() + size, initial_value);
            }
        }

        // Create storage from external data (takes ownership)
        TensorStorage(T* external_data, size_t size) : size_(size) {
            if (size > 0) {
                data_ = std::shared_ptr<T[]>(external_data, std::default_delete<T[]>());
            }
        }

        // Copy constructor (shallow copy with ref counting)
        TensorStorage(const TensorStorage& other) = default;

        // Move constructor
        TensorStorage(TensorStorage&& other) noexcept = default;

        // Copy assignment (shallow copy with ref counting)
        TensorStorage& operator=(const TensorStorage& other) = default;

        // Move assignment
        TensorStorage& operator=(TensorStorage&& other) noexcept = default;

        // Destructor (handled by shared_ptr)
        ~TensorStorage() = default;

        // Get data pointer
        T* data() {
            return data_ ? data_.get() : nullptr;
        }
        const T* data() const {
            return data_ ? data_.get() : nullptr;
        }

        // Get size
        size_t size() const {
            return size_;
        }

        // Check if storage is allocated
        bool is_allocated() const {
            return data_ != nullptr;
        }

        // Reset storage
        void reset() {
            data_.reset();
            size_ = 0;
        }

        // Fill with value
        void fill(T value) {
            if (data_) {
                std::fill(data_.get(), data_.get() + size_, value);
            }
        }

        // Get reference count
        long use_count() const {
            return data_ ? data_.use_count() : 0;
        }

        // Clone storage (deep copy)
        TensorStorage clone() const {
            TensorStorage result(size_);
            if (size_ > 0 && data_) {
                std::copy(data_.get(), data_.get() + size_, result.data());
            }
            return result;
        }

    private:
        size_t size_;
        std::shared_ptr<T[]> data_;  // Auto-managed memory with reference counting
    };

} // namespace tensor

#endif // TENSOR_STORAGE_H