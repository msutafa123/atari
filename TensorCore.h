// TensorCore.h - Main tensor implementation with improved features

#ifndef TENSOR_CORE_H
#define TENSOR_CORE_H

#include <vector>
#include <memory>
#include <algorithm>
#include <optional>
#include <type_traits>
#include "DataType.h"
#include "DeviceType.h"
#include "MemoryAllocator.h"

namespace tensor {

    template <typename T>
    class TensorCore {
    public:
        // Constructors with support for various initializations
        TensorCore();
        explicit TensorCore(const std::vector<size_t>& shape);
        TensorCore(const std::vector<size_t>& shape, T value);
        TensorCore(const std::vector<size_t>& shape, const T* data);

        // Enhanced memory management with custom allocators
        explicit TensorCore(const std::vector<size_t>& shape,
            std::shared_ptr<MemoryAllocator> allocator);

        // Device-specific instantiation
        TensorCore(const std::vector<size_t>& shape, const Device& device);

        // Copy and move semantics
        TensorCore(const TensorCore& other);
        TensorCore(TensorCore&& other) noexcept;
        TensorCore& operator=(const TensorCore& other);
        TensorCore& operator=(TensorCore&& other) noexcept;

        // Enhanced accessors
        T& at(const std::vector<size_t>& indices);
        const T& at(const std::vector<size_t>& indices) const;

        // Optimized element access for common dimensions
        T& at(size_t i);
        T& at(size_t i, size_t j);
        T& at(size_t i, size_t j, size_t k);

        // Data manipulation
        void fill(T value);
        TensorCore<T> reshape(const std::vector<size_t>& new_shape) const;
        TensorCore<T> clone() const;

        // Device operations
        TensorCore<T> to(const Device& device) const;
        bool is_on_device(const Device& device) const;

        // Memory and shape properties
        const std::vector<size_t>& shape() const;
        size_t size() const;
        size_t ndim() const;
        T* data();
        const T* data() const;

        // Static factory methods
        static TensorCore<T> zeros(const std::vector<size_t>& shape);
        static TensorCore<T> ones(const std::vector<size_t>& shape);
        static TensorCore<T> random(const std::vector<size_t>& shape, T min = T(0), T max = T(1));

    private:
        std::vector<size_t> shape_;
        std::shared_ptr<T[]> data_;
        size_t size_;
        Device device_;
        std::shared_ptr<MemoryAllocator> allocator_;

        void allocate_memory();
        size_t calculate_offset(const std::vector<size_t>& indices) const;
    };

} // namespace tensor

#endif // TENSOR_CORE_H