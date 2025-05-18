// MemoryAllocator.h - v0.2.0
// Custom memory allocation for tensors

#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include "DeviceType.h"
#include <cstddef>
#include <string>
#include <mutex>
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>

namespace tensor {

    class MemoryAllocator {
    public:
        virtual ~MemoryAllocator() = default;

        // Allocate memory
        virtual void* allocate(size_t size_bytes) = 0;

        // Free memory
        virtual void free(void* ptr) = 0;

        // Get allocator name
        virtual std::string name() const = 0;

        // Get device associated with this allocator
        virtual Device device() const = 0;

        // Reset allocator state (free all caches, etc.)
        virtual void reset() = 0;

        // Get memory stats
        virtual size_t allocated_bytes() const = 0;
        virtual size_t reserved_bytes() const = 0;
    };

    // Standard allocator using malloc/free
    class StandardAllocator : public MemoryAllocator {
    public:
        void* allocate(size_t size_bytes) override {
            std::lock_guard<std::mutex> lock(mutex_);
            void* ptr = std::malloc(size_bytes);

            if (ptr) {
                allocated_memory_[ptr] = size_bytes;
                total_allocated_ += size_bytes;
            }

            return ptr;
        }

        void free(void* ptr) override {
            if (!ptr) return;

            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocated_memory_.find(ptr);
            if (it != allocated_memory_.end()) {
                total_allocated_ -= it->second;
                allocated_memory_.erase(it);
            }

            std::free(ptr);
        }

        std::string name() const override {
            return "StandardAllocator";
        }

        Device device() const override {
            return Device::cpu();
        }

        void reset() override {
            // Nothing to do for standard allocator
        }

        size_t allocated_bytes() const override {
            return total_allocated_;
        }

        size_t reserved_bytes() const override {
            return total_allocated_;  // No extra memory reservation
        }

        // Get singleton instance
        static StandardAllocator& instance() {
            static StandardAllocator instance;
            return instance;
        }

    private:
        std::mutex mutex_;
        std::unordered_map<void*, size_t> allocated_memory_;
        size_t total_allocated_ = 0;
    };

    // Aligned allocator for SIMD operations
    class AlignedAllocator : public MemoryAllocator {
    public:
        explicit AlignedAllocator(size_t alignment = 16) : alignment_(alignment) {}

        void* allocate(size_t size_bytes) override {
            std::lock_guard<std::mutex> lock(mutex_);
            void* ptr = nullptr;

#ifdef _MSC_VER
            ptr = _aligned_malloc(size_bytes, alignment_);
#else
            if (posix_memalign(&ptr, alignment_, size_bytes) != 0) {
                ptr = nullptr;
            }
#endif

            if (ptr) {
                allocated_memory_[ptr] = size_bytes;
                total_allocated_ += size_bytes;
            }

            return ptr;
        }

        void free(void* ptr) override {
            if (!ptr) return;

            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocated_memory_.find(ptr);
            if (it != allocated_memory_.end()) {
                total_allocated_ -= it->second;
                allocated_memory_.erase(it);
            }

#ifdef _MSC_VER
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }

        std::string name() const override {
            return "AlignedAllocator(" + std::to_string(alignment_) + ")";
        }

        Device device() const override {
            return Device::cpu();
        }

        void reset() override {
            // Nothing to do for aligned allocator
        }

        size_t allocated_bytes() const override {
            return total_allocated_;
        }

        size_t reserved_bytes() const override {
            return total_allocated_;
        }

        // Get alignment
        size_t alignment() const {
            return alignment_;
        }

    private:
        std::mutex mutex_;
        size_t alignment_;
        std::unordered_map<void*, size_t> allocated_memory_;
        size_t total_allocated_ = 0;
    };

    // Cache allocator to reduce allocation overhead
    class CachingAllocator : public MemoryAllocator {
    public:
        explicit CachingAllocator(size_t max_cached_bytes = 1024 * 1024 * 1024)  // 1GB default cache
            : max_cached_bytes_(max_cached_bytes), cached_bytes_(0), total_allocated_(0) {
        }

        ~CachingAllocator() {
            reset();
        }

        void* allocate(size_t size_bytes) override {
            std::lock_guard<std::mutex> lock(mutex_);

            // Try to find a cached block of appropriate size
            auto it = free_blocks_.lower_bound(size_bytes);
            if (it != free_blocks_.end()) {
                void* ptr = it->second;
                size_t block_size = it->first;

                free_blocks_.erase(it);
                size_map_[ptr] = block_size;
                cached_bytes_ -= block_size;

                return ptr;
            }

            // Allocate new block
            void* ptr = std::malloc(size_bytes);
            if (ptr) {
                size_map_[ptr] = size_bytes;
                total_allocated_ += size_bytes;
            }

            return ptr;
        }

        void free(void* ptr) override {
            if (!ptr) return;

            std::lock_guard<std::mutex> lock(mutex_);

            auto it = size_map_.find(ptr);
            if (it == size_map_.end()) {
                // This block wasn't allocated by us
                std::free(ptr);
                return;
            }

            size_t size = it->second;

            // Check if we should cache this block
            if (cached_bytes_ + size <= max_cached_bytes_) {
                free_blocks_.insert(std::make_pair(size, ptr));
                cached_bytes_ += size;
            }
            else {
                // Cache is full, just free the memory
                std::free(ptr);
                total_allocated_ -= size;
            }

            size_map_.erase(it);
        }

        std::string name() const override {
            return "CachingAllocator";
        }

        Device device() const override {
            return Device::cpu();
        }

        void reset() override {
            std::lock_guard<std::mutex> lock(mutex_);

            // Free all cached blocks
            for (auto& block : free_blocks_) {
                std::free(block.second);
                total_allocated_ -= block.first;
            }

            free_blocks_.clear();
            cached_bytes_ = 0;
        }

        size_t allocated_bytes() const override {
            return total_allocated_ - cached_bytes_;
        }

        size_t reserved_bytes() const override {
            return total_allocated_;
        }

        // Get max cache size
        size_t max_cached_bytes() const {
            return max_cached_bytes_;
        }

        // Set max cache size
        void set_max_cached_bytes(size_t max_bytes) {
            std::lock_guard<std::mutex> lock(mutex_);
            max_cached_bytes_ = max_bytes;

            // Free cached blocks if needed
            if (cached_bytes_ > max_cached_bytes_) {
                trim_cache();
            }
        }

        // Get current cached size
        size_t cached_bytes() const {
            return cached_bytes_;
        }

    private:
        std::mutex mutex_;
        size_t max_cached_bytes_;
        size_t cached_bytes_;
        size_t total_allocated_;

        // Map of size to pointers (free blocks)
        std::multimap<size_t, void*> free_blocks_;

        // Map of pointer to size
        std::unordered_map<void*, size_t> size_map_;

        // Trim cache to fit within max_cached_bytes
        void trim_cache() {
            while (cached_bytes_ > max_cached_bytes_ && !free_blocks_.empty()) {
                // Remove the largest block first
                auto it = --free_blocks_.end();
                void* ptr = it->second;
                size_t size = it->first;

                free_blocks_.erase(it);
                std::free(ptr);
                total_allocated_ -= size;
                cached_bytes_ -= size;
            }
        }
    };

} // namespace tensor

#endif // MEMORY_ALLOCATOR_H