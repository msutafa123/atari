// math_utils.h - v1.0.0
// Mathematical utilities for tensor operations
// C++17 standards compliant

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "Tensor.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>
#include <functional>
#include <stdexcept>
#include <queue>

namespace tensor {
    namespace math_utils {

        // ===== Constants =====
        template<typename T>
        struct Constants {
            static constexpr T pi = T(3.14159265358979323846);
            static constexpr T e = T(2.71828182845904523536);
            static constexpr T epsilon = std::numeric_limits<T>::epsilon();
            static constexpr T infinity = std::numeric_limits<T>::infinity();
        };

        // ===== Thread Pool for Parallel Computing =====
        class ThreadPool {
        public:
            explicit ThreadPool(size_t num_threads = 0) {
                if (num_threads == 0) {
                    num_threads = std::thread::hardware_concurrency();
                    if (num_threads == 0) num_threads = 2; // Fallback
                }
                threads.reserve(num_threads);

                for (size_t i = 0; i < num_threads; ++i) {
                    threads.emplace_back([this] {
                        while (true) {
                            std::function<void()> task;
                            {
                                std::unique_lock<std::mutex> lock(queue_mutex);
                                condition.wait(lock, [this] { return stop || !tasks.empty(); });
                                if (stop && tasks.empty()) return;
                                task = std::move(tasks.front());
                                tasks.pop();
                            }
                            task();
                        }
                        });
                }
            }

            template<typename F, typename... Args>
            auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
                using return_type = typename std::invoke_result<F, Args...>::type;

                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                );

                std::future<return_type> result = task->get_future();
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    if (stop) throw std::runtime_error("ThreadPool is stopped");
                    tasks.emplace([task]() { (*task)(); });
                }
                condition.notify_one();
                return result;
            }

            size_t size() const {
                return threads.size();
            }

            ~ThreadPool() {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    stop = true;
                }
                condition.notify_all();
                for (std::thread& worker : threads) {
                    worker.join();
                }
            }

        private:
            std::vector<std::thread> threads;
            std::queue<std::function<void()>> tasks;
            std::mutex queue_mutex;
            std::condition_variable condition;
            bool stop = false;
        };

        // Global thread pool instance
        inline ThreadPool& get_thread_pool() {
            static ThreadPool pool;
            return pool;
        }

        // ===== Basic Matrix Operations =====

        /**
         * @brief Matrix multiplication between two tensors
         * @param a First tensor [batch, M, K]
         * @param b Second tensor [batch, K, N]
         * @return Result tensor [batch, M, N]
         */
        template<typename T>
        Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
            // Check dimensions
            if (a.ndim() < 2 || b.ndim() < 2) {
                throw std::invalid_argument("Both tensors must have at least 2 dimensions for matmul");
            }

            // Check inner dimensions match
            size_t a_dims = a.ndim();
            size_t b_dims = b.ndim();

            size_t a_inner = a.shape()[a_dims - 1];
            size_t b_inner = b.shape()[b_dims - 2];

            if (a_inner != b_inner) {
                throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
            }

            // Get outer dimensions
            size_t m = a.shape()[a_dims - 2]; // Rows of A
            size_t n = b.shape()[b_dims - 1]; // Cols of B
            size_t k = a_inner; // Inner dimension

            // Calculate batch dimensions
            std::vector<size_t> result_shape;

            // If both tensors have the same number of dimensions
            if (a_dims == b_dims) {
                // Check batch dimensions
                for (size_t i = 0; i < a_dims - 2; ++i) {
                    if (a.shape()[i] != b.shape()[i]) {
                        throw std::invalid_argument("Batch dimensions must match");
                    }
                    result_shape.push_back(a.shape()[i]);
                }
            }
            else if (a_dims > b_dims) {
                // A has more dimensions, use those for batch
                for (size_t i = 0; i < a_dims - 2; ++i) {
                    result_shape.push_back(a.shape()[i]);
                }
            }
            else {
                // B has more dimensions, use those for batch
                for (size_t i = 0; i < b_dims - 2; ++i) {
                    result_shape.push_back(b.shape()[i]);
                }
            }

            // Add matrix dimensions
            result_shape.push_back(m);
            result_shape.push_back(n);

            Tensor<T> result(result_shape);
            result.fill(T(0));

            // Handle batch dimensions by flattening them
            size_t batch_size = 1;
            for (size_t i = 0; i < result_shape.size() - 2; ++i) {
                batch_size *= result_shape[i];
            }

            // Use thread pool for parallel computation
            auto& pool = get_thread_pool();
            std::vector<std::future<void>> futures;

            // Process each batch in parallel
            for (size_t batch = 0; batch < batch_size; ++batch) {
                futures.push_back(pool.enqueue([&, batch]() {
                    // Calculate indices for current batch
                    std::vector<size_t> batch_indices;
                    size_t temp = batch;
                    for (size_t i = 0; i < result_shape.size() - 2; ++i) {
                        batch_indices.push_back(temp % result_shape[i]);
                        temp /= result_shape[i];
                    }

                    // Perform matrix multiplication for this batch
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            T sum = T(0);

                            for (size_t x = 0; x < k; ++x) {
                                // Build indices for a and b
                                std::vector<size_t> a_idx = batch_indices;
                                a_idx.push_back(i);
                                a_idx.push_back(x);

                                std::vector<size_t> b_idx = batch_indices;
                                b_idx.push_back(x);
                                b_idx.push_back(j);

                                sum += a.at(a_idx) * b.at(b_idx);
                            }

                            // Build index for result
                            std::vector<size_t> result_idx = batch_indices;
                            result_idx.push_back(i);
                            result_idx.push_back(j);

                            result.at(result_idx) = sum;
                        }
                    }
                    }));
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Transpose a tensor by swapping last two dimensions
         * @param tensor Input tensor
         * @return Transposed tensor
         */
        template<typename T>
        Tensor<T> transpose(const Tensor<T>& tensor) {
            if (tensor.ndim() < 2) {
                throw std::invalid_argument("Tensor must have at least 2 dimensions for transpose");
            }

            // Get shape and swap last two dimensions
            std::vector<size_t> new_shape = tensor.shape();
            std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);

            Tensor<T> result(new_shape);

            // Get the total size of the tensor
            size_t total_size = tensor.size();
            size_t last_dim = tensor.shape()[tensor.ndim() - 1];
            size_t second_last_dim = tensor.shape()[tensor.ndim() - 2];

            // Calculate the size of the batch dimensions
            size_t batch_size = total_size / (last_dim * second_last_dim);

            // Use thread pool for parallel computation
            auto& pool = get_thread_pool();
            std::vector<std::future<void>> futures;

            // Process each batch element in parallel
            for (size_t batch = 0; batch < batch_size; ++batch) {
                futures.push_back(pool.enqueue([&, batch]() {
                    // Calculate batch indices
                    std::vector<size_t> batch_indices;
                    size_t temp = batch;
                    for (size_t i = 0; i < tensor.ndim() - 2; ++i) {
                        batch_indices.push_back(temp % tensor.shape()[i]);
                        temp /= tensor.shape()[i];
                    }

                    // Transpose the matrix for this batch
                    for (size_t i = 0; i < second_last_dim; ++i) {
                        for (size_t j = 0; j < last_dim; ++j) {
                            // Build indices for input tensor
                            std::vector<size_t> input_idx = batch_indices;
                            input_idx.push_back(i);
                            input_idx.push_back(j);

                            // Build indices for output tensor (swapped i and j)
                            std::vector<size_t> output_idx = batch_indices;
                            output_idx.push_back(j);
                            output_idx.push_back(i);

                            result.at(output_idx) = tensor.at(input_idx);
                        }
                    }
                    }));
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Transpose a tensor with custom dimension order
         * @param tensor Input tensor
         * @param perm Permutation indices
         * @return Transposed tensor
         */
        template<typename T>
        Tensor<T> transpose(const Tensor<T>& tensor, const std::vector<size_t>& perm) {
            if (tensor.ndim() != perm.size()) {
                throw std::invalid_argument("Permutation size must match tensor dimensions");
            }

            // Check permutation is valid
            std::vector<bool> used(perm.size(), false);
            for (size_t i : perm) {
                if (i >= perm.size() || used[i]) {
                    throw std::invalid_argument("Invalid permutation");
                }
                used[i] = true;
            }

            // Get new shape based on permutation
            std::vector<size_t> new_shape(perm.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                new_shape[i] = tensor.shape()[perm[i]];
            }

            Tensor<T> result(new_shape);

            // Create mapping function for indices
            auto map_indices = [&perm](const std::vector<size_t>& indices) {
                std::vector<size_t> new_indices(indices.size());
                for (size_t i = 0; i < indices.size(); ++i) {
                    new_indices[perm[i]] = indices[i];
                }
                return new_indices;
                };

            // Iterate through all elements (can be optimized further)
            std::vector<size_t> idx(tensor.ndim(), 0);
            bool done = false;

            while (!done) {
                // Map indices based on permutation
                std::vector<size_t> new_idx = map_indices(idx);

                // Copy value
                result.at(new_idx) = tensor.at(idx);

                // Increment indices
                for (int i = idx.size() - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < tensor.shape()[i]) {
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
         * @brief Compute dot product between two 1D tensors
         * @param a First tensor
         * @param b Second tensor
         * @return Scalar dot product
         */
        template<typename T>
        T dot(const Tensor<T>& a, const Tensor<T>& b) {
            if (a.ndim() != 1 || b.ndim() != 1) {
                throw std::invalid_argument("Both tensors must be 1D for dot product");
            }

            if (a.shape()[0] != b.shape()[0]) {
                throw std::invalid_argument("Tensors must have the same length for dot product");
            }

            T result = T(0);
            for (size_t i = 0; i < a.shape()[0]; ++i) {
                result += a.at({ i }) * b.at({ i });
            }

            return result;
        }

        // ===== Element-wise Operations =====

        /**
         * @brief Element-wise addition of two tensors
         * @param a First tensor
         * @param b Second tensor
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
            // Check if shapes are compatible for broadcasting
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensor shapes must match for add operation");
            }

            Tensor<T> result(a.shape());

            // Parallel processing for large tensors
            const size_t size = a.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            result.data()[j] = a.data()[j] + b.data()[j];
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise subtraction of two tensors
         * @param a First tensor
         * @param b Second tensor
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
            // Check if shapes are compatible for broadcasting
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensor shapes must match for subtract operation");
            }

            Tensor<T> result(a.shape());

            // Parallel processing for large tensors
            const size_t size = a.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            result.data()[j] = a.data()[j] - b.data()[j];
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise multiplication of two tensors
         * @param a First tensor
         * @param b Second tensor
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
            // Check if shapes are compatible for broadcasting
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensor shapes must match for multiply operation");
            }

            Tensor<T> result(a.shape());

            // Parallel processing for large tensors
            const size_t size = a.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            result.data()[j] = a.data()[j] * b.data()[j];
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise division of two tensors
         * @param a First tensor
         * @param b Second tensor
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b) {
            // Check if shapes are compatible for broadcasting
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensor shapes must match for divide operation");
            }

            Tensor<T> result(a.shape());

            // Parallel processing for large tensors
            const size_t size = a.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            if (b.data()[j] == T(0)) {
                                throw std::domain_error("Division by zero");
                            }
                            result.data()[j] = a.data()[j] / b.data()[j];
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise exponential function
         * @param tensor Input tensor
         * @return Result tensor with exp applied
         */
        template<typename T>
        Tensor<T> exp(const Tensor<T>& tensor) {
            Tensor<T> result(tensor.shape());

            // Parallel processing for large tensors
            const size_t size = tensor.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            // Handle numerical stability: clamp very large values
                            T value = tensor.data()[j];
                            // Max value that won't cause float overflow in exp
                            if (value > T(88.0)) value = T(88.0);
                            result.data()[j] = std::exp(value);
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise natural logarithm
         * @param tensor Input tensor
         * @return Result tensor with log applied
         */
        template<typename T>
        Tensor<T> log(const Tensor<T>& tensor) {
            Tensor<T> result(tensor.shape());

            // Parallel processing for large tensors
            const size_t size = tensor.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            // Small epsilon to prevent log(0)
            const T epsilon = std::numeric_limits<T>::epsilon();

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            // Handle numerical stability: add small epsilon to avoid log(0)
                            T value = tensor.data()[j];
                            if (value <= T(0)) {
                                value = epsilon;
                            }
                            result.data()[j] = std::log(value);
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        /**
         * @brief Element-wise power function
         * @param tensor Input tensor
         * @param exponent Power to raise each element to
         * @return Result tensor with power applied
         */
        template<typename T>
        Tensor<T> pow(const Tensor<T>& tensor, T exponent) {
            Tensor<T> result(tensor.shape());

            // Parallel processing for large tensors
            const size_t size = tensor.size();
            const size_t num_threads = get_thread_pool().size();
            const size_t chunk_size = (size + num_threads - 1) / num_threads;

            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, size);

                if (start < end) {
                    futures.push_back(get_thread_pool().enqueue([&, start, end]() {
                        for (size_t j = start; j < end; ++j) {
                            // Handle numerical stability: check for negative values with non-integer exponents
                            T value = tensor.data()[j];
                            if (value < T(0) && std::floor(exponent) != exponent) {
                                throw std::domain_error("Negative base with non-integer exponent");
                            }
                            result.data()[j] = std::pow(value, exponent);
                        }
                        }));
                }
            }

            // Wait for all threads to finish
            for (auto& future : futures) {
                future.get();
            }

            return result;
        }

        // ===== Convolution and Pooling Operations =====

        /**
         * @brief 2D convolution operation
         * @param input Input tensor [batch, channels, height, width]
         * @param kernel Convolution kernel [out_channels, in_channels, kernel_h, kernel_w]
         * @param stride Stride for convolution (pair of height, width)
         * @param padding Padding (pair of height, width)
         * @return Convolved tensor
         */
        template<typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel,
            const std::pair<size_t, size_t>& stride = { 1, 1 },
            const std::pair<size_t, size_t>& padding = { 0, 0 }) {
            // Check dimensions
            if (input.ndim() != 4) {
                throw std::invalid_argument("Input tensor must be 4D [batch, channels, height, width]");
            }

            if (kernel.ndim() != 4) {
                throw std::invalid_argument("Kernel tensor must be 4D [out_channels, in_channels, kernel_h, kernel_w]");
            }

            // Extract dimensions
            size_t batch_size = input.shape()[0];
            size_t in_channels = input.shape()[1];
            size_t in_height = input.shape()[2];
            size_t in_width = input.shape()[3];

            size_t out_channels = kernel.shape()[0];
            size_t kernel_in_channels = kernel.shape()[1];
            size_t kernel_height = kernel.shape()[2];
            size_t kernel_width = kernel.shape()[3];

            size_t stride_h = stride.first;
            size_t stride_w = stride.second;
            size_t padding_h = padding.first;
            size_t padding_w = padding.second;

            // Check consistency
            if (in_channels != kernel_in_channels) {
                throw std::invalid_argument("Input channels must match kernel input channels");
            }

            // Calculate output dimensions
            size_t out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
            size_t out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;

            // Create output tensor
            Tensor<T> output({ batch_size, out_channels, out_height, out_width });
            output.fill(T(0));

            // Perform convolution
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t oc = 0; oc < out_channels; ++oc) {
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            T sum = T(0);

                            for (size_t ic = 0; ic < in_channels; ++ic) {
                                for (size_t kh = 0; kh < kernel_height; ++kh) {
                                    for (size_t kw = 0; kw < kernel_width; ++kw) {
                                        int h_pos = static_cast<int>(oh * stride_h + kh) - static_cast<int>(padding_h);
                                        int w_pos = static_cast<int>(ow * stride_w + kw) - static_cast<int>(padding_w);

                                        // Check if within input bounds
                                        if (h_pos >= 0 && h_pos < static_cast<int>(in_height) &&
                                            w_pos >= 0 && w_pos < static_cast<int>(in_width)) {
                                            sum += input.at({ b, ic, static_cast<size_t>(h_pos), static_cast<size_t>(w_pos) }) *
                                                kernel.at({ oc, ic, kh, kw });
                                        }
                                    }
                                }
                            }

                            output.at({ b, oc, oh, ow }) = sum;
                        }
                    }
                }
            }

            return output;
        }

        /**
         * @brief Max pooling operation
         * @param input Input tensor [batch, channels, height, width]
         * @param kernel_size Size of pooling window (pair of height, width)
         * @param stride Stride for pooling (pair of height, width)
         * @return Pooled tensor
         */
        template<typename T>
        Tensor<T> max_pool(const Tensor<T>& input,
            const std::pair<size_t, size_t>& kernel_size,
            const std::pair<size_t, size_t>& stride) {
            // Check dimensions
            if (input.ndim() != 4) {
                throw std::invalid_argument("Input tensor must be 4D [batch, channels, height, width]");
            }

            // Extract dimensions
            size_t batch_size = input.shape()[0];
            size_t channels = input.shape()[1];
            size_t in_height = input.shape()[2];
            size_t in_width = input.shape()[3];

            size_t kernel_h = kernel_size.first;
            size_t kernel_w = kernel_size.second;
            size_t stride_h = stride.first;
            size_t stride_w = stride.second;

            // Calculate output dimensions
            size_t out_height = (in_height - kernel_h) / stride_h + 1;
            size_t out_width = (in_width - kernel_w) / stride_w + 1;

            // Create output tensor
            Tensor<T> output({ batch_size, channels, out_height, out_width });

            // Perform max pooling
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            T max_val = std::numeric_limits<T>::lowest();

                            for (size_t kh = 0; kh < kernel_h; ++kh) {
                                for (size_t kw = 0; kw < kernel_w; ++kw) {
                                    size_t h_pos = oh * stride_h + kh;
                                    size_t w_pos = ow * stride_w + kw;

                                    T val = input.at({ b, c, h_pos, w_pos });
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }

                            output.at({ b, c, oh, ow }) = max_val;
                        }
                    }
                }
            }

            return output;
        }

        /**
         * @brief Average pooling operation
         * @param input Input tensor [batch, channels, height, width]
         * @param kernel_size Size of pooling window (pair of height, width)
         * @param stride Stride for pooling (pair of height, width)
         * @return Pooled tensor
         */
        template<typename T>
        Tensor<T> avg_pool(const Tensor<T>& input,
            const std::pair<size_t, size_t>& kernel_size,
            const std::pair<size_t, size_t>& stride) {
            // Check dimensions
            if (input.ndim() != 4) {
                throw std::invalid_argument("Input tensor must be 4D [batch, channels, height, width]");
            }

            // Extract dimensions
            size_t batch_size = input.shape()[0];
            size_t channels = input.shape()[1];
            size_t in_height = input.shape()[2];
            size_t in_width = input.shape()[3];

            size_t kernel_h = kernel_size.first;
            size_t kernel_w = kernel_size.second;
            size_t stride_h = stride.first;
            size_t stride_w = stride.second;

            // Calculate output dimensions
            size_t out_height = (in_height - kernel_h) / stride_h + 1;
            size_t out_width = (in_width - kernel_w) / stride_w + 1;

            // Create output tensor
            Tensor<T> output({ batch_size, channels, out_height, out_width });

            // Perform average pooling
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            T sum = T(0);

                            for (size_t kh = 0; kh < kernel_h; ++kh) {
                                for (size_t kw = 0; kw < kernel_w; ++kw) {
                                    size_t h_pos = oh * stride_h + kh;
                                    size_t w_pos = ow * stride_w + kw;

                                    sum += input.at({ b, c, h_pos, w_pos });
                                }
                            }

                            output.at({ b, c, oh, ow }) = sum / static_cast<T>(kernel_h * kernel_w);
                        }
                    }
                }
            }

            return output;
        }

        // ===== Random Number Generation =====

        /**
         * @brief Generate tensor with random values from normal distribution
         * @param shape Shape of tensor to generate
         * @param mean Mean of normal distribution
         * @param std Standard deviation of normal distribution
         * @return Random tensor
         */
        template<typename T>
        Tensor<T> randn(const std::vector<size_t>& shape, T mean = T(0), T std = T(1)) {
            Tensor<T> result(shape);

            // Create random number generator
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, std);

            // Fill tensor with random values
            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] = dist(gen);
            }

            return result;
        }

        /**
         * @brief Generate tensor with random values from uniform distribution
         * @param shape Shape of tensor to generate
         * @param min Minimum value
         * @param max Maximum value
         * @return Random tensor
         */
        template<typename T>
        Tensor<T> uniform(const std::vector<size_t>& shape, T min = T(0), T max = T(1)) {
            Tensor<T> result(shape);

            // Create random number generator
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(min, max);

            // Fill tensor with random values
            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] = dist(gen);
            }

            return result;
        }

        // ===== Advanced Mathematical Operations =====

        /**
         * @brief Fast Fourier Transform (1D)
         * @param tensor Input tensor (complex values)
         * @param inverse Whether to perform inverse FFT
         * @return Transformed tensor
         */
        template<typename T>
        Tensor<std::complex<T>> fft(const Tensor<std::complex<T>>& tensor, bool inverse = false) {
            // Check dimensions
            if (tensor.ndim() != 1) {
                throw std::invalid_argument("FFT currently only supports 1D tensors");
            }

            size_t n = tensor.shape()[0];

            // Check if size is power of 2 (for simplicity)
            if ((n & (n - 1)) != 0) {
                throw std::invalid_argument("FFT size must be a power of 2");
            }

            Tensor<std::complex<T>> result(tensor.shape());
            std::copy(tensor.data(), tensor.data() + n, result.data());

            // Bit-reverse permutation
            size_t j = 0;
            for (size_t i = 0; i < n - 1; ++i) {
                if (i < j) {
                    std::swap(result.data()[i], result.data()[j]);
                }

                size_t k = n >> 1;
                while (k <= j) {
                    j -= k;
                    k >>= 1;
                }
                j += k;
            }

            // Cooley-Tukey decimation-in-time FFT
            for (size_t step = 2; step <= n; step <<= 1) {
                size_t m = step >> 1;

                std::complex<T> omega_m = std::polar<T>(T(1), (inverse ? T(2) : T(-2)) * Constants<T>::pi / T(step));

                for (size_t k = 0; k < n; k += step) {
                    std::complex<T> omega = std::complex<T>(1, 0);

                    for (size_t j = 0; j < m; ++j) {
                        std::complex<T> t = omega * result.data()[k + j + m];
                        std::complex<T> u = result.data()[k + j];

                        result.data()[k + j] = u + t;
                        result.data()[k + j + m] = u - t;

                        omega *= omega_m;
                    }
                }
            }

            // Scale if inverse
            if (inverse) {
                for (size_t i = 0; i < n; ++i) {
                    result.data()[i] /= static_cast<T>(n);
                }
            }

            return result;
        }

        /**
         * @brief Compute eigenvalues and eigenvectors of symmetric matrix
         * @param matrix Input matrix (must be symmetric)
         * @return Pair of tensors (eigenvalues, eigenvectors)
         */
        template<typename T>
        std::pair<Tensor<T>, Tensor<T>> eig(const Tensor<T>& matrix) {
            // Check dimensions
            if (matrix.ndim() != 2 || matrix.shape()[0] != matrix.shape()[1]) {
                throw std::invalid_argument("Matrix must be square");
            }

            // TODO: Implement eigenvalue decomposition
            // This is a placeholder - a real implementation would use
            // an algorithm like QR iteration

            size_t n = matrix.shape()[0];

            // For now, return identity matrices as placeholder
            Tensor<T> eigenvalues({ n });
            eigenvalues.fill(T(1));

            Tensor<T> eigenvectors({ n, n });
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    eigenvectors.at({ i, j }) = (i == j) ? T(1) : T(0);
                }
            }

            return { eigenvalues, eigenvectors };
        }

        /**
         * @brief Compute Singular Value Decomposition (SVD)
         * @param matrix Input matrix
         * @return Tuple of tensors (U, S, V)
         */
        template<typename T>
        std::tuple<Tensor<T>, Tensor<T>, Tensor<T>> svd(const Tensor<T>& matrix) {
            // Check dimensions
            if (matrix.ndim() != 2) {
                throw std::invalid_argument("Matrix must be 2D for SVD");
            }

            // TODO: Implement SVD
            // This is a placeholder - a real implementation would use
            // an algorithm like divide-and-conquer SVD

            size_t m = matrix.shape()[0];
            size_t n = matrix.shape()[1];
            size_t k = std::min(m, n);

            // For now, return identity matrices as placeholder
            Tensor<T> U({ m, k });
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    U.at({ i, j }) = (i == j) ? T(1) : T(0);
                }
            }

            Tensor<T> S({ k });
            S.fill(T(1));

            Tensor<T> V({ n, k });
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    V.at({ i, j }) = (i == j) ? T(1) : T(0);
                }
            }

            return { U, S, V };
        }

        // ===== Statistical Functions =====

        /**
         * @brief Compute mean along specified axis
         * @param tensor Input tensor
         * @param axis Axis to reduce (if negative, counts from end)
         * @param keepdims Whether to keep reduced dimensions
         * @return Mean tensor
         */
        template<typename T>
        Tensor<T> mean(const Tensor<T>& tensor, int axis = -1, bool keepdims = false) {
            // Handle negative axis
            if (axis < 0) {
                axis += tensor.ndim();
            }

            if (axis < 0 || axis >= static_cast<int>(tensor.ndim())) {
                throw std::invalid_argument("Axis out of range");
            }

            // Create new shape with reduced dimension
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < tensor.ndim(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(tensor.shape()[i]);
                }
                else if (keepdims) {
                    new_shape.push_back(1);
                }
            }

            if (new_shape.empty()) {
                new_shape.push_back(1);
            }

            Tensor<T> result(new_shape);
            result.fill(T(0));

            // Compute mean
            size_t axis_size = tensor.shape()[axis];

            // Iterator for all indices
            std::vector<size_t> idx(tensor.ndim(), 0);
            bool done = false;

            while (!done) {
                // Calculate target index in result tensor
                std::vector<size_t> target_idx;
                size_t dim_count = 0;

                for (size_t i = 0; i < tensor.ndim(); ++i) {
                    if (i != static_cast<size_t>(axis)) {
                        target_idx.push_back(idx[i]);
                        dim_count++;
                    }
                    else if (keepdims) {
                        target_idx.push_back(0);
                        dim_count++;
                    }
                }

                // Add current value to sum
                result.at(target_idx) += tensor.at(idx);

                // Increment indices, skipping the reduction axis
                for (int i = tensor.ndim() - 1; i >= 0; --i) {
                    if (i != axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Divide by axis size to get mean
            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] /= static_cast<T>(axis_size);
            }

            return result;
        }

        /**
         * @brief Compute variance along specified axis
         * @param tensor Input tensor
         * @param axis Axis to reduce (if negative, counts from end)
         * @param keepdims Whether to keep reduced dimensions
         * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
         * @return Variance tensor
         */
        template<typename T>
        Tensor<T> var(const Tensor<T>& tensor, int axis = -1, bool keepdims = false, size_t ddof = 0) {
            // First compute mean
            Tensor<T> means = mean(tensor, axis, true);

            // Handle negative axis
            if (axis < 0) {
                axis += tensor.ndim();
            }

            if (axis < 0 || axis >= static_cast<int>(tensor.ndim())) {
                throw std::invalid_argument("Axis out of range");
            }

            // Create new shape with reduced dimension
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < tensor.ndim(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(tensor.shape()[i]);
                }
                else if (keepdims) {
                    new_shape.push_back(1);
                }
            }

            if (new_shape.empty()) {
                new_shape.push_back(1);
            }

            Tensor<T> result(new_shape);
            result.fill(T(0));

            // Compute variance
            size_t axis_size = tensor.shape()[axis];

            // Iterator for all indices
            std::vector<size_t> idx(tensor.ndim(), 0);
            bool done = false;

            while (!done) {
                // Calculate target index in result tensor
                std::vector<size_t> target_idx;
                std::vector<size_t> mean_idx = idx;

                for (size_t i = 0; i < tensor.ndim(); ++i) {
                    if (i != static_cast<size_t>(axis)) {
                        target_idx.push_back(idx[i]);
                    }
                    else if (keepdims) {
                        target_idx.push_back(0);
                        mean_idx[i] = 0;
                    }
                }

                // Add squared difference to sum
                T diff = tensor.at(idx) - means.at(mean_idx);
                result.at(target_idx) += diff * diff;

                // Increment indices, skipping the reduction axis
                for (int i = tensor.ndim() - 1; i >= 0; --i) {
                    if (i != axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Divide by (axis_size - ddof) to get variance
            T divisor = static_cast<T>(axis_size - ddof);
            if (divisor <= T(0)) {
                throw std::invalid_argument("Divisor must be positive");
            }

            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] /= divisor;
            }

            return result;
        }

        /**
         * @brief Compute p-norm of tensor
         * @param tensor Input tensor
         * @param p Norm order (1 for L1, 2 for L2, etc.)
         * @param axis Axis to reduce (if negative, counts from end)
         * @param keepdims Whether to keep reduced dimensions
         * @return Norm tensor
         */
        template<typename T>
        Tensor<T> norm(const Tensor<T>& tensor, T p = T(2), int axis = -1, bool keepdims = false) {
            // Handle negative axis
            if (axis < 0) {
                axis += tensor.ndim();
            }

            if (axis < 0 || axis >= static_cast<int>(tensor.ndim())) {
                throw std::invalid_argument("Axis out of range");
            }

            // Create new shape with reduced dimension
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < tensor.ndim(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(tensor.shape()[i]);
                }
                else if (keepdims) {
                    new_shape.push_back(1);
                }
            }

            if (new_shape.empty()) {
                new_shape.push_back(1);
            }

            Tensor<T> result(new_shape);
            result.fill(T(0));

            // Special case for infinity norm
            if (p == std::numeric_limits<T>::infinity()) {
                // Iterator for all indices
                std::vector<size_t> idx(tensor.ndim(), 0);
                bool done = false;

                while (!done) {
                    // Calculate target index in result tensor
                    std::vector<size_t> target_idx;

                    for (size_t i = 0; i < tensor.ndim(); ++i) {
                        if (i != static_cast<size_t>(axis)) {
                            target_idx.push_back(idx[i]);
                        }
                        else if (keepdims) {
                            target_idx.push_back(0);
                        }
                    }

                    // Update maximum
                    T abs_val = std::abs(tensor.at(idx));
                    if (abs_val > result.at(target_idx)) {
                        result.at(target_idx) = abs_val;
                    }

                    // Increment indices, skipping the reduction axis
                    for (int i = tensor.ndim() - 1; i >= 0; --i) {
                        if (i != axis) {
                            idx[i]++;
                            if (idx[i] < tensor.shape()[i]) {
                                break;
                            }
                            idx[i] = 0;
                        }

                        if (i == axis) {
                            idx[i]++;
                            if (idx[i] < tensor.shape()[i]) {
                                break;
                            }
                            idx[i] = 0;
                        }

                        if (i == 0) {
                            done = true;
                        }
                    }
                }

                return result;
            }

            // Compute p-norm
            // Iterator for all indices
            std::vector<size_t> idx(tensor.ndim(), 0);
            bool done = false;

            while (!done) {
                // Calculate target index in result tensor
                std::vector<size_t> target_idx;

                for (size_t i = 0; i < tensor.ndim(); ++i) {
                    if (i != static_cast<size_t>(axis)) {
                        target_idx.push_back(idx[i]);
                    }
                    else if (keepdims) {
                        target_idx.push_back(0);
                    }
                }

                // Add p-th power of absolute value to sum
                result.at(target_idx) += std::pow(std::abs(tensor.at(idx)), p);

                // Increment indices, skipping the reduction axis
                for (int i = tensor.ndim() - 1; i >= 0; --i) {
                    if (i != axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Take p-th root
            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] = std::pow(result.data()[i], T(1) / p);
            }

            return result;
        }

        // ===== Numerical Stabilization =====

        /**
         * @brief Clip tensor values to specified range
         * @param tensor Input tensor
         * @param min_val Minimum value
         * @param max_val Maximum value
         * @return Clipped tensor
         */
        template<typename T>
        Tensor<T> clip(const Tensor<T>& tensor, T min_val, T max_val) {
            if (min_val > max_val) {
                throw std::invalid_argument("min_val must be less than or equal to max_val");
            }

            Tensor<T> result(tensor.shape());

            for (size_t i = 0; i < tensor.size(); ++i) {
                T val = tensor.data()[i];
                if (val < min_val) {
                    result.data()[i] = min_val;
                }
                else if (val > max_val) {
                    result.data()[i] = max_val;
                }
                else {
                    result.data()[i] = val;
                }
            }

            return result;
        }

        /**
         * @brief Softplus function (log(1 + exp(x)))
         * @param tensor Input tensor
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> softplus(const Tensor<T>& tensor) {
            Tensor<T> result(tensor.shape());

            for (size_t i = 0; i < tensor.size(); ++i) {
                T x = tensor.data()[i];

                // Handle numerical stability for large values
                if (x > T(20)) {
                    result.data()[i] = x;
                }
                else {
                    result.data()[i] = std::log(T(1) + std::exp(x));
                }
            }

            return result;
        }

        /**
         * @brief Log-sum-exp trick for numerical stability
         * @param tensor Input tensor
         * @param axis Axis to reduce (if negative, counts from end)
         * @param keepdims Whether to keep reduced dimensions
         * @return Result tensor
         */
        template<typename T>
        Tensor<T> logsumexp(const Tensor<T>& tensor, int axis = -1, bool keepdims = false) {
            // Find maximum value along axis
            // Handle negative axis
            if (axis < 0) {
                axis += tensor.ndim();
            }

            if (axis < 0 || axis >= static_cast<int>(tensor.ndim())) {
                throw std::invalid_argument("Axis out of range");
            }

            // Create new shape with reduced dimension
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < tensor.ndim(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(tensor.shape()[i]);
                }
                else if (keepdims) {
                    new_shape.push_back(1);
                }
            }

            if (new_shape.empty()) {
                new_shape.push_back(1);
            }

            Tensor<T> max_vals(new_shape);
            max_vals.fill(std::numeric_limits<T>::lowest());

            // Find maximum values
            // Iterator for all indices
            std::vector<size_t> idx(tensor.ndim(), 0);
            bool done = false;

            while (!done) {
                // Calculate target index in result tensor
                std::vector<size_t> target_idx;

                for (size_t i = 0; i < tensor.ndim(); ++i) {
                    if (i != static_cast<size_t>(axis)) {
                        target_idx.push_back(idx[i]);
                    }
                    else if (keepdims) {
                        target_idx.push_back(0);
                    }
                }

                // Update maximum
                T val = tensor.at(idx);
                if (val > max_vals.at(target_idx)) {
                    max_vals.at(target_idx) = val;
                }

                // Increment indices, skipping the reduction axis
                for (int i = tensor.ndim() - 1; i >= 0; --i) {
                    if (i != axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Compute sum of exp(x - max_x)
            Tensor<T> sum_exp(new_shape);
            sum_exp.fill(T(0));

            // Reset for another pass
            std::fill(idx.begin(), idx.end(), 0);
            done = false;

            while (!done) {
                // Calculate target index in result tensor
                std::vector<size_t> target_idx;

                for (size_t i = 0; i < tensor.ndim(); ++i) {
                    if (i != static_cast<size_t>(axis)) {
                        target_idx.push_back(idx[i]);
                    }
                    else if (keepdims) {
                        target_idx.push_back(0);
                    }
                }

                // Add to sum
                T val = tensor.at(idx);
                T max_val = max_vals.at(target_idx);
                sum_exp.at(target_idx) += std::exp(val - max_val);

                // Increment indices, skipping the reduction axis
                for (int i = tensor.ndim() - 1; i >= 0; --i) {
                    if (i != axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == axis) {
                        idx[i]++;
                        if (idx[i] < tensor.shape()[i]) {
                            break;
                        }
                        idx[i] = 0;
                    }

                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Compute log(sum(exp(x - max_x))) + max_x
            Tensor<T> result(new_shape);

            for (size_t i = 0; i < result.size(); ++i) {
                result.data()[i] = std::log(sum_exp.data()[i]) + max_vals.data()[i];
            }

            return result;
        }

} // namespace math_utils
} // namespace tensor

#endif // MATH_UTILS_H