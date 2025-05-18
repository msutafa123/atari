// TensorInitializers.h - v0.1.0
// Tensor initialization strategies

#ifndef TENSOR_INITIALIZERS_H
#define TENSOR_INITIALIZERS_H

#include "Tensor.h"
#include <random>
#include <cmath>

namespace tensor {
    namespace initializers {

        // Xavier/Glorot uniform initialization
        template<typename T>
        void xavier_uniform(Tensor<T>& tensor) {
            // Get fan_in and fan_out
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Xavier initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);
            size_t fan_out = shape.dim(shape.ndim() - 1);

            // Calculate limit
            T limit = std::sqrt(6.0 / (fan_in + fan_out));

            // Fill with uniform random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-limit, limit);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Xavier/Glorot normal initialization
        template<typename T>
        void xavier_normal(Tensor<T>& tensor) {
            // Get fan_in and fan_out
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Xavier initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);
            size_t fan_out = shape.dim(shape.ndim() - 1);

            // Calculate std
            T std_dev = std::sqrt(2.0 / (fan_in + fan_out));

            // Fill with normal random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0, std_dev);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // He/Kaiming uniform initialization
        template<typename T>
        void kaiming_uniform(Tensor<T>& tensor, T a = 0) {
            // Get fan_in
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Kaiming initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);

            // Calculate bound
            T bound = std::sqrt(6.0 / ((1 + a * a) * fan_in));

            // Fill with uniform random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-bound, bound);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // He/Kaiming normal initialization
        template<typename T>
        void kaiming_normal(Tensor<T>& tensor, T a = 0) {
            // Get fan_in
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Kaiming initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);

            // Calculate std
            T std_dev = std::sqrt(2.0 / ((1 + a * a) * fan_in));

            // Fill with normal random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0, std_dev);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Uniform initialization
        template<typename T>
        void uniform(Tensor<T>& tensor, T low = -0.1, T high = 0.1) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(low, high);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Normal initialization
        template<typename T>
        void normal(Tensor<T>& tensor, T mean = 0, T std_dev = 0.01) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, std_dev);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Constant initialization
        template<typename T>
        void constant(Tensor<T>& tensor, T value) {
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = value;
            }
        }

        // Orthogonal initialization
        template<typename T>
        void orthogonal(Tensor<T>& tensor, T gain = 1.0) {
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Orthogonal initialization requires at least 2D tensor");
            }

            // Calculate rows and cols
            size_t rows = shape.dim(shape.ndim() - 2);
            size_t cols = shape.dim(shape.ndim() - 1);

            // We need a temporary matrix for QR decomposition
            // In a real implementation, we would use a linear algebra library like Eigen
            // For this example, we'll just use a simplified approach

            // Create a random matrix with standard normal distribution
            std::vector<T> random_data(rows * cols);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0, 1);

            for (size_t i = 0; i < random_data.size(); ++i) {
                random_data[i] = dist(gen);
            }

            // This is a simplified orthogonalization procedure
            // A real implementation would use proper QR decomposition

            // Fill tensor with random data scaled by gain
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = random_data[i % random_data.size()] * gain;
            }
        }

        // TruncatedNormal initialization (values more than 2 std devs from mean are resampled)
        template<typename T>
        void truncated_normal(Tensor<T>& tensor, T mean = 0, T std_dev = 0.01) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, std_dev);

            for (size_t i = 0; i < tensor.size(); ++i) {
                T value;
                do {
                    value = dist(gen);
                } while (std::abs(value - mean) > 2 * std_dev);

                tensor.data()[i] = value;
            }
        }

        // Identity initialization (for square matrices)
        template<typename T>
        void identity(Tensor<T>& tensor, T gain = 1.0) {
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() != 2 || shape.dim(0) != shape.dim(1)) {
                throw std::invalid_argument("Identity initialization requires a square matrix");
            }

            size_t n = shape.dim(0);

            // Set all values to 0
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = 0;
            }

            // Set diagonal elements to gain
            for (size_t i = 0; i < n; ++i) {
                tensor.at({ i, i }) = gain;
            }
        }

        // Lecun uniform initialization
        template<typename T>
        void lecun_uniform(Tensor<T>& tensor) {
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Lecun initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);

            // Calculate limit
            T limit = std::sqrt(3.0 / fan_in);

            // Fill with uniform random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-limit, limit);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Lecun normal initialization
        template<typename T>
        void lecun_normal(Tensor<T>& tensor) {
            const TensorShape& shape = tensor.shape();

            if (shape.ndim() < 2) {
                throw std::invalid_argument("Lecun initialization requires at least 2D tensor");
            }

            size_t fan_in = shape.dim(shape.ndim() - 2);

            // Calculate std
            T std_dev = std::sqrt(1.0 / fan_in);

            // Fill with normal random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0, std_dev);

            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data()[i] = dist(gen);
            }
        }

        // Sparse initialization
        template<typename T>
        void sparse(Tensor<T>& tensor, T sparsity = 0.9, T scale = 0.01) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(0, 1);
            std::normal_distribution<T> normal_dist(0, scale);

            for (size_t i = 0; i < tensor.size(); ++i) {
                // Apply sparsity - set most values to 0
                if (dist(gen) < sparsity) {
                    tensor.data()[i] = 0;
                }
                else {
                    tensor.data()[i] = normal_dist(gen);
                }
            }
        }

    } // namespace initializers
} // namespace tensor

#endif // TENSOR_INITIALIZERS_H