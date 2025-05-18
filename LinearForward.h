// LinearForward.h - v1.0.4
// Linear layer implementation - C++17 standards compliant

#ifndef LINEAR_FORWARD_H
#define LINEAR_FORWARD_H

#include "ForwardModule.h"
#include "Tensor.h"
#include "TensorShape.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <random>
#include <cmath>

namespace tensor {
    namespace forward {

        // Initialization methods for weights
        enum class LinearInit {
            ZEROS,
            ONES,
            UNIFORM,
            NORMAL,
            XAVIER_UNIFORM,
            XAVIER_NORMAL,
            KAIMING_UNIFORM,
            KAIMING_NORMAL
        };

        template<typename T>
        class LinearForward : public ForwardModule<T> {
        public:
            // Constructor with dimensions
            LinearForward(size_t input_dim, size_t output_dim, bool use_bias = true)
                : input_dim_(input_dim), output_dim_(output_dim), use_bias_(use_bias) {

                // Create weight tensor - use std::vector<size_t> for dimensions
                std::vector<size_t> weight_dims = { input_dim_, output_dim_ };
                weights_ = Tensor<T>(weight_dims);

                // Create bias tensor if needed
                if (use_bias_) {
                    std::vector<size_t> bias_dims = { output_dim_ };
                    bias_ = Tensor<T>(bias_dims);
                    bias_.fill(T(0));
                }

                // Initialize weights using Kaiming uniform by default
                initialize_weights(LinearInit::KAIMING_UNIFORM);
            }

            // Constructor with input shape
            LinearForward(const TensorShape& input_shape, size_t output_dim, bool use_bias = true)
                : input_dim_(input_shape.ndim() > 0 ? input_shape.dim(input_shape.ndim() - 1) : 0),
                output_dim_(output_dim),
                use_bias_(use_bias) {

                if (input_dim_ == 0) {
                    throw std::invalid_argument("Input shape must have at least one dimension");
                }

                // Create weight tensor
                std::vector<size_t> weight_dims = { input_dim_, output_dim_ };
                weights_ = Tensor<T>(weight_dims);

                // Create bias tensor if needed
                if (use_bias_) {
                    std::vector<size_t> bias_dims = { output_dim_ };
                    bias_ = Tensor<T>(bias_dims);
                    bias_.fill(T(0));
                }

                // Initialize weights using Kaiming uniform by default
                initialize_weights(LinearInit::KAIMING_UNIFORM);
            }

            // Forward pass implementation
            Tensor<T> forward(const Tensor<T>& input) override {
                // Validate input dimensions
                if (input.ndim() < 1) {
                    throw std::invalid_argument("Linear layer requires input with at least 1 dimension");
                }

                size_t last_dim = input.ndim() - 1;
                if (input.dim(last_dim) != input_dim_) {
                    throw std::invalid_argument(
                        "Input dimension mismatch: expected " + std::to_string(input_dim_) +
                        ", got " + std::to_string(input.dim(last_dim)));
                }

                // Create output tensor with proper shape
                std::vector<size_t> output_dims;
                for (size_t i = 0; i < input.ndim(); ++i) {
                    if (i == last_dim) {
                        output_dims.push_back(output_dim_);
                    }
                    else {
                        output_dims.push_back(input.dim(i));
                    }
                }
                Tensor<T> output(output_dims);

                // Calculate batch size (product of all dimensions except the last one)
                size_t batch_size = 1;
                for (size_t i = 0; i < last_dim; ++i) {
                    batch_size *= input.dim(i);
                }

                // For 2D input (most common case), use optimized implementation
                if (input.ndim() == 2) {
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t o = 0; o < output_dim_; ++o) {
                            // Start with bias if available
                            T sum = T(0);
                            if (use_bias_) {
                                std::vector<size_t> bias_idx = { o };
                                sum = bias_.at(bias_idx);
                            }

                            // Compute weighted sum
                            for (size_t i = 0; i < input_dim_; ++i) {
                                std::vector<size_t> input_idx = { b, i };
                                std::vector<size_t> weight_idx = { i, o };
                                sum += input.at(input_idx) * weights_.at(weight_idx);
                            }

                            std::vector<size_t> output_idx = { b, o };
                            output.at(output_idx) = sum;
                        }
                    }
                }
                // For higher dimensional inputs, use a more general approach
                else {
                    // Process each element
                    std::vector<size_t> input_idx(input.ndim(), 0);
                    std::vector<size_t> output_idx(output_dims.size(), 0);

                    // Iterate through all batch elements
                    bool done = false;
                    while (!done) {
                        // Copy batch indices (all except last dimension)
                        for (size_t i = 0; i < last_dim; ++i) {
                            output_idx[i] = input_idx[i];
                        }

                        // For each output feature
                        for (size_t o = 0; o < output_dim_; ++o) {
                            // Set output feature index
                            output_idx[last_dim] = o;

                            // Start with bias
                            T sum = T(0);
                            if (use_bias_) {
                                std::vector<size_t> bias_idx = { o };
                                sum = bias_.at(bias_idx);
                            }

                            // Compute weighted sum over input features
                            for (size_t i = 0; i < input_dim_; ++i) {
                                // Set input feature index
                                input_idx[last_dim] = i;

                                // Get weight
                                std::vector<size_t> weight_idx = { i, o };

                                // Add to sum
                                sum += input.at(input_idx) * weights_.at(weight_idx);
                            }

                            // Store result
                            output.at(output_idx) = sum;
                        }

                        // Update batch indices for next iteration
                        bool carry = true;
                        for (int i = static_cast<int>(last_dim) - 1; i >= 0 && carry; --i) {
                            input_idx[i]++;
                            if (input_idx[i] >= input.dim(i)) {
                                input_idx[i] = 0;
                            }
                            else {
                                carry = false;
                            }
                        }

                        // If we carried all the way, we're done
                        if (carry) {
                            done = true;
                        }
                    }
                }

                return output;
            }

            // Get parameters of the layer
            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;
                params.push_back(&weights_);
                if (use_bias_) {
                    params.push_back(&bias_);
                }
                return params;
            }

            // Get layer name
            std::string name() const override {
                return "Linear(" + std::to_string(input_dim_) + ", " + std::to_string(output_dim_) + ")";
            }

            // Get input dimension
            size_t input_dim() const { return input_dim_; }

            // Get output dimension
            size_t output_dim() const { return output_dim_; }

            // Check if layer uses bias
            bool uses_bias() const { return use_bias_; }

            // Get weights tensor
            Tensor<T>& weights() { return weights_; }
            const Tensor<T>& weights() const { return weights_; }

            // Get bias tensor
            Tensor<T>& bias() {
                if (!use_bias_) {
                    throw std::logic_error("Linear layer does not use bias");
                }
                return bias_;
            }

            const Tensor<T>& bias() const {
                if (!use_bias_) {
                    throw std::logic_error("Linear layer does not use bias");
                }
                return bias_;
            }

            // Set weights
            void set_weights(const Tensor<T>& new_weights) {
                if (new_weights.ndim() != 2 ||
                    new_weights.dim(0) != input_dim_ ||
                    new_weights.dim(1) != output_dim_) {
                    throw std::invalid_argument("Weight tensor shape mismatch");
                }
                weights_ = new_weights;
            }

            // Set bias
            void set_bias(const Tensor<T>& new_bias) {
                if (!use_bias_) {
                    throw std::logic_error("Linear layer does not use bias");
                }
                if (new_bias.ndim() != 1 || new_bias.dim(0) != output_dim_) {
                    throw std::invalid_argument("Bias tensor shape mismatch");
                }
                bias_ = new_bias;
            }

            // Initialize weights using specified method
            void initialize_weights(LinearInit method) {
                std::random_device rd;
                std::mt19937 gen(rd());

                switch (method) {
                case LinearInit::ZEROS:
                    weights_.fill(T(0));
                    break;

                case LinearInit::ONES:
                    weights_.fill(T(1));
                    break;

                case LinearInit::UNIFORM: {
                    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }

                case LinearInit::NORMAL: {
                    std::normal_distribution<float> dist(0.0f, 0.01f);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }

                case LinearInit::XAVIER_UNIFORM: {
                    float limit = std::sqrt(6.0f / (input_dim_ + output_dim_));
                    std::uniform_real_distribution<float> dist(-limit, limit);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }

                case LinearInit::XAVIER_NORMAL: {
                    float stddev = std::sqrt(2.0f / (input_dim_ + output_dim_));
                    std::normal_distribution<float> dist(0.0f, stddev);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }

                case LinearInit::KAIMING_UNIFORM: {
                    float limit = std::sqrt(6.0f / input_dim_);
                    std::uniform_real_distribution<float> dist(-limit, limit);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }

                case LinearInit::KAIMING_NORMAL: {
                    float stddev = std::sqrt(2.0f / input_dim_);
                    std::normal_distribution<float> dist(0.0f, stddev);
                    for (size_t i = 0; i < input_dim_; ++i) {
                        for (size_t j = 0; j < output_dim_; ++j) {
                            std::vector<size_t> idx = { i, j };
                            weights_.at(idx) = static_cast<T>(dist(gen));
                        }
                    }
                    break;
                }
                }
            }

        private:
            size_t input_dim_;
            size_t output_dim_;
            bool use_bias_;
            Tensor<T> weights_;
            Tensor<T> bias_;
        };

    } // namespace forward
} // namespace tensor

#endif // LINEAR_FORWARD_H