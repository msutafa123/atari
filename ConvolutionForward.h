// ConvolutionForward.h - v0.2.0
// Convolution forward pass implementation

#ifndef CONVOLUTION_FORWARD_H
#define CONVOLUTION_FORWARD_H

#include "ForwardModule.h"
#include "TensorInitializers.h"
#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>

namespace tensor {
    namespace forward {

        // Padding modes
        enum class PaddingMode {
            VALID,  // No padding
            SAME,   // Padding to maintain input spatial dimensions
            FULL,   // Full convolution
            CUSTOM  // Custom padding values
        };

        template<typename T>
        class Conv2DForward : public ForwardModule<T> {
        public:
            // Constructor with basic parameters
            Conv2DForward(size_t in_channels, size_t out_channels,
                size_t kernel_size, size_t stride = 1,
                PaddingMode padding_mode = PaddingMode::VALID,
                size_t padding = 0, size_t dilation = 1,
                bool use_bias = true, size_t groups = 1)
                : in_channels_(in_channels), out_channels_(out_channels),
                kernel_size_h_(kernel_size), kernel_size_w_(kernel_size),
                stride_h_(stride), stride_w_(stride),
                padding_mode_(padding_mode), padding_h_(padding), padding_w_(padding),
                dilation_h_(dilation), dilation_w_(dilation),
                use_bias_(use_bias), groups_(groups) {

                validate_parameters();
                initialize_parameters();
            }

            // Constructor with separate dimensions
            Conv2DForward(size_t in_channels, size_t out_channels,
                std::pair<size_t, size_t> kernel_size,
                std::pair<size_t, size_t> stride = { 1, 1 },
                PaddingMode padding_mode = PaddingMode::VALID,
                std::pair<size_t, size_t> padding = { 0, 0 },
                std::pair<size_t, size_t> dilation = { 1, 1 },
                bool use_bias = true, size_t groups = 1)
                : in_channels_(in_channels), out_channels_(out_channels),
                kernel_size_h_(kernel_size.first), kernel_size_w_(kernel_size.second),
                stride_h_(stride.first), stride_w_(stride.second),
                padding_mode_(padding_mode),
                padding_h_(padding.first), padding_w_(padding.second),
                dilation_h_(dilation.first), dilation_w_(dilation.second),
                use_bias_(use_bias), groups_(groups) {

                validate_parameters();
                initialize_parameters();
            }

            // Forward pass implementation
            Tensor<T> forward(const Tensor<T>& input) override {
                // Input must be [batch, channels, height, width]
                validate_input(input);

                size_t batch_size = input.shape().dim(0);
                size_t in_height = input.shape().dim(2);
                size_t in_width = input.shape().dim(3);

                // Adjust padding if using SAME mode
                if (padding_mode_ == PaddingMode::SAME) {
                    // Calculate padding needed to preserve spatial dimensions
                    size_t effective_kernel_h = kernel_size_h_ + (kernel_size_h_ - 1) * (dilation_h_ - 1);
                    size_t effective_kernel_w = kernel_size_w_ + (kernel_size_w_ - 1) * (dilation_w_ - 1);

                    padding_h_ = ((in_height - 1) * stride_h_ + effective_kernel_h - in_height) / 2;
                    padding_w_ = ((in_width - 1) * stride_w_ + effective_kernel_w - in_width) / 2;
                }

                // Calculate output dimensions
                size_t out_height = calculate_output_size(in_height, kernel_size_h_, padding_h_, stride_h_, dilation_h_);
                size_t out_width = calculate_output_size(in_width, kernel_size_w_, padding_w_, stride_w_, dilation_w_);

                // Create output tensor
                std::vector<size_t> output_dims = { batch_size, out_channels_, out_height, out_width };
                Tensor<T> output(TensorShape(output_dims));
                output.fill(T(0));

                // Perform convolution by groups
                size_t in_channels_per_group = in_channels_ / groups_;
                size_t out_channels_per_group = out_channels_ / groups_;

                for (size_t g = 0; g < groups_; ++g) {
                    size_t in_ch_start = g * in_channels_per_group;
                    size_t out_ch_start = g * out_channels_per_group;

                    // Perform convolution for this group
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t c_out = 0; c_out < out_channels_per_group; ++c_out) {
                            size_t actual_c_out = out_ch_start + c_out;

                            // Apply bias if enabled
                            if (use_bias_) {
                                for (size_t h_out = 0; h_out < out_height; ++h_out) {
                                    for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                        output.at({ b, actual_c_out, h_out, w_out }) = bias_.at({ actual_c_out });
                                    }
                                }
                            }

                            // Apply convolution kernel
                            for (size_t c_in = 0; c_in < in_channels_per_group; ++c_in) {
                                size_t actual_c_in = in_ch_start + c_in;

                                for (size_t h_out = 0; h_out < out_height; ++h_out) {
                                    for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                        // Calculate input position
                                        size_t h_in_start = h_out * stride_h_ - padding_h_;
                                        size_t w_in_start = w_out * stride_w_ - padding_w_;

                                        // Apply convolution at this position
                                        for (size_t kh = 0; kh < kernel_size_h_; ++kh) {
                                            size_t h_in = h_in_start + kh * dilation_h_;

                                            // Skip if outside input bounds
                                            if (h_in >= in_height) continue;

                                            for (size_t kw = 0; kw < kernel_size_w_; ++kw) {
                                                size_t w_in = w_in_start + kw * dilation_w_;

                                                // Skip if outside input bounds
                                                if (w_in >= in_width) continue;

                                                // Skip if in padding area
                                                if (h_in < 0 || w_in < 0) continue;

                                                // Input value
                                                T in_val = input.at({ b, actual_c_in, h_in, w_in });

                                                // Weight value
                                                T weight_val = weights_.at({ actual_c_out, c_in, kh, kw });

                                                // Accumulate
                                                output.at({ b, actual_c_out, h_out, w_out }) += in_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }

            // Backward pass (computes gradients)
            std::tuple<Tensor<T>, Tensor<T>, Tensor<T>> backward(
                const Tensor<T>& input, const Tensor<T>& grad_output) {

                // Input must be [batch, channels, height, width]
                validate_input(input);

                size_t batch_size = input.shape().dim(0);
                size_t in_height = input.shape().dim(2);
                size_t in_width = input.shape().dim(3);

                size_t out_height = grad_output.shape().dim(2);
                size_t out_width = grad_output.shape().dim(3);

                // Create gradient tensors
                Tensor<T> grad_input(input.shape());
                Tensor<T> grad_weights(weights_.shape());
                Tensor<T> grad_bias;

                grad_input.fill(T(0));
                grad_weights.fill(T(0));

                if (use_bias_) {
                    grad_bias = Tensor<T>(bias_.shape());
                    grad_bias.fill(T(0));

                    // Compute gradient w.r.t bias (sum of gradients)
                    for (size_t c_out = 0; c_out < out_channels_; ++c_out) {
                        T sum = T(0);

                        for (size_t b = 0; b < batch_size; ++b) {
                            for (size_t h_out = 0; h_out < out_height; ++h_out) {
                                for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                    sum += grad_output.at({ b, c_out, h_out, w_out });
                                }
                            }
                        }

                        grad_bias.at({ c_out }) = sum;
                    }
                }

                // Perform backpropagation by groups
                size_t in_channels_per_group = in_channels_ / groups_;
                size_t out_channels_per_group = out_channels_ / groups_;

                for (size_t g = 0; g < groups_; ++g) {
                    size_t in_ch_start = g * in_channels_per_group;
                    size_t out_ch_start = g * out_channels_per_group;

                    // Compute gradient w.r.t weights
                    for (size_t c_out = 0; c_out < out_channels_per_group; ++c_out) {
                        size_t actual_c_out = out_ch_start + c_out;

                        for (size_t c_in = 0; c_in < in_channels_per_group; ++c_in) {
                            size_t actual_c_in = in_ch_start + c_in;

                            for (size_t kh = 0; kh < kernel_size_h_; ++kh) {
                                for (size_t kw = 0; kw < kernel_size_w_; ++kw) {
                                    T sum = T(0);

                                    for (size_t b = 0; b < batch_size; ++b) {
                                        for (size_t h_out = 0; h_out < out_height; ++h_out) {
                                            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                                // Calculate input position
                                                size_t h_in = h_out * stride_h_ - padding_h_ + kh * dilation_h_;
                                                size_t w_in = w_out * stride_w_ - padding_w_ + kw * dilation_w_;

                                                // Skip if outside input bounds
                                                if (h_in >= in_height || w_in >= in_width ||
                                                    h_in < 0 || w_in < 0) continue;

                                                sum += input.at({ b, actual_c_in, h_in, w_in }) *
                                                    grad_output.at({ b, actual_c_out, h_out, w_out });
                                            }
                                        }
                                    }

                                    grad_weights.at({ actual_c_out, c_in, kh, kw }) = sum;
                                }
                            }
                        }
                    }

                    // Compute gradient w.r.t input
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t c_in = 0; c_in < in_channels_per_group; ++c_in) {
                            size_t actual_c_in = in_ch_start + c_in;

                            for (size_t h_in = 0; h_in < in_height; ++h_in) {
                                for (size_t w_in = 0; w_in < in_width; ++w_in) {
                                    T sum = T(0);

                                    for (size_t c_out = 0; c_out < out_channels_per_group; ++c_out) {
                                        size_t actual_c_out = out_ch_start + c_out;

                                        for (size_t kh = 0; kh < kernel_size_h_; ++kh) {
                                            for (size_t kw = 0; kw < kernel_size_w_; ++kw) {
                                                // Calculate output position
                                                int h_out = (h_in + padding_h_ - kh * dilation_h_) / stride_h_;
                                                int w_out = (w_in + padding_w_ - kw * dilation_w_) / stride_w_;

                                                // Check if the output position is valid
                                                if (h_out < 0 || h_out >= static_cast<int>(out_height) ||
                                                    w_out < 0 || w_out >= static_cast<int>(out_width)) continue;

                                                // Check if this input position contributes to the output
                                                if ((h_in + padding_h_ - kh * dilation_h_) % stride_h_ != 0 ||
                                                    (w_in + padding_w_ - kw * dilation_w_) % stride_w_ != 0) continue;

                                                sum += weights_.at({ actual_c_out, c_in, kh, kw }) *
                                                    grad_output.at({ b, actual_c_out, static_cast<size_t>(h_out),
                                                                    static_cast<size_t>(w_out) });
                                            }
                                        }
                                    }

                                    grad_input.at({ b, actual_c_in, h_in, w_in }) = sum;
                                }
                            }
                        }
                    }
                }

                return { grad_input, grad_weights, grad_bias };
            }

            // Get parameters
            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;
                params.push_back(&weights_);
                if (use_bias_) {
                    params.push_back(&bias_);
                }
                return params;
            }

            // Calculate output shape for a given input shape
            TensorShape output_shape(const TensorShape& input_shape) const {
                if (input_shape.ndim() != 4) {
                    throw std::invalid_argument("Conv2D requires 4D input [batch, channels, height, width]");
                }

                if (input_shape.dim(1) != in_channels_) {
                    throw std::invalid_argument("Input channel count mismatch");
                }

                size_t in_height = input_shape.dim(2);
                size_t in_width = input_shape.dim(3);

                // Adjust padding for SAME mode
                size_t padding_h = padding_h_;
                size_t padding_w = padding_w_;

                if (padding_mode_ == PaddingMode::SAME) {
                    size_t effective_kernel_h = kernel_size_h_ + (kernel_size_h_ - 1) * (dilation_h_ - 1);
                    size_t effective_kernel_w = kernel_size_w_ + (kernel_size_w_ - 1) * (dilation_w_ - 1);

                    padding_h = ((in_height - 1) * stride_h_ + effective_kernel_h - in_height) / 2;
                    padding_w = ((in_width - 1) * stride_w_ + effective_kernel_w - in_width) / 2;
                }

                size_t out_height = calculate_output_size(in_height, kernel_size_h_, padding_h, stride_h_, dilation_h_);
                size_t out_width = calculate_output_size(in_width, kernel_size_w_, padding_w, stride_w_, dilation_w_);

                return TensorShape({ input_shape.dim(0), out_channels_, out_height, out_width });
            }

            // Get input channels
            size_t in_channels() const { return in_channels_; }

            // Get output channels
            size_t out_channels() const { return out_channels_; }

            // Get kernel size
            std::pair<size_t, size_t> kernel_size() const { return { kernel_size_h_, kernel_size_w_ }; }

            // Get stride
            std::pair<size_t, size_t> stride() const { return { stride_h_, stride_w_ }; }

            // Get padding
            std::pair<size_t, size_t> padding() const { return { padding_h_, padding_w_ }; }

            // Get dilation
            std::pair<size_t, size_t> dilation() const { return { dilation_h_, dilation_w_ }; }

            // Get padding mode
            PaddingMode padding_mode() const { return padding_mode_; }

            // Check if bias is used
            bool uses_bias() const { return use_bias_; }

            // Get groups
            size_t groups() const { return groups_; }

            // Get weights
            Tensor<T>& weights() { return weights_; }
            const Tensor<T>& weights() const { return weights_; }

            // Get bias
            Tensor<T>& bias() {
                if (!use_bias_) {
                    throw std::logic_error("Conv2D layer does not use bias");
                }
                return bias_;
            }

            const Tensor<T>& bias() const {
                if (!use_bias_) {
                    throw std::logic_error("Conv2D layer does not use bias");
                }
                return bias_;
            }

            // Set weights
            void set_weights(const Tensor<T>& weights) {
                if (weights.shape() != weights_.shape()) {
                    throw std::invalid_argument("Weights shape mismatch");
                }
                weights_ = weights;
            }

            // Set bias
            void set_bias(const Tensor<T>& bias) {
                if (!use_bias_) {
                    throw std::logic_error("Conv2D layer does not use bias");
                }
                if (bias.shape() != bias_.shape()) {
                    throw std::invalid_argument("Bias shape mismatch");
                }
                bias_ = bias;
            }

            // Module name
            std::string name() const override {
                return "Conv2D(" + std::to_string(in_channels_) + ", " +
                    std::to_string(out_channels_) + ", " +
                    "kernel_size=(" + std::to_string(kernel_size_h_) + ", " +
                    std::to_string(kernel_size_w_) + "))";
            }

        private:
            size_t in_channels_;
            size_t out_channels_;
            size_t kernel_size_h_;
            size_t kernel_size_w_;
            size_t stride_h_;
            size_t stride_w_;
            PaddingMode padding_mode_;
            size_t padding_h_;
            size_t padding_w_;
            size_t dilation_h_;
            size_t dilation_w_;
            bool use_bias_;
            size_t groups_;
            Tensor<T> weights_;
            Tensor<T> bias_;

            // Validate parameters
            void validate_parameters() {
                if (in_channels_ == 0 || out_channels_ == 0) {
                    throw std::invalid_argument("Channel counts must be positive");
                }

                if (kernel_size_h_ == 0 || kernel_size_w_ == 0) {
                    throw std::invalid_argument("Kernel size must be positive");
                }

                if (stride_h_ == 0 || stride_w_ == 0) {
                    throw std::invalid_argument("Stride must be positive");
                }

                if (dilation_h_ == 0 || dilation_w_ == 0) {
                    throw std::invalid_argument("Dilation must be positive");
                }

                if (groups_ == 0 || in_channels_ % groups_ != 0 || out_channels_ % groups_ != 0) {
                    throw std::invalid_argument("Invalid groups value");
                }
            }

            // Validate input shape
            void validate_input(const Tensor<T>& input) {
                if (input.shape().ndim() != 4) {
                    throw std::invalid_argument("Conv2D requires 4D input [batch, channels, height, width]");
                }

                if (input.shape().dim(1) != in_channels_) {
                    throw std::invalid_argument("Input channel count mismatch");
                }
            }

            // Calculate output size for a dimension
            size_t calculate_output_size(size_t input_size, size_t kernel_size,
                size_t padding, size_t stride, size_t dilation) const {
                // Effective kernel size with dilation
                size_t effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1);

                // Formula: floor((input_size + 2*padding - effective_kernel_size) / stride) + 1
                if (input_size + 2 * padding < effective_kernel_size) {
                    return 0;
                }

                return (input_size + 2 * padding - effective_kernel_size) / stride + 1;
            }

            // Initialize parameters
            void initialize_parameters() {
                // Initialize weights
                std::vector<size_t> weight_dims = { out_channels_, in_channels_ / groups_,
                                                    kernel_size_h_, kernel_size_w_ };
                weights_ = Tensor<T>(TensorShape(weight_dims));

                // Initialize bias if used
                if (use_bias_) {
                    std::vector<size_t> bias_dims = { out_channels_ };
                    bias_ = Tensor<T>(TensorShape(bias_dims));
                    bias_.fill(T(0));
                }

                // Kaiming initialization (suitable for Conv layers)
                initializers::kaiming_uniform(weights_);
            }
        };

    } // namespace forward
} // namespace tensor

#endif // CONVOLUTION_FORWARD_H