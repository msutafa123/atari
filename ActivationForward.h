// ActivationForward.h - v0.3.0
// Activation function forward implementations
// C++17 standards compliant

#ifndef ACTIVATION_FORWARD_H
#define ACTIVATION_FORWARD_H

#include "ForwardModule.h"
#include "TensorMath.h"
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <stdexcept>
#include <limits>

namespace tensor {
    namespace forward {

        // Forward declarations of all activation classes
        template<typename T> class ReLUForward;
        template<typename T> class LeakyReLUForward;
        template<typename T> class SigmoidForward;
        template<typename T> class TanhForward;
        template<typename T> class ELUForward;
        template<typename T> class SELUForward;
        template<typename T> class SoftplusForward;
        template<typename T> class SoftsignForward;
        template<typename T> class SwishForward;
        template<typename T> class MishForward;
        template<typename T> class GELUForward;

        // Base class for activation functions
        template<typename T>
        class ActivationForward : public ForwardModule<T> {
        public:
            virtual ~ActivationForward() = default;

            // Factory method will be defined at the end of the file
            static std::shared_ptr<ActivationForward<T>> create(const std::string& name);
        };

        // ReLU Activation
        template<typename T>
        class ReLUForward : public ActivationForward<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                return tensor::math::relu(input);
            }

            std::string name() const override {
                return "ReLU";
            }

            // Derivative of ReLU
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    result_data[i] = input_data[i] > T(0) ? grad_data[i] : T(0);
                }
                return result;
            }
        };

        // Leaky ReLU Activation
        template<typename T>
        class LeakyReLUForward : public ActivationForward<T> {
        public:
            explicit LeakyReLUForward(T alpha = T(0.01)) : alpha_(alpha) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                return tensor::math::leaky_relu(input, alpha_);
            }

            std::string name() const override {
                return "LeakyReLU";
            }

            // Derivative of Leaky ReLU
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    result_data[i] = input_data[i] > T(0) ? grad_data[i] : alpha_ * grad_data[i];
                }
                return result;
            }

            // Get alpha value
            T alpha() const { return alpha_; }

            // Set alpha value
            void set_alpha(T alpha) { alpha_ = alpha; }

        private:
            T alpha_;
        };

        // Sigmoid Activation
        template<typename T>
        class SigmoidForward : public ActivationForward<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                return tensor::math::sigmoid(input);
            }

            std::string name() const override {
                return "Sigmoid";
            }

            // Derivative of Sigmoid: sigmoid * (1 - sigmoid)
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> sigmoid_val = forward(input);
                Tensor<T> result(input.shape());
                const T* sigmoid_data = sigmoid_val.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T s = sigmoid_data[i];
                    result_data[i] = grad_data[i] * s * (T(1) - s);
                }

                return result;
            }
        };

        // Tanh Activation
        template<typename T>
        class TanhForward : public ActivationForward<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                return tensor::math::tanh(input);
            }

            std::string name() const override {
                return "Tanh";
            }

            // Derivative of Tanh: 1 - tanh^2
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> tanh_val = forward(input);
                Tensor<T> result(input.shape());
                const T* tanh_data = tanh_val.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T t = tanh_data[i];
                    result_data[i] = grad_data[i] * (T(1) - t * t);
                }

                return result;
            }
        };

        // ELU Activation
        template<typename T>
        class ELUForward : public ActivationForward<T> {
        public:
            explicit ELUForward(T alpha = T(1.0)) : alpha_(alpha) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    output_data[i] = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
                }

                return output;
            }

            std::string name() const override {
                return "ELU";
            }

            // Derivative of ELU
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    result_data[i] = x > T(0) ? grad_data[i] :
                        grad_data[i] * alpha_ * std::exp(x);
                }

                return result;
            }

            // Get alpha value
            T alpha() const { return alpha_; }

            // Set alpha value
            void set_alpha(T alpha) { alpha_ = alpha; }

        private:
            T alpha_;
        };

        // SELU Activation (Scaled ELU)
        template<typename T>
        class SELUForward : public ActivationForward<T> {
        public:
            SELUForward(T alpha = T(1.6732632423543772848170429916717),
                T scale = T(1.0507009873554804934193349852946))
                : alpha_(alpha), scale_(scale) {
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    output_data[i] = scale_ * (x > T(0) ? x : alpha_ * (std::exp(x) - T(1)));
                }

                return output;
            }

            std::string name() const override {
                return "SELU";
            }

            // Derivative of SELU
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    if (x > T(0)) {
                        result_data[i] = grad_data[i] * scale_;
                    }
                    else {
                        result_data[i] = grad_data[i] * (scale_ * alpha_ * std::exp(x));
                    }
                }

                return result;
            }

            // Get alpha and scale values
            T alpha() const { return alpha_; }
            T scale() const { return scale_; }

        private:
            T alpha_;
            T scale_;
        };

        // Softplus Activation
        template<typename T>
        class SoftplusForward : public ActivationForward<T> {
        public:
            explicit SoftplusForward(T beta = T(1.0)) : beta_(beta) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = beta_ * input_data[i];

                    // Use stable computation for large values
                    if (x > T(20)) {
                        output_data[i] = input_data[i];
                    }
                    else {
                        output_data[i] = std::log(T(1) + std::exp(x)) / beta_;
                    }
                }

                return output;
            }

            std::string name() const override {
                return "Softplus";
            }

            // Derivative of Softplus: sigmoid(beta * x) * beta
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = beta_ * input_data[i];
                    T sigmoid_val = T(1) / (T(1) + std::exp(-x));
                    result_data[i] = grad_data[i] * sigmoid_val;
                }

                return result;
            }

            // Get beta value
            T beta() const { return beta_; }

            // Set beta value
            void set_beta(T beta) { beta_ = beta; }

        private:
            T beta_;
        };

        // Softsign Activation
        template<typename T>
        class SoftsignForward : public ActivationForward<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    output_data[i] = x / (T(1) + std::abs(x));
                }

                return output;
            }

            std::string name() const override {
                return "Softsign";
            }

            // Derivative of Softsign: 1 / (1 + |x|)^2
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    T denom = T(1) + std::abs(x);
                    result_data[i] = grad_data[i] / (denom * denom);
                }

                return result;
            }
        };

        // Swish Activation (x * sigmoid(x))
        template<typename T>
        class SwishForward : public ActivationForward<T> {
        public:
            explicit SwishForward(T beta = T(1.0)) : beta_(beta) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    T sigmoid_val = T(1) / (T(1) + std::exp(-beta_ * x));
                    output_data[i] = x * sigmoid_val;
                }

                return output;
            }

            std::string name() const override {
                return "Swish";
            }

            // Derivative of Swish
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    T sigmoid_val = T(1) / (T(1) + std::exp(-beta_ * x));
                    T swish_val = x * sigmoid_val;
                    result_data[i] = grad_data[i] * (swish_val + sigmoid_val * (T(1) - swish_val));
                }

                return result;
            }

            // Get beta value
            T beta() const { return beta_; }

            // Set beta value
            void set_beta(T beta) { beta_ = beta; }

        private:
            T beta_;
        };

        // Mish Activation (x * tanh(softplus(x)))
        template<typename T>
        class MishForward : public ActivationForward<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];

                    // Compute softplus with numerical stability
                    T sp;
                    if (x > T(20)) {
                        sp = x;
                    }
                    else {
                        sp = std::log(T(1) + std::exp(x));
                    }

                    output_data[i] = x * std::tanh(sp);
                }

                return output;
            }

            std::string name() const override {
                return "Mish";
            }

            // Derivative of Mish
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];

                    // Compute softplus with numerical stability
                    T sp;
                    if (x > T(20)) {
                        sp = x;
                    }
                    else {
                        sp = std::log(T(1) + std::exp(x));
                    }

                    T tanh_sp = std::tanh(sp);
                    T sech_sp = T(1) / std::cosh(sp); // sech(x) = 1/cosh(x)
                    T sigmoid_x = T(1) / (T(1) + std::exp(-x));

                    // Derivative of mish: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
                    T derivative = tanh_sp + x * sech_sp * sech_sp * sigmoid_x;

                    result_data[i] = grad_data[i] * derivative;
                }

                return result;
            }
        };

        // GELU Activation (Gaussian Error Linear Unit)
        template<typename T>
        class GELUForward : public ActivationForward<T> {
        public:
            // GELU has two approximations: 'erf' (exact) and 'tanh' (faster approximation)
            explicit GELUForward(bool use_tanh_approximation = false) :
                use_tanh_approximation_(use_tanh_approximation) {
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                if (use_tanh_approximation_) {
                    // Tanh approximation
                    const T sqrt_2_over_pi = T(0.7978845608028654);
                    const T coeff = T(0.044715);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        T cube = x * x * x;
                        T tanh_arg = sqrt_2_over_pi * (x + coeff * cube);
                        output_data[i] = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
                    }
                }
                else {
                    // Exact formula using error function
                    const T sqrt_1_over_2 = T(0.7071067811865475);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        output_data[i] = x * T(0.5) * (T(1) + std::erf(x * sqrt_1_over_2));
                    }
                }

                return output;
            }

            std::string name() const override {
                return "GELU";
            }

            // Derivative of GELU
            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) {
                Tensor<T> result(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* result_data = result.data();

                if (use_tanh_approximation_) {
                    // Derivative of tanh approximation
                    const T sqrt_2_over_pi = T(0.7978845608028654);
                    const T coeff = T(0.044715);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        T x2 = x * x;
                        T x3 = x2 * x;
                        T tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
                        T tanh_val = std::tanh(tanh_arg);

                        T derivative = T(0.5) * (T(1) + tanh_val) +
                            T(0.5) * x * (T(1) - tanh_val * tanh_val) *
                            sqrt_2_over_pi * (T(1) + T(3) * coeff * x2);

                        result_data[i] = grad_data[i] * derivative;
                    }
                }
                else {
                    // Derivative of exact GELU
                    const T sqrt_1_over_2 = T(0.7071067811865475);
                    const T sqrt_1_over_2pi = T(0.3989422804014327);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        T cdf = T(0.5) * (T(1) + std::erf(x * sqrt_1_over_2));
                        T pdf = sqrt_1_over_2pi * std::exp(-T(0.5) * x * x);

                        T derivative = cdf + x * pdf;

                        result_data[i] = grad_data[i] * derivative;
                    }
                }

                return result;
            }

            // Get approximation type
            bool uses_tanh_approximation() const { return use_tanh_approximation_; }

        private:
            bool use_tanh_approximation_;
        };

        // Factory method implementation - MUST BE AT THE END OF THE FILE 
        // after all activation classes are fully defined
        template<typename T>
        std::shared_ptr<ActivationForward<T>> ActivationForward<T>::create(const std::string& name) {
            if (name == "relu" || name == "ReLU") {
                return std::make_shared<ReLUForward<T>>();
            }
            else if (name == "sigmoid") {
                return std::make_shared<SigmoidForward<T>>();
            }
            else if (name == "tanh") {
                return std::make_shared<TanhForward<T>>();
            }
            else if (name == "leaky_relu") {
                return std::make_shared<LeakyReLUForward<T>>();
            }
            else if (name == "elu") {
                return std::make_shared<ELUForward<T>>();
            }
            else if (name == "selu") {
                return std::make_shared<SELUForward<T>>();
            }
            else if (name == "softplus") {
                return std::make_shared<SoftplusForward<T>>();
            }
            else if (name == "softsign") {
                return std::make_shared<SoftsignForward<T>>();
            }
            else if (name == "swish") {
                return std::make_shared<SwishForward<T>>();
            }
            else if (name == "mish") {
                return std::make_shared<MishForward<T>>();
            }
            else if (name == "gelu") {
                return std::make_shared<GELUForward<T>>();
            }
            else {
                throw std::invalid_argument("Unknown activation function: " + name);
            }
        }

    } // namespace forward
} // namespace tensor

#endif // ACTIVATION_FORWARD_H