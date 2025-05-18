// activation.h - v1.0.0
// C++17 standartlarýnda yazýlmýþ aktivasyon fonksiyonlarý kütüphanesi

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Tensor.h"
#include <memory>
#include <string>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace tensor {
    namespace nn {

        // Aktivasyon fonksiyonlarý için temel sýnýf
        template<typename T>
        class Activation {
        public:
            virtual ~Activation() = default;

            // Ýleri yayýlým
            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            // Geriye yayýlým (türevi)
            virtual Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) = 0;

            // Aktivasyon adý
            virtual std::string name() const = 0;

            // Factory metodu
            static std::shared_ptr<Activation<T>> create(const std::string& name);

            // Kolay kullaným için operatör aþýrý yükleme
            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }
        };

        // ReLU aktivasyonu
        template<typename T>
        class ReLU : public Activation<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    output_data[i] = std::max(T(0), input_data[i]);
                }

                return output;
            }

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    grad_input_data[i] = input_data[i] > T(0) ? grad_data[i] : T(0);
                }

                return grad_input;
            }

            std::string name() const override {
                return "ReLU";
            }
        };

        // Leaky ReLU aktivasyonu
        template<typename T>
        class LeakyReLU : public Activation<T> {
        public:
            explicit LeakyReLU(T alpha = T(0.01)) : alpha_(alpha) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    output_data[i] = x > T(0) ? x : alpha_ * x;
                }

                return output;
            }

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    grad_input_data[i] = input_data[i] > T(0) ? grad_data[i] : alpha_ * grad_data[i];
                }

                return grad_input;
            }

            std::string name() const override {
                return "LeakyReLU";
            }

            // Alpha deðerini ayarla
            void set_alpha(T alpha) { alpha_ = alpha; }

            // Alpha deðerini al
            T alpha() const { return alpha_; }

        private:
            T alpha_;
        };

        // Sigmoid aktivasyonu
        template<typename T>
        class Sigmoid : public Activation<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
                }

                return output;
            }

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> sigmoid_val = forward(input);
                Tensor<T> grad_input(input.shape());
                const T* sigmoid_data = sigmoid_val.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T s = sigmoid_data[i];
                    grad_input_data[i] = grad_data[i] * s * (T(1) - s);
                }

                return grad_input;
            }

            std::string name() const override {
                return "Sigmoid";
            }
        };

        // Tanh aktivasyonu
        template<typename T>
        class Tanh : public Activation<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    output_data[i] = std::tanh(input_data[i]);
                }

                return output;
            }

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> tanh_val = forward(input);
                Tensor<T> grad_input(input.shape());
                const T* tanh_data = tanh_val.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T t = tanh_data[i];
                    grad_input_data[i] = grad_data[i] * (T(1) - t * t);
                }

                return grad_input;
            }

            std::string name() const override {
                return "Tanh";
            }
        };

        // ELU aktivasyonu
        template<typename T>
        class ELU : public Activation<T> {
        public:
            explicit ELU(T alpha = T(1.0)) : alpha_(alpha) {}

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

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    grad_input_data[i] = x > T(0) ? grad_data[i] : grad_data[i] * alpha_ * std::exp(x);
                }

                return grad_input;
            }

            std::string name() const override {
                return "ELU";
            }

            // Alpha deðerini ayarla
            void set_alpha(T alpha) { alpha_ = alpha; }

            // Alpha deðerini al
            T alpha() const { return alpha_; }

        private:
            T alpha_;
        };

        // GELU aktivasyonu
        template<typename T>
        class GELU : public Activation<T> {
        public:
            // GELU için iki yaklaþým: 'erf' (kesin) ve 'tanh' (hýzlý yaklaþým)
            explicit GELU(bool use_tanh_approximation = false) : use_tanh_approximation_(use_tanh_approximation) {}

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                if (use_tanh_approximation_) {
                    // Tanh yaklaþýmý
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
                    // Kesin formül (error function kullanarak)
                    const T sqrt_1_over_2 = T(0.7071067811865475);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        output_data[i] = x * T(0.5) * (T(1) + std::erf(x * sqrt_1_over_2));
                    }
                }

                return output;
            }

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                if (use_tanh_approximation_) {
                    // Tanh yaklaþýmýnýn türevi
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

                        grad_input_data[i] = grad_data[i] * derivative;
                    }
                }
                else {
                    // Kesin GELU türevi
                    const T sqrt_1_over_2 = T(0.7071067811865475);
                    const T sqrt_1_over_2pi = T(0.3989422804014327);

                    for (size_t i = 0; i < input.size(); ++i) {
                        T x = input_data[i];
                        T cdf = T(0.5) * (T(1) + std::erf(x * sqrt_1_over_2));
                        T pdf = sqrt_1_over_2pi * std::exp(-T(0.5) * x * x);

                        T derivative = cdf + x * pdf;

                        grad_input_data[i] = grad_data[i] * derivative;
                    }
                }

                return grad_input;
            }

            std::string name() const override {
                return "GELU";
            }

            // Yaklaþým türünü al
            bool uses_tanh_approximation() const { return use_tanh_approximation_; }

        private:
            bool use_tanh_approximation_;
        };

        // Swish aktivasyonu (x * sigmoid(x))
        template<typename T>
        class Swish : public Activation<T> {
        public:
            explicit Swish(T beta = T(1.0)) : beta_(beta) {}

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

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];
                    T sigmoid_val = T(1) / (T(1) + std::exp(-beta_ * x));
                    T swish_val = x * sigmoid_val;

                    grad_input_data[i] = grad_data[i] * (swish_val + sigmoid_val * (T(1) - swish_val));
                }

                return grad_input;
            }

            std::string name() const override {
                return "Swish";
            }

            // Beta deðerini ayarla
            void set_beta(T beta) { beta_ = beta; }

            // Beta deðerini al
            T beta() const { return beta_; }

        private:
            T beta_;
        };

        // Mish aktivasyonu (x * tanh(softplus(x)))
        template<typename T>
        class Mish : public Activation<T> {
        public:
            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output(input.shape());
                const T* input_data = input.data();
                T* output_data = output.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];

                    // Sayýsal kararlýlýk için softplus hesapla
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

            Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& grad_output) override {
                Tensor<T> grad_input(input.shape());
                const T* input_data = input.data();
                const T* grad_data = grad_output.data();
                T* grad_input_data = grad_input.data();

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input_data[i];

                    // Sayýsal kararlýlýk için softplus hesapla
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

                    // Mish türevi
                    T derivative = tanh_sp + x * sech_sp * sech_sp * sigmoid_x;

                    grad_input_data[i] = grad_data[i] * derivative;
                }

                return grad_input;
            }

            std::string name() const override {
                return "Mish";
            }
        };

        // Factory metodu implementasyonu
        template<typename T>
        std::shared_ptr<Activation<T>> Activation<T>::create(const std::string& name) {
            if (name == "relu" || name == "ReLU") {
                return std::make_shared<ReLU<T>>();
            }
            else if (name == "leaky_relu" || name == "LeakyReLU") {
                return std::make_shared<LeakyReLU<T>>();
            }
            else if (name == "sigmoid" || name == "Sigmoid") {
                return std::make_shared<Sigmoid<T>>();
            }
            else if (name == "tanh" || name == "Tanh") {
                return std::make_shared<Tanh<T>>();
            }
            else if (name == "elu" || name == "ELU") {
                return std::make_shared<ELU<T>>();
            }
            else if (name == "gelu" || name == "GELU") {
                return std::make_shared<GELU<T>>();
            }
            else if (name == "swish" || name == "Swish") {
                return std::make_shared<Swish<T>>();
            }
            else if (name == "mish" || name == "Mish") {
                return std::make_shared<Mish<T>>();
            }
            else {
                throw std::invalid_argument("Bilinmeyen aktivasyon fonksiyonu: " + name);
            }
        }

    } // namespace nn
} // namespace tensor

#endif // ACTIVATION_H