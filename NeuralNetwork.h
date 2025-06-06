// NeuralNetwork.h
// v1.0.0 - C++17 uyumlu sinir a�� mod�lleri

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Tensor.h"
#include "ForwardModule.h"
#include "ActivationForward.h"
#include "LinearForward.h"
#include "Sequential.h"
#include "TensorGrad.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>

namespace tensor {
    namespace nn {

        // T�m sinir a�� mod�lleri i�in temel s�n�f
        template<typename T>
        class Module {
        public:
            virtual ~Module() = default;

            // �leri yay�l�m
            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            // Parametreler listesi
            virtual std::vector<Tensor<T>*> parameters() = 0;

            // Geriye yay�l�m
            virtual void backward(const Tensor<T>& grad) {
                // Alt s�n�flar taraf�ndan uygulanacak
                (void)grad; // Kullan�lmayan parametre uyar�s� �nleme
            }

            // E�itim modu ayarlama
            virtual void train(bool mode = true) {
                training_ = mode;
                for (auto& module : submodules_) {
                    module->train(mode);
                }
            }

            // De�erlendirme moduna ge�
            virtual void eval() {
                train(false);
            }

            // E�itim modunda m�?
            bool is_training() const {
                return training_;
            }

            // Alt mod�l ekleme
            void add_submodule(std::shared_ptr<Module<T>> module) {
                submodules_.push_back(module);
            }

            // Kolay kullan�m i�in operat�r
            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }

        protected:
            bool training_ = true;
            std::vector<std::shared_ptr<Module<T>>> submodules_;
        };

        // Do�rusal katman s�n�f� - LinearForward'� kullanarak 
        template<typename T>
        class Linear : public Module<T> {
        public:
            Linear(size_t in_features, size_t out_features, bool bias = true)
                : in_features_(in_features), out_features_(out_features) {

                linear_impl_ = std::make_shared<forward::LinearForward<T>>(in_features, out_features, bias);
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                last_input_ = input;
                return linear_impl_->forward(input);
            }

            void backward(const Tensor<T>& grad) override {
                // Gradient hesaplama - TensorGrad y�ntemlerini kullanarak
                // Bu k�sm� TensorGrad yap�n�za g�re uyarlaman�z gerekecek
                auto& weights = linear_impl_->weights();

                // Ger�ek bir backward hesaplamas� i�in autograd'� 
                // TensorGrad ve autograd yap�n�za g�re uyarlay�n
            }

            std::vector<Tensor<T>*> parameters() override {
                return linear_impl_->parameters();
            }

            size_t in_features() const { return in_features_; }
            size_t out_features() const { return out_features_; }

        private:
            size_t in_features_;
            size_t out_features_;
            std::shared_ptr<forward::LinearForward<T>> linear_impl_;
            Tensor<T> last_input_;
        };

        // ReLU aktivasyon s�n�f�
        template<typename T>
        class ReLU : public Module<T> {
        public:
            ReLU() {
                relu_impl_ = std::make_shared<forward::ReLUForward<T>>();
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                last_input_ = input;
                return relu_impl_->forward(input);
            }

            void backward(const Tensor<T>& grad) override {
                // Gradient hesaplama - relu_impl_'nin backward metodunu kullanarak
            }

            std::vector<Tensor<T>*> parameters() override {
                return {};
            }

        private:
            std::shared_ptr<forward::ReLUForward<T>> relu_impl_;
            Tensor<T> last_input_;
        };

        // Benzer �ekilde Sigmoid, Tanh ve di�er aktivasyonlar i�in de s�n�flar eklenebilir

        // S�ral� konteyner - Sequential'� kullanarak
        template<typename T>
        class Sequential : public Module<T> {
        public:
            Sequential() = default;

            void add(std::shared_ptr<Module<T>> module) {
                modules_.push_back(module);
                this->add_submodule(module);
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output = input;
                for (auto& module : modules_) {
                    output = module->forward(output);
                }
                return output;
            }

            void backward(const Tensor<T>& grad) override {
                if (modules_.empty()) return;

                Tensor<T> current_grad = grad;
                for (int i = modules_.size() - 1; i >= 0; --i) {
                    modules_[i]->backward(current_grad);
                    // Burada gradient iletimi eksik, her mod�l kendi input
                    // gradientini hesaplamal� ve bir �nceki mod�le iletmeli
                }
            }

            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;
                for (auto& module : modules_) {
                    auto module_params = module->parameters();
                    params.insert(params.end(), module_params.begin(), module_params.end());
                }
                return params;
            }

            // Model �zeti yazd�rma
            void summary(std::ostream& os = std::cout) const {
                os << "Model �zeti:" << std::endl;
                os << "--------------------------------" << std::endl;

                size_t total_params = 0;
                for (size_t i = 0; i < modules_.size(); ++i) {
                    auto& module = modules_[i];
                    auto params = module->parameters();

                    size_t layer_params = 0;
                    for (auto* param : params) {
                        layer_params += param->size();
                    }
                    total_params += layer_params;

                    os << "Katman " << i + 1 << ": " << typeid(*module).name()
                        << ", Parametreler: " << layer_params << std::endl;
                }

                os << "--------------------------------" << std::endl;
                os << "Toplam parametreler: " << total_params << std::endl;
                os << "--------------------------------" << std::endl;
            }

        private:
            std::vector<std::shared_ptr<Module<T>>> modules_;
        };

        // Basit XOR sinir a�� �rne�i - opsiyonel
        template<typename T>
        class XORNetwork : public Module<T> {
        public:
            XORNetwork() {
                linear1_ = std::make_shared<Linear<T>>(2, 4);
                relu_ = std::make_shared<ReLU<T>>();
                linear2_ = std::make_shared<Linear<T>>(4, 1);

                this->add_submodule(linear1_);
                this->add_submodule(relu_);
                this->add_submodule(linear2_);
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                auto x = linear1_->forward(input);
                x = relu_->forward(x);
                x = linear2_->forward(x);
                return x;
            }

            void backward(const Tensor<T>& grad) override {
                linear2_->backward(grad);
                relu_->backward(grad);
                linear1_->backward(grad);
            }

            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;

                auto linear1_params = linear1_->parameters();
                params.insert(params.end(), linear1_params.begin(), linear1_params.end());

                auto linear2_params = linear2_->parameters();
                params.insert(params.end(), linear2_params.begin(), linear2_params.end());

                return params;
            }

        private:
            std::shared_ptr<Linear<T>> linear1_;
            std::shared_ptr<ReLU<T>> relu_;
            std::shared_ptr<Linear<T>> linear2_;
        };

    } // namespace nn
} // namespace tensor

#endif // NEURAL_NETWORK_H