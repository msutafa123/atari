// TensorLayers.h - v1.0.0
// C++17 standartlarýnda geliþmiþ yapay sinir aðý katmanlarý

#ifndef TENSOR_LAYERS_H
#define TENSOR_LAYERS_H

#include "Tensor.h"
#include "ForwardModule.h"
#include "TensorMath.h"
#include "TensorOps.h"
#include "TensorReshape.h"

#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <optional>
#include <functional>
#include <iostream>
#include <thread>

namespace tensor {
    namespace nn {

        // Paralel hesaplama yardýmcýsý
        class ParallelCompute {
        public:
            // Maksimum kullanýlabilir iþ parçacýðý sayýsýný al
            static size_t get_max_threads() {
                return std::thread::hardware_concurrency();
            }

            // Paralel for döngüsü
            template<typename Func>
            static void parallel_for(size_t begin, size_t end, Func func, size_t num_threads = 0) {
                if (num_threads == 0) {
                    num_threads = get_max_threads();
                }

                // Tek çekirdekli veya küçük veri durumunu optimize et
                if (num_threads <= 1 || end - begin <= 1000) {
                    for (size_t i = begin; i < end; ++i) {
                        func(i);
                    }
                    return;
                }

                std::vector<std::thread> threads;
                threads.reserve(num_threads);

                // Ýþ parçacýðý baþýna iþlenecek eleman sayýsý
                size_t chunk_size = (end - begin) / num_threads;

                // Her iþ parçacýðýný baþlat
                for (size_t t = 0; t < num_threads; ++t) {
                    size_t chunk_begin = begin + t * chunk_size;
                    size_t chunk_end = (t == num_threads - 1) ? end : chunk_begin + chunk_size;

                    threads.emplace_back([chunk_begin, chunk_end, &func]() {
                        for (size_t i = chunk_begin; i < chunk_end; ++i) {
                            func(i);
                        }
                        });
                }

                // Tüm iþ parçacýklarýnýn tamamlanmasýný bekle
                for (auto& thread : threads) {
                    thread.join();
                }
            }
        };

        // -------------------- KATMAN SINIFI --------------------
        // Temel modül arayüzü
        template<typename T = float>
        class Module {
        public:
            virtual ~Module() = default;

            // Ýleri yayýlým
            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            // Kolay kullaným için parantez operatörü
            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }

            // Parametreleri döndür
            virtual std::vector<Tensor<T>*> parameters() = 0;

            // Toplam parametre sayýsýný hesapla
            size_t parameter_count() const {
                size_t count = 0;
                for (auto param : const_cast<Module<T>*>(this)->parameters()) {
                    count += param->size();
                }
                return count;
            }

            // Eðitim modunu ayarla
            virtual void train(bool is_training = true) {
                training_ = is_training;
            }

            // Deðerlendirme moduna geç
            virtual void eval() {
                train(false);
            }

            // Eðitim modunda mý?
            bool is_training() const {
                return training_;
            }

        protected:
            bool training_ = true;
        };

        // AI Modeli sýnýfý (önceden eðitilmiþ modeller için)
        template<typename T = float>
        class AIModel : public Module<T> {
        public:
            // Model adýný al
            virtual std::string name() const {
                return "GenericAIModel";
            }

            // Modeli kaydet
            virtual bool save(const std::string& filename) const {
                // Gerçek uygulamada parametre serileþtirme kodu buraya gelebilir
                std::cout << "Model kaydediliyor: " << filename << std::endl;
                return true;
            }

            // Modeli yükle
            virtual bool load(const std::string& filename) {
                // Gerçek uygulamada parametre serileþtirme kodu buraya gelebilir
                std::cout << "Model yükleniyor: " << filename << std::endl;
                return true;
            }
        };

        // -------------------- KATMAN SARMALAYICILARI --------------------

        // Geliþmiþ doðrusal katman
        template<typename T = float>
        class EnhancedLinear : public forward::ForwardModule<T> {
        public:
            EnhancedLinear(size_t in_features, size_t out_features, bool bias = true)
                : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
                // He baþlatma
                T stddev = std::sqrt(T(2) / in_features);

                // Aðýrlýklar matrisini oluþtur
                std::vector<size_t> weight_dims = { in_features, out_features };
                weights_ = Tensor<T>(weight_dims);

                // Rastgele baþlatma
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> dist(0.0f, stddev);

                for (size_t i = 0; i < weights_.size(); ++i) {
                    weights_.data()[i] = static_cast<T>(dist(gen));
                }

                if (use_bias_) {
                    std::vector<size_t> bias_dims = { out_features };
                    bias_ = Tensor<T>(bias_dims);
                    bias_.fill(T(0));
                }
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                size_t last_dim = input.ndim() - 1;
                if (input.dim(last_dim) != in_features_) {
                    throw std::runtime_error("Giriþ boyutu hatalý: " +
                        std::to_string(input.dim(last_dim)) + " != " +
                        std::to_string(in_features_));
                }

                // Matris çarpýmý yap
                auto output = matmul(input, weights_);

                if (use_bias_) {
                    // Bias ekleme - broadcasting ile
                    for (size_t i = 0; i < output.size(); ++i) {
                        size_t bias_idx = i % out_features_;
                        output.data()[i] += bias_.data()[bias_idx];
                    }
                }

                return output;
            }

            std::vector<Tensor<T>*> parameters() override {
                if (use_bias_) {
                    return { &weights_, &bias_ };
                }
                else {
                    return { &weights_ };
                }
            }

            std::string name() const override {
                return "EnhancedLinear(" + std::to_string(in_features_) + ", " +
                    std::to_string(out_features_) + ")";
            }

            size_t in_features() const { return in_features_; }
            size_t out_features() const { return out_features_; }
            bool has_bias() const { return use_bias_; }

            // Aðýrlýklarý al
            Tensor<T>& weights() { return weights_; }
            const Tensor<T>& weights() const { return weights_; }

            // Bias al
            Tensor<T>& bias() {
                if (!use_bias_) {
                    throw std::logic_error("Doðrusal katmanda bias kullanýlmýyor");
                }
                return bias_;
            }

            const Tensor<T>& bias() const {
                if (!use_bias_) {
                    throw std::logic_error("Doðrusal katmanda bias kullanýlmýyor");
                }
                return bias_;
            }

        private:
            size_t in_features_;
            size_t out_features_;
            bool use_bias_;
            Tensor<T> weights_;
            Tensor<T> bias_;
        };

        // Geliþmiþ aktivasyon katmanlarý
        template<typename T = float>
        class EnhancedActivation : public forward::ForwardModule<T> {
        public:
            enum class ActivationType {
                ReLU,
                Sigmoid,
                Tanh,
                LeakyReLU,
                GELU
            };

            EnhancedActivation(ActivationType type = ActivationType::ReLU, T alpha = 0.01)
                : type_(type), alpha_(alpha) {
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                switch (type_) {
                case ActivationType::ReLU:
                    return tensor::math::relu(input);
                case ActivationType::Sigmoid:
                    return tensor::math::sigmoid(input);
                case ActivationType::Tanh:
                    return tensor::math::tanh(input);
                case ActivationType::LeakyReLU:
                    return leaky_relu(input);
                case ActivationType::GELU:
                    return gelu(input);
                default:
                    return input;
                }
            }

            std::vector<Tensor<T>*> parameters() override {
                return {};
            }

            std::string name() const override {
                switch (type_) {
                case ActivationType::ReLU: return "ReLU";
                case ActivationType::Sigmoid: return "Sigmoid";
                case ActivationType::Tanh: return "Tanh";
                case ActivationType::LeakyReLU: return "LeakyReLU(" + std::to_string(alpha_) + ")";
                case ActivationType::GELU: return "GELU";
                default: return "Unknown";
                }
            }

        private:
            ActivationType type_;
            T alpha_; // LeakyReLU için eðim parametresi

            // LeakyReLU uygulamasý
            Tensor<T> leaky_relu(const Tensor<T>& input) {
                return tensor::math::leaky_relu(input, alpha_);
            }

            // GELU uygulamasý (yaklaþýk)
            Tensor<T> gelu(const Tensor<T>& input) {
                Tensor<T> result(input.shape());
                const T sqrt_2_over_pi = std::sqrt(T(2) / T(3.14159265358979323846));

                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input.data()[i];
                    T cube = x * x * x;
                    T tanh_arg = sqrt_2_over_pi * (x + T(0.044715) * cube);
                    result.data()[i] = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
                }
                return result;
            }
        };

        // -------------------- EVRIÞIM KATMANLARI --------------------

        // Evriþimli katman
        template<typename T = float>
        class Conv2d : public forward::ForwardModule<T> {
        public:
            Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride = 1, size_t padding = 0, bool bias = true)
                : in_channels_(in_channels), out_channels_(out_channels),
                kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(bias) {

                // Çekirdek aðýrlýklarýný baþlat
                std::vector<size_t> weight_dims = { out_channels, in_channels, kernel_size, kernel_size };
                weights_ = Tensor<T>(weight_dims);

                // He baþlatma ile rastgele aðýrlýklar
                T stddev = std::sqrt(T(2) / (in_channels * kernel_size * kernel_size));
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> dist(0.0f, stddev);

                for (size_t i = 0; i < weights_.size(); ++i) {
                    weights_.data()[i] = static_cast<T>(dist(gen));
                }

                if (use_bias_) {
                    std::vector<size_t> bias_dims = { out_channels };
                    bias_ = Tensor<T>(bias_dims);
                    bias_.fill(T(0));
                }
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                if (input.ndim() != 4) {
                    throw std::runtime_error("Conv2d katmaný 4B tensör bekliyor [batch, channels, height, width]");
                }

                size_t batch_size = input.dim(0);
                size_t height = input.dim(2);
                size_t width = input.dim(3);

                // Çýkýþ boyutlarýný hesapla
                size_t out_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
                size_t out_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;

                // Çýkýþ tensörü oluþtur
                std::vector<size_t> output_dims = { batch_size, out_channels_, out_height, out_width };
                Tensor<T> output(output_dims);
                output.fill(T(0));

                // Evriþim hesaplama
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t c_out = 0; c_out < out_channels_; ++c_out) {
                        for (size_t h_out = 0; h_out < out_height; ++h_out) {
                            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                T sum = 0;

                                // Çekirdek kaydýrma
                                for (size_t c_in = 0; c_in < in_channels_; ++c_in) {
                                    for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                        for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                            int h_in = static_cast<int>(h_out * stride_ + kh - padding_);
                                            int w_in = static_cast<int>(w_out * stride_ + kw - padding_);

                                            // Sýnýrý kontrol et
                                            if (h_in >= 0 && h_in < static_cast<int>(height) &&
                                                w_in >= 0 && w_in < static_cast<int>(width)) {
                                                sum += input.at({ b, c_in, (size_t)h_in, (size_t)w_in }) *
                                                    weights_.at({ c_out, c_in, kh, kw });
                                            }
                                        }
                                    }
                                }

                                // Bias ekle
                                if (use_bias_) {
                                    sum += bias_.at({ c_out });
                                }

                                output.at({ b, c_out, h_out, w_out }) = sum;
                            }
                        }
                    }
                }

                return output;
            }

            std::vector<Tensor<T>*> parameters() override {
                if (use_bias_) {
                    return { &weights_, &bias_ };
                }
                else {
                    return { &weights_ };
                }
            }

            std::string name() const override {
                return "Conv2d(" + std::to_string(in_channels_) + ", " +
                    std::to_string(out_channels_) + ", kernel_size=" +
                    std::to_string(kernel_size_) + ")";
            }

        private:
            size_t in_channels_;
            size_t out_channels_;
            size_t kernel_size_;
            size_t stride_;
            size_t padding_;
            bool use_bias_;
            Tensor<T> weights_;
            Tensor<T> bias_;
        };

        // Havuzlama katmaný (maksimum havuzlama)
        template<typename T = float>
        class MaxPool2d : public forward::ForwardModule<T> {
        public:
            MaxPool2d(size_t kernel_size, size_t stride = 0)
                : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride) {
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                if (input.ndim() != 4) {
                    throw std::runtime_error("MaxPool2d katmaný 4B tensör bekliyor [batch, channels, height, width]");
                }

                size_t batch_size = input.dim(0);
                size_t channels = input.dim(1);
                size_t height = input.dim(2);
                size_t width = input.dim(3);

                // Çýkýþ boyutlarýný hesapla
                size_t out_height = (height - kernel_size_) / stride_ + 1;
                size_t out_width = (width - kernel_size_) / stride_ + 1;

                // Çýkýþ tensörü oluþtur
                std::vector<size_t> output_dims = { batch_size, channels, out_height, out_width };
                Tensor<T> output(output_dims);

                // Maksimum havuzlama hesaplama
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t c = 0; c < channels; ++c) {
                        for (size_t h_out = 0; h_out < out_height; ++h_out) {
                            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                T max_val = std::numeric_limits<T>::lowest();

                                // Havuzlama penceresi
                                for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                    for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                        size_t h_in = h_out * stride_ + kh;
                                        size_t w_in = w_out * stride_ + kw;

                                        max_val = std::max(max_val, input.at({ b, c, h_in, w_in }));
                                    }
                                }

                                output.at({ b, c, h_out, w_out }) = max_val;
                            }
                        }
                    }
                }

                return output;
            }

            std::vector<Tensor<T>*> parameters() override {
                return {};
            }

            std::string name() const override {
                return "MaxPool2d(" + std::to_string(kernel_size_) + ", " +
                    std::to_string(stride_) + ")";
            }

        private:
            size_t kernel_size_;
            size_t stride_;
        };

        // -------------------- OPTÝMÝZASYON SINIFI --------------------

        // Temel optimize edici arayüzü
        template<typename T = float>
        class Optimizer {
        public:
            Optimizer(const std::vector<Tensor<T>*>& parameters, T learning_rate = T(0.01))
                : parameters_(parameters), learning_rate_(learning_rate) {
            }

            virtual ~Optimizer() = default;

            // Adým gerçekleþtir (parametreleri güncelle)
            virtual void step() = 0;

            // Gradyanlarý sýfýrla
            virtual void zero_grad() {
                for (auto& param : parameters_) {
                    // Gerçek implementasyonda gradyanlar sýfýrlanýr
                    // Bu örnek implementasyonda sadece placeholder
                }
            }

            // Öðrenme oranýný ayarla
            virtual void set_learning_rate(T lr) {
                learning_rate_ = lr;
            }

            // Öðrenme oranýný al
            T get_learning_rate() const { return learning_rate_; }

        protected:
            std::vector<Tensor<T>*> parameters_;
            T learning_rate_;
        };

        // Adam optimize edici (geliþtirilmiþ)
        template<typename T = float>
        class EnhancedAdam : public Optimizer<T> {
        public:
            EnhancedAdam(const std::vector<Tensor<T>*>& parameters,
                T learning_rate = T(0.001),
                T beta1 = T(0.9),
                T beta2 = T(0.999),
                T epsilon = T(1e-8),
                T weight_decay = T(0))
                : Optimizer<T>(parameters, learning_rate),
                beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
                weight_decay_(weight_decay), t_(0) {

                // Moment vektörlerini baþlat
                for (const auto& param : this->parameters_) {
                    m_.push_back(Tensor<T>::zeros(param->shape()));
                    v_.push_back(Tensor<T>::zeros(param->shape()));
                }
            }

            void step() override {
                t_++;

                // Paralel hesaplama için CPU çekirdek sayýsý
                size_t num_threads = ParallelCompute::get_max_threads();

                for (size_t i = 0; i < this->parameters_.size(); ++i) {
                    auto& param = *(this->parameters_[i]);
                    auto& m = m_[i];
                    auto& v = v_[i];

                    // Aðýrlýk çürütme uygula
                    if (weight_decay_ > 0) {
                        for (size_t j = 0; j < param.size(); ++j) {
                            // Gerçek implementasyonda gradyanlar burada güncellenir
                        }
                    }

                    // Moment güncellemeleri için paralel hesaplama
                    ParallelCompute::parallel_for(0, param.size(), [&](size_t j) {
                        // Yanlý ilk moment tahmini güncelle
                        m.data()[j] = beta1_ * m.data()[j] + (T(1) - beta1_) * param.grad().data()[j];

                        // Yanlý ikinci ham moment tahmini güncelle
                        v.data()[j] = beta2_ * v.data()[j] + (T(1) - beta2_) *
                            param.grad().data()[j] * param.grad().data()[j];

                        // Bias-düzeltilmiþ momentler
                        T m_hat = m.data()[j] / (T(1) - std::pow(beta1_, t_));
                        T v_hat = v.data()[j] / (T(1) - std::pow(beta2_, t_));

                        // Parametreleri güncelle
                        param.data()[j] -= this->learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                        }, num_threads);
                }
            }

        private:
            T beta1_;
            T beta2_;
            T epsilon_;
            T weight_decay_;
            size_t t_;  // Zaman adýmý
            std::vector<Tensor<T>> m_;  // Ýlk moment tahminleri
            std::vector<Tensor<T>> v_;  // Ýkinci moment tahminleri
        };

        // -------------------- ÖNCEDEN EÐÝTÝLMÝÞ MODEL DEPOSU --------------------

        // Önceden eðitilmiþ model yükleyici
        template<typename T = float>
        class PretrainedModelRepository {
        public:
            // Kullanýlabilir model türleri
            enum class ModelType {
                MLP,
                SimpleResNet,
                MobileNet
            };

            // Önceden eðitilmiþ model yükle
            static std::shared_ptr<AIModel<T>> load_model(ModelType type, bool pretrained = true) {
                switch (type) {
                case ModelType::MLP:
                    return create_mlp(pretrained);
                case ModelType::SimpleResNet:
                    return create_simple_resnet(pretrained);
                case ModelType::MobileNet:
                    return create_simple_mobilenet(pretrained);
                default:
                    throw std::runtime_error("Bilinmeyen model türü");
                }
            }

        private:
            // Model oluþturma metotlarý...
            // Örnek olarak placeholder metotlarý ekledim, gerçek implementasyonlarda
            // bu modellerin tam tanýmlarý bulunmalýdýr.

            static std::shared_ptr<AIModel<T>> create_mlp(bool pretrained) {
                // Gerçek implementasyon burada olacak
                // Þimdilik sadece nullptr döndürüyorum
                return nullptr;
            }

            static std::shared_ptr<AIModel<T>> create_simple_resnet(bool pretrained) {
                // Gerçek implementasyon burada olacak
                return nullptr;
            }

            static std::shared_ptr<AIModel<T>> create_simple_mobilenet(bool pretrained) {
                // Gerçek implementasyon burada olacak
                return nullptr;
            }
        };

    } // namespace nn
} // namespace tensor

#endif // TENSOR_LAYERS_H