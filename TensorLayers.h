// TensorLayers.h - v1.0.0
// C++17 standartlar�nda geli�mi� yapay sinir a�� katmanlar�

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

        // Paralel hesaplama yard�mc�s�
        class ParallelCompute {
        public:
            // Maksimum kullan�labilir i� par�ac��� say�s�n� al
            static size_t get_max_threads() {
                return std::thread::hardware_concurrency();
            }

            // Paralel for d�ng�s�
            template<typename Func>
            static void parallel_for(size_t begin, size_t end, Func func, size_t num_threads = 0) {
                if (num_threads == 0) {
                    num_threads = get_max_threads();
                }

                // Tek �ekirdekli veya k���k veri durumunu optimize et
                if (num_threads <= 1 || end - begin <= 1000) {
                    for (size_t i = begin; i < end; ++i) {
                        func(i);
                    }
                    return;
                }

                std::vector<std::thread> threads;
                threads.reserve(num_threads);

                // �� par�ac��� ba��na i�lenecek eleman say�s�
                size_t chunk_size = (end - begin) / num_threads;

                // Her i� par�ac���n� ba�lat
                for (size_t t = 0; t < num_threads; ++t) {
                    size_t chunk_begin = begin + t * chunk_size;
                    size_t chunk_end = (t == num_threads - 1) ? end : chunk_begin + chunk_size;

                    threads.emplace_back([chunk_begin, chunk_end, &func]() {
                        for (size_t i = chunk_begin; i < chunk_end; ++i) {
                            func(i);
                        }
                        });
                }

                // T�m i� par�ac�klar�n�n tamamlanmas�n� bekle
                for (auto& thread : threads) {
                    thread.join();
                }
            }
        };

        // -------------------- KATMAN SINIFI --------------------
        // Temel mod�l aray�z�
        template<typename T = float>
        class Module {
        public:
            virtual ~Module() = default;

            // �leri yay�l�m
            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            // Kolay kullan�m i�in parantez operat�r�
            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }

            // Parametreleri d�nd�r
            virtual std::vector<Tensor<T>*> parameters() = 0;

            // Toplam parametre say�s�n� hesapla
            size_t parameter_count() const {
                size_t count = 0;
                for (auto param : const_cast<Module<T>*>(this)->parameters()) {
                    count += param->size();
                }
                return count;
            }

            // E�itim modunu ayarla
            virtual void train(bool is_training = true) {
                training_ = is_training;
            }

            // De�erlendirme moduna ge�
            virtual void eval() {
                train(false);
            }

            // E�itim modunda m�?
            bool is_training() const {
                return training_;
            }

        protected:
            bool training_ = true;
        };

        // AI Modeli s�n�f� (�nceden e�itilmi� modeller i�in)
        template<typename T = float>
        class AIModel : public Module<T> {
        public:
            // Model ad�n� al
            virtual std::string name() const {
                return "GenericAIModel";
            }

            // Modeli kaydet
            virtual bool save(const std::string& filename) const {
                // Ger�ek uygulamada parametre serile�tirme kodu buraya gelebilir
                std::cout << "Model kaydediliyor: " << filename << std::endl;
                return true;
            }

            // Modeli y�kle
            virtual bool load(const std::string& filename) {
                // Ger�ek uygulamada parametre serile�tirme kodu buraya gelebilir
                std::cout << "Model y�kleniyor: " << filename << std::endl;
                return true;
            }
        };

        // -------------------- KATMAN SARMALAYICILARI --------------------

        // Geli�mi� do�rusal katman
        template<typename T = float>
        class EnhancedLinear : public forward::ForwardModule<T> {
        public:
            EnhancedLinear(size_t in_features, size_t out_features, bool bias = true)
                : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
                // He ba�latma
                T stddev = std::sqrt(T(2) / in_features);

                // A��rl�klar matrisini olu�tur
                std::vector<size_t> weight_dims = { in_features, out_features };
                weights_ = Tensor<T>(weight_dims);

                // Rastgele ba�latma
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
                    throw std::runtime_error("Giri� boyutu hatal�: " +
                        std::to_string(input.dim(last_dim)) + " != " +
                        std::to_string(in_features_));
                }

                // Matris �arp�m� yap
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

            // A��rl�klar� al
            Tensor<T>& weights() { return weights_; }
            const Tensor<T>& weights() const { return weights_; }

            // Bias al
            Tensor<T>& bias() {
                if (!use_bias_) {
                    throw std::logic_error("Do�rusal katmanda bias kullan�lm�yor");
                }
                return bias_;
            }

            const Tensor<T>& bias() const {
                if (!use_bias_) {
                    throw std::logic_error("Do�rusal katmanda bias kullan�lm�yor");
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

        // Geli�mi� aktivasyon katmanlar�
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
            T alpha_; // LeakyReLU i�in e�im parametresi

            // LeakyReLU uygulamas�
            Tensor<T> leaky_relu(const Tensor<T>& input) {
                return tensor::math::leaky_relu(input, alpha_);
            }

            // GELU uygulamas� (yakla��k)
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

        // -------------------- EVRI�IM KATMANLARI --------------------

        // Evri�imli katman
        template<typename T = float>
        class Conv2d : public forward::ForwardModule<T> {
        public:
            Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride = 1, size_t padding = 0, bool bias = true)
                : in_channels_(in_channels), out_channels_(out_channels),
                kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(bias) {

                // �ekirdek a��rl�klar�n� ba�lat
                std::vector<size_t> weight_dims = { out_channels, in_channels, kernel_size, kernel_size };
                weights_ = Tensor<T>(weight_dims);

                // He ba�latma ile rastgele a��rl�klar
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
                    throw std::runtime_error("Conv2d katman� 4B tens�r bekliyor [batch, channels, height, width]");
                }

                size_t batch_size = input.dim(0);
                size_t height = input.dim(2);
                size_t width = input.dim(3);

                // ��k�� boyutlar�n� hesapla
                size_t out_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
                size_t out_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;

                // ��k�� tens�r� olu�tur
                std::vector<size_t> output_dims = { batch_size, out_channels_, out_height, out_width };
                Tensor<T> output(output_dims);
                output.fill(T(0));

                // Evri�im hesaplama
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t c_out = 0; c_out < out_channels_; ++c_out) {
                        for (size_t h_out = 0; h_out < out_height; ++h_out) {
                            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                T sum = 0;

                                // �ekirdek kayd�rma
                                for (size_t c_in = 0; c_in < in_channels_; ++c_in) {
                                    for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                        for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                            int h_in = static_cast<int>(h_out * stride_ + kh - padding_);
                                            int w_in = static_cast<int>(w_out * stride_ + kw - padding_);

                                            // S�n�r� kontrol et
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

        // Havuzlama katman� (maksimum havuzlama)
        template<typename T = float>
        class MaxPool2d : public forward::ForwardModule<T> {
        public:
            MaxPool2d(size_t kernel_size, size_t stride = 0)
                : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride) {
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                if (input.ndim() != 4) {
                    throw std::runtime_error("MaxPool2d katman� 4B tens�r bekliyor [batch, channels, height, width]");
                }

                size_t batch_size = input.dim(0);
                size_t channels = input.dim(1);
                size_t height = input.dim(2);
                size_t width = input.dim(3);

                // ��k�� boyutlar�n� hesapla
                size_t out_height = (height - kernel_size_) / stride_ + 1;
                size_t out_width = (width - kernel_size_) / stride_ + 1;

                // ��k�� tens�r� olu�tur
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

        // -------------------- OPT�M�ZASYON SINIFI --------------------

        // Temel optimize edici aray�z�
        template<typename T = float>
        class Optimizer {
        public:
            Optimizer(const std::vector<Tensor<T>*>& parameters, T learning_rate = T(0.01))
                : parameters_(parameters), learning_rate_(learning_rate) {
            }

            virtual ~Optimizer() = default;

            // Ad�m ger�ekle�tir (parametreleri g�ncelle)
            virtual void step() = 0;

            // Gradyanlar� s�f�rla
            virtual void zero_grad() {
                for (auto& param : parameters_) {
                    // Ger�ek implementasyonda gradyanlar s�f�rlan�r
                    // Bu �rnek implementasyonda sadece placeholder
                }
            }

            // ��renme oran�n� ayarla
            virtual void set_learning_rate(T lr) {
                learning_rate_ = lr;
            }

            // ��renme oran�n� al
            T get_learning_rate() const { return learning_rate_; }

        protected:
            std::vector<Tensor<T>*> parameters_;
            T learning_rate_;
        };

        // Adam optimize edici (geli�tirilmi�)
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

                // Moment vekt�rlerini ba�lat
                for (const auto& param : this->parameters_) {
                    m_.push_back(Tensor<T>::zeros(param->shape()));
                    v_.push_back(Tensor<T>::zeros(param->shape()));
                }
            }

            void step() override {
                t_++;

                // Paralel hesaplama i�in CPU �ekirdek say�s�
                size_t num_threads = ParallelCompute::get_max_threads();

                for (size_t i = 0; i < this->parameters_.size(); ++i) {
                    auto& param = *(this->parameters_[i]);
                    auto& m = m_[i];
                    auto& v = v_[i];

                    // A��rl�k ��r�tme uygula
                    if (weight_decay_ > 0) {
                        for (size_t j = 0; j < param.size(); ++j) {
                            // Ger�ek implementasyonda gradyanlar burada g�ncellenir
                        }
                    }

                    // Moment g�ncellemeleri i�in paralel hesaplama
                    ParallelCompute::parallel_for(0, param.size(), [&](size_t j) {
                        // Yanl� ilk moment tahmini g�ncelle
                        m.data()[j] = beta1_ * m.data()[j] + (T(1) - beta1_) * param.grad().data()[j];

                        // Yanl� ikinci ham moment tahmini g�ncelle
                        v.data()[j] = beta2_ * v.data()[j] + (T(1) - beta2_) *
                            param.grad().data()[j] * param.grad().data()[j];

                        // Bias-d�zeltilmi� momentler
                        T m_hat = m.data()[j] / (T(1) - std::pow(beta1_, t_));
                        T v_hat = v.data()[j] / (T(1) - std::pow(beta2_, t_));

                        // Parametreleri g�ncelle
                        param.data()[j] -= this->learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                        }, num_threads);
                }
            }

        private:
            T beta1_;
            T beta2_;
            T epsilon_;
            T weight_decay_;
            size_t t_;  // Zaman ad�m�
            std::vector<Tensor<T>> m_;  // �lk moment tahminleri
            std::vector<Tensor<T>> v_;  // �kinci moment tahminleri
        };

        // -------------------- �NCEDEN E��T�LM�� MODEL DEPOSU --------------------

        // �nceden e�itilmi� model y�kleyici
        template<typename T = float>
        class PretrainedModelRepository {
        public:
            // Kullan�labilir model t�rleri
            enum class ModelType {
                MLP,
                SimpleResNet,
                MobileNet
            };

            // �nceden e�itilmi� model y�kle
            static std::shared_ptr<AIModel<T>> load_model(ModelType type, bool pretrained = true) {
                switch (type) {
                case ModelType::MLP:
                    return create_mlp(pretrained);
                case ModelType::SimpleResNet:
                    return create_simple_resnet(pretrained);
                case ModelType::MobileNet:
                    return create_simple_mobilenet(pretrained);
                default:
                    throw std::runtime_error("Bilinmeyen model t�r�");
                }
            }

        private:
            // Model olu�turma metotlar�...
            // �rnek olarak placeholder metotlar� ekledim, ger�ek implementasyonlarda
            // bu modellerin tam tan�mlar� bulunmal�d�r.

            static std::shared_ptr<AIModel<T>> create_mlp(bool pretrained) {
                // Ger�ek implementasyon burada olacak
                // �imdilik sadece nullptr d�nd�r�yorum
                return nullptr;
            }

            static std::shared_ptr<AIModel<T>> create_simple_resnet(bool pretrained) {
                // Ger�ek implementasyon burada olacak
                return nullptr;
            }

            static std::shared_ptr<AIModel<T>> create_simple_mobilenet(bool pretrained) {
                // Ger�ek implementasyon burada olacak
                return nullptr;
            }
        };

    } // namespace nn
} // namespace tensor

#endif // TENSOR_LAYERS_H