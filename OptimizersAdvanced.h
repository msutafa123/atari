// OptimizersAdvanced.h
#ifndef OPTIMIZERS_ADVANCED_H
#define OPTIMIZERS_ADVANCED_H

#include "Tensor.h"
#include "TensorLayers.h"
#include <vector>
#include <cmath>

namespace tensor {
    namespace optim {

        template<typename T>
        class Adam {
        public:
            Adam(const std::vector<Tensor<T>*>& parameters,
                T learning_rate = T(0.001),
                T beta1 = T(0.9),
                T beta2 = T(0.999),
                T epsilon = T(1e-8),
                T weight_decay = T(0))
                : parameters_(parameters),
                learning_rate_(learning_rate),
                beta1_(beta1), beta2_(beta2),
                epsilon_(epsilon),
                weight_decay_(weight_decay),
                t_(0) {

                // Initialize moment vectors
                for (const auto& param : parameters_) {
                    m_.push_back(Tensor<T>::zeros(param->shape()));
                    v_.push_back(Tensor<T>::zeros(param->shape()));
                }
            }

            void step() {
                t_++;

                // Get available CPU threads for parallel computation
                size_t num_threads = ParallelCompute::get_max_threads();

                for (size_t i = 0; i < parameters_.size(); ++i) {
                    auto& param = *(parameters_[i]);
                    auto& m = m_[i];
                    auto& v = v_[i];

                    // Apply weight decay if needed
                    if (weight_decay_ > 0) {
                        for (size_t j = 0; j < param.size(); ++j) {
                            param.grad().data()[j] += weight_decay_ * param.data()[j];
                        }
                    }

                    // Parallel computation of moment updates
                    ParallelCompute::parallel_for(0, param.size(), [&](size_t j) {
                        // Update biased first moment estimate
                        m.data()[j] = beta1_ * m.data()[j] + (T(1) - beta1_) * param.grad().data()[j];

                        // Update biased second raw moment estimate
                        v.data()[j] = beta2_ * v.data()[j] + (T(1) - beta2_) *
                            param.grad().data()[j] * param.grad().data()[j];

                        // Bias-corrected moment estimates
                        T m_hat = m.data()[j] / (T(1) - std::pow(beta1_, t_));
                        T v_hat = v.data()[j] / (T(1) - std::pow(beta2_, t_));

                        // Update parameters
                        param.data()[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                        }, num_threads);
                }
            }

            void set_learning_rate(T lr) {
                learning_rate_ = lr;
            }

        private:
            std::vector<Tensor<T>*> parameters_;
            T learning_rate_;
            T beta1_;
            T beta2_;
            T epsilon_;
            T weight_decay_;
            size_t t_;  // Time step
            std::vector<Tensor<T>> m_;  // First moment estimates
            std::vector<Tensor<T>> v_;  // Second moment estimates
        };

    } // namespace optim
} // namespace tensor

#endif // OPTIMIZERS_ADVANCED_H