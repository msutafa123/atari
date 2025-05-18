// Optimizer.h - v0.1.0
// Parameter optimization for gradient-based learning

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Tensor.h"
#include "TensorGrad.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace tensor {
    namespace optim {

        // Base optimizer class
        template<typename T>
        class Optimizer {
        public:
            Optimizer(const std::vector<Tensor<T>*>& parameters, T learning_rate = T(0.01))
                : parameters_(parameters), learning_rate_(learning_rate) {
            }

            virtual ~Optimizer() = default;

            // Update parameters based on their gradients
            virtual void step() = 0;

            // Zero all parameter gradients
            void zero_grad() {
                for (auto& param : parameters_) {
                    // In a real implementation, would clear gradients
                }
            }

            // Set learning rate
            void set_learning_rate(T lr) {
                learning_rate_ = lr;
            }

            // Get learning rate
            T get_learning_rate() const {
                return learning_rate_;
            }

        protected:
            std::vector<Tensor<T>*> parameters_;
            T learning_rate_;
        };

        // SGD Optimizer with momentum option
        template<typename T>
        class SGD : public Optimizer<T> {
        public:
            SGD(const std::vector<Tensor<T>*>& parameters,
                T learning_rate = T(0.01),
                T momentum = T(0),
                T weight_decay = T(0))
                : Optimizer<T>(parameters, learning_rate),
                momentum_(momentum),
                weight_decay_(weight_decay) {

                if (momentum > 0) {
                    // Initialize velocity for each parameter
                    for (const auto& param : this->parameters_) {
                        velocities_.push_back(Tensor<T>(param->shape()));
                        velocities_.back().fill(T(0));
                    }
                }
            }

            void step() override {
                for (size_t i = 0; i < this->parameters_.size(); ++i) {
                    auto& param = *(this->parameters_[i]);

                    // Apply weight decay
                    if (weight_decay_ > 0) {
                        // In real implementation, would modify gradients here
                    }

                    if (momentum_ > 0) {
                        auto& velocity = velocities_[i];

                        // Update velocity and parameters
                        for (size_t j = 0; j < param.size(); ++j) {
                            // In real implementation would use gradients to update parameters
                        }
                    }
                    else {
                        // Standard SGD update
                        for (size_t j = 0; j < param.size(); ++j) {
                            // In real implementation would use gradients to update parameters
                        }
                    }
                }
            }

        private:
            T momentum_;
            T weight_decay_;
            std::vector<Tensor<T>> velocities_;
        };

        // Adam optimizer
        template<typename T>
        class Adam : public Optimizer<T> {
        public:
            Adam(const std::vector<Tensor<T>*>& parameters,
                T learning_rate = T(0.001),
                T beta1 = T(0.9),
                T beta2 = T(0.999),
                T epsilon = T(1e-8),
                T weight_decay = T(0))
                : Optimizer<T>(parameters, learning_rate),
                beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
                weight_decay_(weight_decay), t_(0) {

                // Initialize moment vectors
                for (const auto& param : this->parameters_) {
                    m_.push_back(Tensor<T>(param->shape()));
                    v_.push_back(Tensor<T>(param->shape()));
                    m_.back().fill(T(0));
                    v_.back().fill(T(0));
                }
            }

            void step() override {
                t_++;

                for (size_t i = 0; i < this->parameters_.size(); ++i) {
                    auto& param = *(this->parameters_[i]);
                    auto& m = m_[i];
                    auto& v = v_[i];

                    // Apply weight decay
                    if (weight_decay_ > 0) {
                        // In real implementation, would modify gradients here
                    }

                    for (size_t j = 0; j < param.size(); ++j) {
                        // In real implementation would use gradients to update parameters using Adam algorithm
                    }
                }
            }

        private:
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

#endif // OPTIMIZER_H