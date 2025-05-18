// TensorNN.h - v0.1.0
// Neural network building blocks for DQN and other algorithms

#ifndef TENSOR_NN_H
#define TENSOR_NN_H

#include "Tensor.h"
#include "TensorGrad.h"
#include "ForwardModule.h"
#include <vector>
#include <memory>

namespace tensor {
    namespace nn {

        // Deep Q-Network architecture
        template<typename T>
        class DQN : public forward::ForwardModule<T> {
        public:
            // Constructor for standard DQN
            DQN(size_t state_dim, size_t action_dim, const std::vector<size_t>& hidden_dims = { 128, 128 })
                : state_dim_(state_dim), action_dim_(action_dim) {

                // Build network layers
                size_t input_dim = state_dim;

                // Input layer
                layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ input_dim }), hidden_dims[0]));
                layers_.push_back(std::make_shared<forward::ReLUForward<T>>());

                // Hidden layers
                for (size_t i = 1; i < hidden_dims.size(); ++i) {
                    layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                        TensorShape({ hidden_dims[i - 1] }), hidden_dims[i]));
                    layers_.push_back(std::make_shared<forward::ReLUForward<T>>());
                }

                // Output layer
                layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ hidden_dims.back() }), action_dim));
            }

            // Forward pass
            Tensor<T> forward(const Tensor<T>& state) override {
                Tensor<T> x = state;

                for (auto& layer : layers_) {
                    x = layer->forward(x);
                }

                return x;
            }

            // Get action with highest Q-value
            size_t get_action(const Tensor<T>& state) {
                Tensor<T> q_values = forward(state);

                // Find action with maximum Q-value
                size_t best_action = 0;
                T max_q = q_values.at({ 0, 0 });

                for (size_t a = 1; a < action_dim_; ++a) {
                    if (q_values.at({ 0, a }) > max_q) {
                        max_q = q_values.at({ 0, a });
                        best_action = a;
                    }
                }

                return best_action;
            }

            // Get parameters
            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;

                for (auto& layer : layers_) {
                    auto layer_params = layer->parameters();
                    params.insert(params.end(), layer_params.begin(), layer_params.end());
                }

                return params;
            }

        private:
            size_t state_dim_;
            size_t action_dim_;
            std::vector<std::shared_ptr<forward::ForwardModule<T>>> layers_;
        };

        // Dueling DQN architecture
        template<typename T>
        class DuelingDQN : public forward::ForwardModule<T> {
        public:
            // Constructor for dueling DQN
            DuelingDQN(size_t state_dim, size_t action_dim,
                const std::vector<size_t>& hidden_dims = { 128, 128 },
                size_t advantage_hidden = 128,
                size_t value_hidden = 128)
                : state_dim_(state_dim), action_dim_(action_dim) {

                // Feature extractor (shared layers)
                feature_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ state_dim }), hidden_dims[0]));
                feature_layers_.push_back(std::make_shared<forward::ReLUForward<T>>());

                for (size_t i = 1; i < hidden_dims.size(); ++i) {
                    feature_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                        TensorShape({ hidden_dims[i - 1] }), hidden_dims[i]));
                    feature_layers_.push_back(std::make_shared<forward::ReLUForward<T>>());
                }

                // Value stream
                value_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ hidden_dims.back() }), value_hidden));
                value_layers_.push_back(std::make_shared<forward::ReLUForward<T>>());
                value_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ value_hidden }), 1));

                // Advantage stream
                advantage_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ hidden_dims.back() }), advantage_hidden));
                advantage_layers_.push_back(std::make_shared<forward::ReLUForward<T>>());
                advantage_layers_.push_back(std::make_shared<forward::LinearForward<T>>(
                    TensorShape({ advantage_hidden }), action_dim));
            }

            // Forward pass
            Tensor<T> forward(const Tensor<T>& state) override {
                // Feature extraction
                Tensor<T> features = state;
                for (auto& layer : feature_layers_) {
                    features = layer->forward(features);
                }

                // Value stream
                Tensor<T> value = features;
                for (auto& layer : value_layers_) {
                    value = layer->forward(value);
                }

                // Advantage stream
                Tensor<T> advantage = features;
                for (auto& layer : advantage_layers_) {
                    advantage = layer->forward(advantage);
                }

                // Combine value and advantage
                Tensor<T> q_values(TensorShape({ state.shape().dim(0), action_dim_ }));

                // Calculate mean advantage
                for (size_t b = 0; b < state.shape().dim(0); ++b) {
                    T mean_advantage = 0;
                    for (size_t a = 0; a < action_dim_; ++a) {
                        mean_advantage += advantage.at({ b, a });
                    }
                    mean_advantage /= action_dim_;

                    // Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
                    for (size_t a = 0; a < action_dim_; ++a) {
                        q_values.at({ b, a }) = value.at({ b, 0 }) +
                            (advantage.at({ b, a }) - mean_advantage);
                    }
                }

                return q_values;
            }

            // Get action with highest Q-value
            size_t get_action(const Tensor<T>& state) {
                Tensor<T> q_values = forward(state);

                // Find action with maximum Q-value
                size_t best_action = 0;
                T max_q = q_values.at({ 0, 0 });

                for (size_t a = 1; a < action_dim_; ++a) {
                    if (q_values.at({ 0, a }) > max_q) {
                        max_q = q_values.at({ 0, a });
                        best_action = a;
                    }
                }

                return best_action;
            }

            // Get parameters
            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> params;

                auto add_params = [&params](const std::vector<std::shared_ptr<forward::ForwardModule<T>>>& layers) {
                    for (auto& layer : layers) {
                        auto layer_params = layer->parameters();
                        params.insert(params.end(), layer_params.begin(), layer_params.end());
                    }
                    };

                add_params(feature_layers_);
                add_params(value_layers_);
                add_params(advantage_layers_);

                return params;
            }

        private:
            size_t state_dim_;
            size_t action_dim_;
            std::vector<std::shared_ptr<forward::ForwardModule<T>>> feature_layers_;
            std::vector<std::shared_ptr<forward::ForwardModule<T>>> value_layers_;
            std::vector<std::shared_ptr<forward::ForwardModule<T>>> advantage_layers_;
        };

    } // namespace nn
} // namespace tensor

#endif // TENSOR_NN_H