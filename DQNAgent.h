// DQNAgent.h - v0.1.0
// Deep Q-Network agent implementation

#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include "TensorNN.h"
#include "ReplayBuffer.h"
#include "PrioritizedReplayBuffer.h"
#include <random>
#include <algorithm>

namespace tensor {
    namespace rl {

        template<typename T>
        class DQNAgent {
        public:
            // Constructor for standard DQN
            DQNAgent(size_t state_dim, size_t action_dim,
                size_t buffer_size = 10000,
                size_t batch_size = 64,
                T gamma = 0.99,
                T epsilon_start = 1.0,
                T epsilon_end = 0.01,
                T epsilon_decay = 0.995,
                bool use_double_dqn = false,
                bool use_prioritized = false)
                : state_dim_(state_dim), action_dim_(action_dim),
                buffer_size_(buffer_size), batch_size_(batch_size),
                gamma_(gamma), epsilon_(epsilon_start),
                epsilon_end_(epsilon_end), epsilon_decay_(epsilon_decay),
                use_double_dqn_(use_double_dqn), use_prioritized_(use_prioritized),
                steps_(0) {

                // Create Q-networks
                policy_net_ = std::make_shared<nn::DQN<T>>(state_dim, action_dim);
                target_net_ = std::make_shared<nn::DQN<T>>(state_dim, action_dim);

                // Initialize target network with policy network weights
                update_target_network();

                // Create replay buffer
                if (use_prioritized_) {
                    prioritized_buffer_ = std::make_shared<PrioritizedReplayBuffer<T>>(buffer_size);
                }
                else {
                    replay_buffer_ = std::make_shared<ReplayBuffer<T>>(buffer_size);
                }

                // Random generator for action selection
                random_engine_ = std::mt19937(std::random_device()());
                action_dist_ = std::uniform_int_distribution<size_t>(0, action_dim - 1);
            }

            // Select action using epsilon-greedy policy
            size_t select_action(const Tensor<T>& state) {
                std::uniform_real_distribution<T> dist(0, 1);

                // Exploration
                if (dist(random_engine_) < epsilon_) {
                    return action_dist_(random_engine_);
                }

                // Exploitation
                return policy_net_->get_action(state);
            }

            // Store experience in replay buffer
            void store_experience(const Experience<T>& experience) {
                if (use_prioritized_) {
                    prioritized_buffer_->add(experience);
                }
                else {
                    replay_buffer_->add(experience);
                }
            }

            // Update neural networks
            void update(size_t update_target_every = 10) {
                // Check if we have enough experiences
                if ((use_prioritized_ && prioritized_buffer_->size() < batch_size_) ||
                    (!use_prioritized_ && replay_buffer_->size() < batch_size_)) {
                    return;
                }

                // Sample batch from replay buffer
                std::vector<Experience<T>> batch;

                if (use_prioritized_) {
                    auto prioritized_batch = prioritized_buffer_->sample(batch_size_);
                    for (const auto& exp : prioritized_batch) {
                        batch.push_back(static_cast<Experience<T>>(exp));
                    }
                }
                else {
                    batch = replay_buffer_->sample(batch_size_);
                }

                // Prepare batch tensors
                Tensor<T> states(TensorShape({ batch_size_, state_dim_ }));
                Tensor<T> actions(TensorShape({ batch_size_, 1 }));
                Tensor<T> rewards(TensorShape({ batch_size_, 1 }));
                Tensor<T> next_states(TensorShape({ batch_size_, state_dim_ }));
                Tensor<T> dones(TensorShape({ batch_size_, 1 }));

                for (size_t i = 0; i < batch_size_; ++i) {
                    // Copy state
                    for (size_t j = 0; j < state_dim_; ++j) {
                        states.at({ i, j }) = batch[i].state.at({ 0, j });
                    }

                    // Copy action (assuming scalar action)
                    actions.at({ i, 0 }) = batch[i].action.at({ 0, 0 });

                    // Copy reward
                    rewards.at({ i, 0 }) = batch[i].reward;

                    // Copy next state
                    for (size_t j = 0; j < state_dim_; ++j) {
                        next_states.at({ i, j }) = batch[i].next_state.at({ 0, j });
                    }

                    // Copy done flag
                    dones.at({ i, 0 }) = batch[i].done ? 1.0 : 0.0;
                }

                // Compute Q-values
                Tensor<T> q_values = policy_net_->forward(states);

                // Compute target Q-values
                Tensor<T> next_q_values;

                if (use_double_dqn_) {
                    // Double DQN: select actions using policy network
                    Tensor<T> next_actions_q = policy_net_->forward(next_states);

                    // Find best actions
                    std::vector<size_t> best_actions(batch_size_);
                    for (size_t i = 0; i < batch_size_; ++i) {
                        size_t best_action = 0;
                        T max_q = next_actions_q.at({ i, 0 });

                        for (size_t a = 1; a < action_dim_; ++a) {
                            if (next_actions_q.at({ i, a }) > max_q) {
                                max_q = next_actions_q.at({ i, a });
                                best_action = a;
                            }
                        }

                        best_actions[i] = best_action;
                    }

                    // Evaluate actions using target network
                    Tensor<T> target_q_values = target_net_->forward(next_states);
                    next_q_values = Tensor<T>(TensorShape({ batch_size_, 1 }));

                    for (size_t i = 0; i < batch_size_; ++i) {
                        next_q_values.at({ i, 0 }) = target_q_values.at({ i, best_actions[i] });
                    }
                }
                else {
                    // Standard DQN: use max Q-value from target network
                    Tensor<T> target_q_values = target_net_->forward(next_states);
                    next_q_values = Tensor<T>(TensorShape({ batch_size_, 1 }));

                    for (size_t i = 0; i < batch_size_; ++i) {
                        T max_q = target_q_values.at({ i, 0 });

                        for (size_t a = 1; a < action_dim_; ++a) {
                            max_q = std::max(max_q, target_q_values.at({ i, a }));
                        }

                        next_q_values.at({ i, 0 }) = max_q;
                    }
                }

                // Compute target values
                Tensor<T> target_values(TensorShape({ batch_size_, 1 }));
                std::vector<T> td_errors(batch_size_);

                for (size_t i = 0; i < batch_size_; ++i) {
                    // Target = reward + gamma * max_a Q(s', a) * (1 - done)
                    T target = rewards.at({ i, 0 }) +
                        gamma_ * next_q_values.at({ i, 0 }) * (1 - dones.at({ i, 0 }));

                    target_values.at({ i, 0 }) = target;

                    // Calculate TD error for prioritized replay
                    size_t action = static_cast<size_t>(actions.at({ i, 0 }));
                    T current_q = q_values.at({ i, action });
                    td_errors[i] = target - current_q;
                }

                // Update priorities if using prioritized replay
                if (use_prioritized_) {
                    std::vector<size_t> indices;
                    auto prioritized_batch = prioritized_buffer_->sample(batch_size_);
                    for (const auto& exp : prioritized_batch) {
                        indices.push_back(exp.index);
                    }

                    prioritized_buffer_->update_priorities(indices, td_errors);
                }

                // Update network weights
                // In a real implementation, this would use a proper optimizer
                // For simplicity, we'll just simulate the update process

                // Update target network periodically
                steps_++;
                if (steps_ % update_target_every == 0) {
                    update_target_network();
                }

                // Decay epsilon
                epsilon_ = std::max(epsilon_end_, epsilon_ * epsilon_decay_);
            }

            // Update target network with policy network weights
            void update_target_network() {
                // In a real implementation, this would copy weights from policy to target
                target_net_ = std::make_shared<nn::DQN<T>>(state_dim_, action_dim_);

                // Simulating weight copying
                auto policy_params = policy_net_->parameters();
                auto target_params = target_net_->parameters();

                for (size_t i = 0; i < policy_params.size() && i < target_params.size(); ++i) {
                    for (size_t j = 0; j < policy_params[i]->size(); ++j) {
                        (*target_params[i])[j] = (*policy_params[i])[j];
                    }
                }
            }

            // Get current exploration rate
            T get_epsilon() const {
                return epsilon_;
            }

        private:
            // DQNAgent.h devamý

private:
    size_t state_dim_;
    size_t action_dim_;
    size_t buffer_size_;
    size_t batch_size_;
    T gamma_;               // Discount factor
    T epsilon_;             // Current exploration rate
    T epsilon_end_;         // Minimum exploration rate
    T epsilon_decay_;       // Decay rate for epsilon
    bool use_double_dqn_;   // Whether to use Double DQN
    bool use_prioritized_;  // Whether to use prioritized replay
    size_t steps_;          // Total steps taken

    std::shared_ptr<nn::DQN<T>> policy_net_;
    std::shared_ptr<nn::DQN<T>> target_net_;
    std::shared_ptr<ReplayBuffer<T>> replay_buffer_;
    std::shared_ptr<PrioritizedReplayBuffer<T>> prioritized_buffer_;

    std::mt19937 random_engine_;
    std::uniform_int_distribution<size_t> action_dist_;

    // Helper methods for training would go here

    // Compute loss for a batch
    T compute_loss(const Tensor<T>& states, const Tensor<T>& actions,
        const Tensor<T>& rewards, const Tensor<T>& next_states,
        const Tensor<T>& dones) {

        // Get predicted Q-values for all actions
        Tensor<T> predicted_q_all = policy_net_->forward(states);

        // Extract Q-values for the actions that were actually taken
        Tensor<T> predicted_q(TensorShape({ batch_size_, 1 }));
        for (size_t i = 0; i < batch_size_; ++i) {
            size_t action = static_cast<size_t>(actions.at({ i, 0 }));
            predicted_q.at({ i, 0 }) = predicted_q_all.at({ i, action });
        }

        // Get target Q-values
        Tensor<T> with_target_q;
        if (use_double_dqn_) {
            with_target_q = compute_double_q_targets(next_states, rewards, dones);
        }
        else {
            with_target_q = compute_q_targets(next_states, rewards, dones);
        }

        // Compute MSE loss
        T loss = 0;
        for (size_t i = 0; i < batch_size_; ++i) {
            T diff = predicted_q.at({ i, 0 }) - with_target_q.at({ i, 0 });
            loss += diff * diff;
        }

        return loss / batch_size_;
    }

    // Compute target Q-values for standard DQN
    Tensor<T> compute_q_targets(const Tensor<T>& next_states,
        const Tensor<T>& rewards,
        const Tensor<T>& dones) {

        // Get max Q-values from target network
        Tensor<T> next_q_values = target_net_->forward(next_states);
        Tensor<T> max_next_q(TensorShape({ batch_size_, 1 }));

        for (size_t i = 0; i < batch_size_; ++i) {
            T max_q = next_q_values.at({ i, 0 });
            for (size_t a = 1; a < action_dim_; ++a) {
                if (next_q_values.at({ i, a }) > max_q) {
                    max_q = next_q_values.at({ i, a });
                }
            }
            max_next_q.at({ i, 0 }) = max_q;
        }

        // Compute target values: r + gamma * max Q(s', a') * (1 - done)
        Tensor<T> target_q(TensorShape({ batch_size_, 1 }));
        for (size_t i = 0; i < batch_size_; ++i) {
            target_q.at({ i, 0 }) = rewards.at({ i, 0 }) +
                gamma_ * max_next_q.at({ i, 0 }) * (1 - dones.at({ i, 0 }));
        }

        return target_q;
    }

    // Compute target Q-values for Double DQN
    Tensor<T> compute_double_q_targets(const Tensor<T>& next_states,
        const Tensor<T>& rewards,
        const Tensor<T>& dones) {

        // Get actions from policy network
        Tensor<T> next_q_policy = policy_net_->forward(next_states);
        std::vector<size_t> best_actions(batch_size_);

        for (size_t i = 0; i < batch_size_; ++i) {
            size_t best_action = 0;
            T max_q = next_q_policy.at({ i, 0 });

            for (size_t a = 1; a < action_dim_; ++a) {
                if (next_q_policy.at({ i, a }) > max_q) {
                    max_q = next_q_policy.at({ i, a });
                    best_action = a;
                }
            }

            best_actions[i] = best_action;
        }

        // Get Q-values from target network for selected actions
        Tensor<T> next_q_target = target_net_->forward(next_states);
        Tensor<T> next_q(TensorShape({ batch_size_, 1 }));

        for (size_t i = 0; i < batch_size_; ++i) {
            next_q.at({ i, 0 }) = next_q_target.at({ i, best_actions[i] });
        }

        // Compute target values: r + gamma * Q_target(s', argmax_a' Q_policy(s', a')) * (1 - done)
        Tensor<T> target_q(TensorShape({ batch_size_, 1 }));
        for (size_t i = 0; i < batch_size_; ++i) {
            target_q.at({ i, 0 }) = rewards.at({ i, 0 }) +
                gamma_ * next_q.at({ i, 0 }) * (1 - dones.at({ i, 0 }));
        }

        return target_q;
    }

    // Save model
    void save_model(const std::string& path) {
        // In a real implementation, this would serialize the networks
        // For now, just a placeholder
    }

    // Load model
    void load_model(const std::string& path) {
        // In a real implementation, this would deserialize the networks
        // For now, just a placeholder
    }
};

} // namespace rl
} // namespace tensor

#endif // DQN_AGENT_H