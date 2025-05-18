// PrioritizedReplayBuffer.h - v0.1.0
// Prioritized experience replay for DQN and similar algorithms

#ifndef PRIORITIZED_REPLAY_BUFFER_H
#define PRIORITIZED_REPLAY_BUFFER_H

#include "ReplayBuffer.h"
#include <cmath>
#include <memory>
#include <algorithm>

namespace tensor {
    namespace rl {

        template<typename T>
        struct PrioritizedExperience : public Experience<T> {
            T priority;
            size_t index;
        };

        template<typename T>
        class PrioritizedReplayBuffer {
        public:
            // Constructor with buffer capacity and hyperparameters
            PrioritizedReplayBuffer(size_t capacity, T alpha = 0.6, T beta = 0.4, T epsilon = 1e-6)
                : capacity_(capacity), position_(0), alpha_(alpha), beta_(beta), beta_increment_(0.001),
                epsilon_(epsilon), max_priority_(1.0) {

                buffer_.reserve(capacity);
                priorities_.reserve(capacity);
            }

            // Add experience to buffer
            void add(const Experience<T>& experience) {
                T priority = max_priority_;

                if (buffer_.size() < capacity_) {
                    buffer_.push_back(experience);
                    priorities_.push_back(priority);
                }
                else {
                    buffer_[position_] = experience;
                    priorities_[position_] = priority;
                }

                position_ = (position_ + 1) % capacity_;
            }

            // Sample batch of experiences with priorities
            std::vector<PrioritizedExperience<T>> sample(size_t batch_size) {
                if (buffer_.size() < batch_size) {
                    throw std::runtime_error("Not enough experiences in buffer");
                }

                std::vector<PrioritizedExperience<T>> batch;
                batch.reserve(batch_size);

                // Calculate sampling probabilities
                std::vector<T> probabilities = calculate_probabilities();

                // Calculate importance sampling weights
                T beta = std::min(T(1.0), beta_ + beta_increment_);

                // Sample experiences
                std::vector<size_t> indices = sample_indices(batch_size, probabilities);

                for (size_t idx : indices) {
                    PrioritizedExperience<T> prioritized_exp;
                    prioritized_exp.state = buffer_[idx].state;
                    prioritized_exp.action = buffer_[idx].action;
                    prioritized_exp.reward = buffer_[idx].reward;
                    prioritized_exp.next_state = buffer_[idx].next_state;
                    prioritized_exp.done = buffer_[idx].done;
                    prioritized_exp.priority = priorities_[idx];
                    prioritized_exp.index = idx;

                    batch.push_back(prioritized_exp);
                }

                return batch;
            }

            // Update priorities based on TD error
            void update_priorities(const std::vector<size_t>& indices, const std::vector<T>& td_errors) {
                if (indices.size() != td_errors.size()) {
                    throw std::invalid_argument("Indices and TD errors must have same size");
                }

                for (size_t i = 0; i < indices.size(); ++i) {
                    size_t idx = indices[i];
                    if (idx >= buffer_.size()) {
                        throw std::out_of_range("Index out of buffer range");
                    }

                    // Update priority based on TD error
                    T priority = std::pow(std::abs(td_errors[i]) + epsilon_, alpha_);
                    priorities_[idx] = priority;

                    // Update max priority
                    max_priority_ = std::max(max_priority_, priority);
                }
            }

            // Get current buffer size
            size_t size() const {
                return buffer_.size();
            }

        private:
            std::vector<Experience<T>> buffer_;
            std::vector<T> priorities_;
            size_t capacity_;
            size_t position_;
            T alpha_;        // Priority exponent
            T beta_;         // Importance sampling exponent
            T beta_increment_; // Annealing rate for beta
            T epsilon_;      // Small constant to avoid zero priority
            T max_priority_; // Maximum priority seen so far

            // Calculate sampling probabilities based on priorities
            std::vector<T> calculate_probabilities() {
                std::vector<T> probs(buffer_.size());

                // Sum of all priorities
                T sum_priorities = 0;
                for (const auto& p : priorities_) {
                    sum_priorities += p;
                }

                // Calculate probabilities
                for (size_t i = 0; i < buffer_.size(); ++i) {
                    probs[i] = priorities_[i] / sum_priorities;
                }

                return probs;
            }

            // Sample indices based on probabilities
            std::vector<size_t> sample_indices(size_t batch_size, const std::vector<T>& probs) {
                std::vector<size_t> indices;
                indices.reserve(batch_size);

                std::random_device rd;
                std::mt19937 gen(rd());

                // Simple weighted sampling implementation
                for (size_t i = 0; i < batch_size; ++i) {
                    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
                    indices.push_back(dist(gen));
                }

                return indices;
            }
        };

    } // namespace rl
} // namespace tensor

#endif // PRIORITIZED_REPLAY_BUFFER_H