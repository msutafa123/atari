// ReplayBuffer.h - v0.1.0
// Experience replay buffer for reinforcement learning

#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include "Tensor.h"
#include <vector>
#include <random>
#include <algorithm>

namespace tensor {
    namespace rl {

        template<typename T>
        struct Experience {
            Tensor<T> state;
            Tensor<T> action;
            T reward;
            Tensor<T> next_state;
            bool done;
        };

        template<typename T>
        class ReplayBuffer {
        public:
            // Constructor with buffer capacity
            explicit ReplayBuffer(size_t capacity) : capacity_(capacity), position_(0) {
                buffer_.reserve(capacity);
            }

            // Add experience to buffer
            void add(const Experience<T>& experience) {
                if (buffer_.size() < capacity_) {
                    buffer_.push_back(experience);
                }
                else {
                    buffer_[position_] = experience;
                }

                position_ = (position_ + 1) % capacity_;
            }

            // Sample batch of experiences
            std::vector<Experience<T>> sample(size_t batch_size) {
                if (buffer_.size() < batch_size) {
                    throw std::runtime_error("Not enough experiences in buffer");
                }

                std::vector<Experience<T>> batch;
                batch.reserve(batch_size);

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);

                for (size_t i = 0; i < batch_size; ++i) {
                    size_t idx = dist(gen);
                    batch.push_back(buffer_[idx]);
                }

                return batch;
            }

            // Get current buffer size
            size_t size() const {
                return buffer_.size();
            }

            // Check if buffer is full
            bool is_full() const {
                return buffer_.size() == capacity_;
            }

        private:
            std::vector<Experience<T>> buffer_;
            size_t capacity_;
            size_t position_;
        };

    } // namespace rl
} // namespace tensor

#endif // REPLAY_BUFFER_H