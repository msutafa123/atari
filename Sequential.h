// Sequential.h - v0.2.0
// Sequential container for forward modules

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "ForwardModule.h"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <initializer_list>
#include <sstream>

namespace tensor {
    namespace forward {

        template<typename T>
        class Sequential : public ForwardModule<T> {
        public:
            // Default constructor
            Sequential() = default;

            // Constructor with initial modules
            Sequential(std::initializer_list<std::shared_ptr<ForwardModule<T>>> modules) {
                for (const auto& module : modules) {
                    add(module);
                }
            }

            // Add a module to the sequence
            void add(std::shared_ptr<ForwardModule<T>> module) {
                if (!module) {
                    throw std::invalid_argument("Cannot add null module to Sequential");
                }
                modules_.push_back(module);
            }

            // Add module by type with constructor arguments
            template<typename ModuleType, typename... Args>
            std::shared_ptr<ModuleType> add(Args&&... args) {
                static_assert(std::is_base_of<ForwardModule<T>, ModuleType>::value,
                    "Module must inherit from ForwardModule");

                auto module = std::make_shared<ModuleType>(std::forward<Args>(args)...);
                modules_.push_back(module);
                return module;
            }

            // Forward pass through all modules in sequence
            Tensor<T> forward(const Tensor<T>& input) override {
                if (modules_.empty()) {
                    return input;  // Identity operation if empty
                }

                Tensor<T> current = input;

                for (auto& module : modules_) {
                    current = module->forward(current);
                }

                return current;
            }

            // Get all parameters from all modules
            std::vector<Tensor<T>*> parameters() override {
                std::vector<Tensor<T>*> all_params;

                for (auto& module : modules_) {
                    auto module_params = module->parameters();
                    all_params.insert(all_params.end(), module_params.begin(), module_params.end());
                }

                return all_params;
            }

            // Get number of modules
            size_t size() const {
                return modules_.size();
            }

            // Access module by index
            std::shared_ptr<ForwardModule<T>> at(size_t index) {
                if (index >= modules_.size()) {
                    throw std::out_of_range("Module index out of range");
                }
                return modules_[index];
            }

            const std::shared_ptr<ForwardModule<T>> at(size_t index) const {
                if (index >= modules_.size()) {
                    throw std::out_of_range("Module index out of range");
                }
                return modules_[index];
            }

            // Check if sequence is empty
            bool empty() const {
                return modules_.empty();
            }

            // Clear all modules
            void clear() {
                modules_.clear();
            }

            // Insert module at specific position
            void insert(size_t index, std::shared_ptr<ForwardModule<T>> module) {
                if (index > modules_.size()) {
                    throw std::out_of_range("Insert index out of range");
                }

                if (!module) {
                    throw std::invalid_argument("Cannot insert null module");
                }

                modules_.insert(modules_.begin() + index, module);
            }

            // Remove module at specific position
            void remove(size_t index) {
                if (index >= modules_.size()) {
                    throw std::out_of_range("Remove index out of range");
                }

                modules_.erase(modules_.begin() + index);
            }

            // Replace module at specific position
            void replace(size_t index, std::shared_ptr<ForwardModule<T>> module) {
                if (index >= modules_.size()) {
                    throw std::out_of_range("Replace index out of range");
                }

                if (!module) {
                    throw std::invalid_argument("Cannot replace with null module");
                }

                modules_[index] = module;
            }

            // Calculate output shape for a given input shape
            TensorShape output_shape(const TensorShape& input_shape) const {
                if (modules_.empty()) {
                    return input_shape;  // Identity operation if empty
                }

                TensorShape current_shape = input_shape;

                for (const auto& module : modules_) {
                    // This requires each module to have an output_shape method
                    // which isn't defined in the base ForwardModule class.
                    // In practice, this would be added or handled differently.
                    if (auto shaped_module = std::dynamic_pointer_cast<HasOutputShape<T>>(module)) {
                        current_shape = shaped_module->output_shape(current_shape);
                    }
                    else {
                        // Cannot determine output shape for this module
                        throw std::runtime_error("Module does not support output shape calculation");
                    }
                }

                return current_shape;
            }

            // Module name
            std::string name() const override {
                std::stringstream ss;
                ss << "Sequential(";

                for (size_t i = 0; i < modules_.size(); ++i) {
                    ss << modules_[i]->name();
                    if (i < modules_.size() - 1) {
                        ss << ", ";
                    }
                }

                ss << ")";
                return ss.str();
            }

            // Iterators for range-based for loops
            typename std::vector<std::shared_ptr<ForwardModule<T>>>::iterator begin() {
                return modules_.begin();
            }

            typename std::vector<std::shared_ptr<ForwardModule<T>>>::iterator end() {
                return modules_.end();
            }

            typename std::vector<std::shared_ptr<ForwardModule<T>>>::const_iterator begin() const {
                return modules_.begin();
            }

            typename std::vector<std::shared_ptr<ForwardModule<T>>>::const_iterator end() const {
                return modules_.end();
            }

        private:
            std::vector<std::shared_ptr<ForwardModule<T>>> modules_;

            // Interface for modules that can compute output shape
            template<typename U>
            class HasOutputShape {
            public:
                virtual ~HasOutputShape() = default;
                virtual TensorShape output_shape(const TensorShape& input_shape) const = 0;
            };
        };

    } // namespace forward
} // namespace tensor

#endif // SEQUENTIAL_H