// ForwardModule.h - v0.2.1
// Base class for all forward propagation modules

#ifndef FORWARD_MODULE_H
#define FORWARD_MODULE_H

#include "Tensor.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace tensor {
    namespace forward {

        template<typename T>
        class ForwardModule {
        public:
            virtual ~ForwardModule() = default;

            // Forward pass - to be implemented by derived classes
            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            // Convenience operator for calling forward
            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }

            // Parameters of the module (if any)
            virtual std::vector<Tensor<T>*> parameters() {
                return {};
            }

            // Get parameter count
            virtual size_t parameter_count() const {
                size_t count = 0;
                for (auto param : const_cast<ForwardModule*>(this)->parameters()) {
                    count += param->size();
                }
                return count;
            }

            // Module name (for debugging and display)
            virtual std::string name() const {
                return "ForwardModule";
            }

            // Set training mode
            virtual void train(bool is_training = true) {
                training_ = is_training;

                // Recursively set training mode for child modules
                for (auto& child : children_) {
                    child.second->train(is_training);
                }
            }

            // Set evaluation mode
            virtual void eval() {
                train(false);
            }

            // Check if module is in training mode
            bool is_training() const {
                return training_;
            }

            // Add a child module
            void add_module(const std::string& name, std::shared_ptr<ForwardModule<T>> module) {
                if (children_.find(name) != children_.end()) {
                    throw std::invalid_argument("Module with name '" + name + "' already exists");
                }

                children_[name] = module;
            }

            // Get a child module by name
            std::shared_ptr<ForwardModule<T>> get_module(const std::string& name) {
                auto it = children_.find(name);
                if (it == children_.end()) {
                    throw std::out_of_range("No module with name '" + name + "'");
                }

                return it->second;
            }

            // Check if module has a child with the given name
            bool has_module(const std::string& name) const {
                return children_.find(name) != children_.end();
            }

            // Apply a function to this module and all children recursively
            void apply(std::function<void(ForwardModule<T>&)> fn) {
                fn(*this);

                for (auto& child : children_) {
                    child.second->apply(fn);
                }
            }

            // Get all modules recursively
            std::vector<ForwardModule<T>*> modules() {
                std::vector<ForwardModule<T>*> result;
                result.push_back(this);

                for (auto& child : children_) {
                    auto child_modules = child.second->modules();
                    result.insert(result.end(), child_modules.begin(), child_modules.end());
                }

                return result;
            }

            // Get all children modules (direct descendants only)
            std::unordered_map<std::string, std::shared_ptr<ForwardModule<T>>>& children() {
                return children_;
            }

            const std::unordered_map<std::string, std::shared_ptr<ForwardModule<T>>>& children() const {
                return children_;
            }

            // Save module parameters to file
            virtual void save_parameters(const std::string& filepath) const {
                // In a complete implementation, this would serialize all parameters
                throw std::runtime_error("save_parameters not implemented for this module");
            }

            // Load module parameters from file
            virtual void load_parameters(const std::string& filepath) {
                // In a complete implementation, this would deserialize all parameters
                throw std::runtime_error("load_parameters not implemented for this module");
            }

            // Zero the gradients of all parameters
            virtual void zero_grad() {
                // This would reset any gradients from autograd
                // For now it's a placeholder
            }

            // Print a summary of the module (ASCII version)
            virtual void summary(std::ostream& os = std::cout, size_t indent = 0) const {
                std::string padding(indent, ' ');
                os << padding << name() << " (Parameters: " << parameter_count() << ")" << std::endl;

                for (auto& child : children_) {
                    os << padding << " +- ";  // ASCII version instead of Unicode characters
                    child.second->summary(os, indent + 4);
                }
            }

        protected:
            bool training_ = true;  // Default to training mode
            std::unordered_map<std::string, std::shared_ptr<ForwardModule<T>>> children_;
        };

        // Helper to create a module with automatic type deduction
        template<typename ModuleType, typename... Args>
        std::shared_ptr<ModuleType> make_module(Args&&... args) {
            return std::make_shared<ModuleType>(std::forward<Args>(args)...);
        }

    } // namespace forward
} // namespace tensor

#endif // FORWARD_MODULE_H