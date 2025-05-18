// autograd.h - v1.0.0
// Automatic differentiation and backpropagation for tensors
// C++17 standards compliant

#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "Tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace tensor {
    namespace autograd {

        // Forward declarations
        template<typename T> class Variable;
        template<typename T> class Node;
        template<typename T> class ComputationGraph;

        // Gradient accumulation mode
        enum class GradientMode {
            ACCUMULATE,  // Add to existing gradients (default)
            REPLACE      // Replace gradients (for optimizers)
        };

        // A node in the computation graph that tracks operation history
        template<typename T>
        class Node {
        public:
            virtual ~Node() = default;

            // Compute gradients for inputs based on gradient of output
            virtual void backward(const Tensor<T>& grad_output) = 0;

            // Operation name for debugging
            virtual std::string op_name() const = 0;

            // Set parent nodes (inputs to this operation)
            void set_parents(const std::vector<std::shared_ptr<Variable<T>>>& parents) {
                parents_ = parents;
            }

            // Get parent nodes
            const std::vector<std::shared_ptr<Variable<T>>>& parents() const {
                return parents_;
            }

        protected:
            std::vector<std::shared_ptr<Variable<T>>> parents_;
        };

        // The central class for automatic differentiation
        template<typename T>
        class Variable {
        public:
            // Create a variable without gradient tracking
            explicit Variable(const Tensor<T>& data, bool requires_grad = false)
                : data_(data), requires_grad_(requires_grad), grad_fn_(nullptr) {
                if (requires_grad) {
                    grad_ = Tensor<T>(data.shape());
                    grad_.fill(T(0));
                }
            }

            // Create a variable with gradient tracking and operation history
            Variable(const Tensor<T>& data,
                std::shared_ptr<Node<T>> grad_fn,
                bool requires_grad = true)
                : data_(data), requires_grad_(requires_grad), grad_fn_(grad_fn) {
                if (requires_grad) {
                    grad_ = Tensor<T>(data.shape());
                    grad_.fill(T(0));
                }
            }

            // Copy constructor
            Variable(const Variable& other)
                : data_(other.data_),
                requires_grad_(other.requires_grad_),
                grad_fn_(other.grad_fn_) {
                if (other.grad_.size() > 0) {
                    grad_ = other.grad_.clone();
                }
            }

            // Move constructor
            Variable(Variable&& other) noexcept
                : data_(std::move(other.data_)),
                requires_grad_(other.requires_grad_),
                grad_fn_(std::move(other.grad_fn_)),
                grad_(std::move(other.grad_)) {
                other.requires_grad_ = false;
            }

            // Copy assignment
            Variable& operator=(const Variable& other) {
                if (this != &other) {
                    data_ = other.data_;
                    requires_grad_ = other.requires_grad_;
                    grad_fn_ = other.grad_fn_;

                    if (other.grad_.size() > 0) {
                        grad_ = other.grad_.clone();
                    }
                    else {
                        grad_ = Tensor<T>();
                    }
                }
                return *this;
            }

            // Move assignment
            Variable& operator=(Variable&& other) noexcept {
                if (this != &other) {
                    data_ = std::move(other.data_);
                    requires_grad_ = other.requires_grad_;
                    grad_fn_ = std::move(other.grad_fn_);
                    grad_ = std::move(other.grad_);

                    other.requires_grad_ = false;
                }
                return *this;
            }

            // Get tensor data
            const Tensor<T>& data() const {
                return data_;
            }

            // Get gradient
            const Tensor<T>& grad() const {
                if (!requires_grad_) {
                    throw std::runtime_error("Variable does not require gradient");
                }
                return grad_;
            }

            // Check if gradient is needed
            bool requires_grad() const {
                return requires_grad_;
            }

            // Set requires_grad flag
            void set_requires_grad(bool requires_grad) {
                requires_grad_ = requires_grad;
                if (requires_grad && grad_.size() == 0) {
                    grad_ = Tensor<T>(data_.shape());
                    grad_.fill(T(0));
                }
            }

            // Get gradient function node
            std::shared_ptr<Node<T>> grad_fn() const {
                return grad_fn_;
            }

            // Set gradient function node
            void set_grad_fn(std::shared_ptr<Node<T>> grad_fn) {
                grad_fn_ = grad_fn;
            }

            // Reset gradients to zero
            void zero_grad() {
                if (requires_grad_ && grad_.size() > 0) {
                    grad_.fill(T(0));
                }
            }

            // Accumulate gradient (used during backpropagation)
            void accumulate_grad(const Tensor<T>& grad, GradientMode mode = GradientMode::ACCUMULATE) {
                if (!requires_grad_) {
                    return;
                }

                if (grad_.size() == 0) {
                    grad_ = grad.clone();
                }
                else {
                    if (mode == GradientMode::ACCUMULATE) {
                        // Accumulate gradients (standard behavior)
                        for (size_t i = 0; i < grad_.size(); ++i) {
                            grad_.data()[i] += grad.data()[i];
                        }
                    }
                    else {
                        // Replace gradients (useful for optimizers)
                        for (size_t i = 0; i < grad_.size(); ++i) {
                            grad_.data()[i] = grad.data()[i];
                        }
                    }
                }
            }

            // Backward pass starting from this variable
            void backward(const Tensor<T>& grad = Tensor<T>()) {
                if (!requires_grad_) {
                    throw std::runtime_error("Cannot backward on a variable that doesn't require gradients");
                }

                // If no gradient is provided, create default gradient for scalar outputs
                Tensor<T> grad_output;
                if (grad.size() == 0) {
                    if (data_.size() != 1) {
                        throw std::runtime_error("Gradient can be implicitly created only for scalar outputs");
                    }
                    grad_output = Tensor<T>(data_.shape(), T(1));
                }
                else {
                    if (grad.shape() != data_.shape()) {
                        throw std::invalid_argument("Gradient shape must match variable shape");
                    }
                    grad_output = grad;
                }

                // Start backward pass
                ComputationGraph<T>::backward(*this, grad_output);
            }

            // Detach variable from computation graph (creates a copy with no gradient tracking)
            Variable detach() const {
                return Variable(data_.clone(), false);
            }

            // Conversion to Tensor for easy use
            operator Tensor<T>() const {
                return data_;
            }

        private:
            Tensor<T> data_;            // The tensor data
            bool requires_grad_;        // Whether to track gradients
            std::shared_ptr<Node<T>> grad_fn_;  // Operation that created this variable
            Tensor<T> grad_;            // Accumulated gradient

            // Allow ComputationGraph to access private members
            friend class ComputationGraph<T>;
        };

        // Manages the computation graph and backpropagation
        template<typename T>
        class ComputationGraph {
        public:
            // Initialize a new computation graph
            static void init() {
                is_recording_ = true;
                nodes_.clear();
            }

            // Record an operation in the graph
            static void record_operation(std::shared_ptr<Node<T>> node) {
                if (is_recording_) {
                    nodes_.push_back(node);
                }
            }

            // Start recording operations
            static void enable_recording() {
                is_recording_ = true;
            }

            // Stop recording operations
            static void disable_recording() {
                is_recording_ = false;
            }

            // Get recording status
            static bool is_recording() {
                return is_recording_;
            }

            // Perform backward pass through the graph
            static void backward(Variable<T>& root, const Tensor<T>& grad_output) {
                // First, accumulate the output gradient
                root.accumulate_grad(grad_output);

                // Then, traverse the graph and compute gradients
                if (root.grad_fn()) {
                    // Create a set of visited nodes to avoid duplicates
                    std::unordered_set<Node<T>*> visited;

                    // Queue of nodes to process
                    std::vector<std::pair<std::shared_ptr<Node<T>>, Tensor<T>>> queue;
                    queue.push_back({ root.grad_fn(), grad_output });

                    // Process the queue
                    while (!queue.empty()) {
                        auto [node, grad] = queue.back();
                        queue.pop_back();

                        // Skip if already visited
                        if (visited.find(node.get()) != visited.end()) {
                            continue;
                        }

                        // Mark as visited
                        visited.insert(node.get());

                        // Compute gradients for inputs
                        node->backward(grad);

                        // Add parent nodes to the queue
                        for (const auto& parent : node->parents()) {
                            if (parent->requires_grad() && parent->grad_fn()) {
                                queue.push_back({ parent->grad_fn(), parent->grad() });
                            }
                        }
                    }
                }
            }

        private:
            static bool is_recording_;
            static std::vector<std::shared_ptr<Node<T>>> nodes_;
        };

        // Initialize static members
        template<typename T>
        bool ComputationGraph<T>::is_recording_ = true;

        template<typename T>
        std::vector<std::shared_ptr<Node<T>>> ComputationGraph<T>::nodes_;

        // Operation nodes for common math operations

        // Addition operation
        template<typename T>
        class AddNode : public Node<T> {
        public:
            AddNode(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b)
                : a_(a), b_(b) {
                this->set_parents({ a, b });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of a + b is the same for both inputs
                if (a_->requires_grad()) {
                    a_->accumulate_grad(grad_output);
                }

                if (b_->requires_grad()) {
                    b_->accumulate_grad(grad_output);
                }
            }

            std::string op_name() const override {
                return "Add";
            }

        private:
            std::shared_ptr<Variable<T>> a_;
            std::shared_ptr<Variable<T>> b_;
        };

        // Subtraction operation
        template<typename T>
        class SubtractNode : public Node<T> {
        public:
            SubtractNode(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b)
                : a_(a), b_(b) {
                this->set_parents({ a, b });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of a - b is positive for a, negative for b
                if (a_->requires_grad()) {
                    a_->accumulate_grad(grad_output);
                }

                if (b_->requires_grad()) {
                    // Gradient for b is negated
                    Tensor<T> neg_grad = grad_output.clone();
                    for (size_t i = 0; i < neg_grad.size(); ++i) {
                        neg_grad.data()[i] = -neg_grad.data()[i];
                    }
                    b_->accumulate_grad(neg_grad);
                }
            }

            std::string op_name() const override {
                return "Subtract";
            }

        private:
            std::shared_ptr<Variable<T>> a_;
            std::shared_ptr<Variable<T>> b_;
        };

        // Multiplication operation (element-wise)
        template<typename T>
        class MultiplyNode : public Node<T> {
        public:
            MultiplyNode(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b)
                : a_(a), b_(b) {
                this->set_parents({ a, b });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of a * b:
                // da = grad_output * b
                // db = grad_output * a

                if (a_->requires_grad()) {
                    Tensor<T> grad_a(a_->data().shape());

                    // Compute gradient for a
                    for (size_t i = 0; i < grad_a.size(); ++i) {
                        grad_a.data()[i] = grad_output.data()[i] * b_->data().data()[i];
                    }

                    a_->accumulate_grad(grad_a);
                }

                if (b_->requires_grad()) {
                    Tensor<T> grad_b(b_->data().shape());

                    // Compute gradient for b
                    for (size_t i = 0; i < grad_b.size(); ++i) {
                        grad_b.data()[i] = grad_output.data()[i] * a_->data().data()[i];
                    }

                    b_->accumulate_grad(grad_b);
                }
            }

            std::string op_name() const override {
                return "Multiply";
            }

        private:
            std::shared_ptr<Variable<T>> a_;
            std::shared_ptr<Variable<T>> b_;
        };

        // Division operation (element-wise)
        template<typename T>
        class DivideNode : public Node<T> {
        public:
            DivideNode(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b)
                : a_(a), b_(b) {
                this->set_parents({ a, b });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of a / b:
                // da = grad_output / b
                // db = -grad_output * a / (b * b)

                if (a_->requires_grad()) {
                    Tensor<T> grad_a(a_->data().shape());

                    // Compute gradient for a
                    for (size_t i = 0; i < grad_a.size(); ++i) {
                        grad_a.data()[i] = grad_output.data()[i] / b_->data().data()[i];
                    }

                    a_->accumulate_grad(grad_a);
                }

                if (b_->requires_grad()) {
                    Tensor<T> grad_b(b_->data().shape());

                    // Compute gradient for b
                    for (size_t i = 0; i < grad_b.size(); ++i) {
                        T b_val = b_->data().data()[i];
                        grad_b.data()[i] = -grad_output.data()[i] * a_->data().data()[i] / (b_val * b_val);
                    }

                    b_->accumulate_grad(grad_b);
                }
            }

            std::string op_name() const override {
                return "Divide";
            }

        private:
            std::shared_ptr<Variable<T>> a_;
            std::shared_ptr<Variable<T>> b_;
        };

        // Matrix multiplication operation
        template<typename T>
        class MatMulNode : public Node<T> {
        public:
            MatMulNode(std::shared_ptr<Variable<T>> a, std::shared_ptr<Variable<T>> b)
                : a_(a), b_(b) {
                this->set_parents({ a, b });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of a @ b:
                // da = grad_output @ b.T
                // db = a.T @ grad_output

                if (a_->requires_grad()) {
                    // Transpose b for gradient of a
                    Tensor<T> b_t = transpose(b_->data());
                    Tensor<T> grad_a = matmul(grad_output, b_t);

                    a_->accumulate_grad(grad_a);
                }

                if (b_->requires_grad()) {
                    // Transpose a for gradient of b
                    Tensor<T> a_t = transpose(a_->data());
                    Tensor<T> grad_b = matmul(a_t, grad_output);

                    b_->accumulate_grad(grad_b);
                }
            }

            std::string op_name() const override {
                return "MatMul";
            }

        private:
            std::shared_ptr<Variable<T>> a_;
            std::shared_ptr<Variable<T>> b_;
        };

        // Scalar Multiply Node
        template<typename T>
        class ScalarMultiplyNode : public Node<T> {
        public:
            ScalarMultiplyNode(std::shared_ptr<Variable<T>> input, T scalar)
                : input_(input), scalar_(scalar) {
                this->set_parents({ input });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of x * c is c * grad_output
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        grad_input.data()[i] = scalar_ * grad_output.data()[i];
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "ScalarMultiply";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            T scalar_;
        };

        // Scalar Addition Node
        template<typename T>
        class ScalarAddNode : public Node<T> {
        public:
            ScalarAddNode(std::shared_ptr<Variable<T>> input, T scalar)
                : input_(input), scalar_(scalar) {
                this->set_parents({ input });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of x + c is gradient of output
                if (input_->requires_grad()) {
                    input_->accumulate_grad(grad_output);
                }
            }

            std::string op_name() const override {
                return "ScalarAdd";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            T scalar_;
        };

        // Sum operation (reduce along dimensions)
        template<typename T>
        class SumNode : public Node<T> {
        public:
            SumNode(std::shared_ptr<Variable<T>> input, const std::vector<size_t>& dims, bool keepdims)
                : input_(input), dims_(dims), keepdims_(keepdims), input_shape_(input->data().shape()) {
                this->set_parents({ input });
            }

            void backward(const Tensor<T>& grad_output) override {
                if (input_->requires_grad()) {
                    // Expand gradient to match original shape
                    Tensor<T> expanded_grad = expand_gradient(grad_output);

                    // Accumulate gradient for input
                    input_->accumulate_grad(expanded_grad);
                }
            }

            std::string op_name() const override {
                return "Sum";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            std::vector<size_t> dims_;
            bool keepdims_;
            std::vector<size_t> input_shape_;

            // Expand gradient to match original input shape
            Tensor<T> expand_gradient(const Tensor<T>& grad) {
                if (keepdims_ && grad.shape() == input_shape_) {
                    return grad;
                }

                // Create result tensor with input shape
                Tensor<T> result(input_shape_);
                result.fill(T(0));

                // Handle each element in the gradient
                std::vector<size_t> grad_idx(grad.ndim(), 0);
                bool done = false;

                while (!done) {
                    // Map gradient index to expanded index
                    std::vector<size_t> expanded_idx = map_index(grad_idx, grad.shape(), input_shape_);

                    // Copy value from gradient to expanded tensor
                    T value = grad.at(grad_idx);

                    // Add the value to the corresponding location in the expanded tensor
                    // This handles the "spreading" of gradient for summed dimensions
                    add_value_to_all(result, expanded_idx, value, dims_);

                    // Increment indices
                    for (int i = static_cast<int>(grad_idx.size()) - 1; i >= 0; --i) {
                        grad_idx[i]++;
                        if (grad_idx[i] < grad.shape()[i]) {
                            break;
                        }
                        grad_idx[i] = 0;
                        if (i == 0) {
                            done = true;
                        }
                    }
                }

                return result;
            }

            // Map gradient index to expanded index (for broadcasting)
            std::vector<size_t> map_index(const std::vector<size_t>& grad_idx,
                const std::vector<size_t>& grad_shape,
                const std::vector<size_t>& input_shape) {
                std::vector<size_t> result(input_shape.size(), 0);

                // Handle reduced dimensions and broadcasting
                size_t grad_dim = 0;
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    // Check if this dimension was reduced
                    bool is_reduced_dim = std::find(dims_.begin(), dims_.end(), i) != dims_.end();

                    if (!is_reduced_dim) {
                        // This dimension wasn't reduced, copy the index
                        if (grad_dim < grad_idx.size()) {
                            result[i] = grad_idx[grad_dim++];
                        }
                    }
                    // For reduced dimensions, the index stays at 0
                }

                return result;
            }

            // Add value to all corresponding positions in reduced dimensions
            void add_value_to_all(Tensor<T>& tensor, const std::vector<size_t>& base_idx,
                T value, const std::vector<size_t>& reduced_dims) {
                // If no dimensions were reduced, just add at the specific index
                if (reduced_dims.empty()) {
                    tensor.at(base_idx) += value;
                    return;
                }

                // Otherwise, we need to add the value to all positions
                // across the reduced dimensions

                // Make a copy of the base index we can modify
                std::vector<size_t> idx = base_idx;

                // Define a recursive helper to handle all combinations
                std::function<void(size_t)> add_recursive = [&](size_t dim_index) {
                    // Base case: we've handled all reduced dimensions
                    if (dim_index >= reduced_dims.size()) {
                        tensor.at(idx) += value;
                        return;
                    }

                    // Get the current reduced dimension
                    size_t dim = reduced_dims[dim_index];

                    // Save the original index value
                    size_t original_idx = idx[dim];

                    // Iterate through all values in this dimension
                    for (size_t i = 0; i < input_shape_[dim]; ++i) {
                        idx[dim] = i;
                        add_recursive(dim_index + 1);
                    }

                    // Restore the original index value
                    idx[dim] = original_idx;
                    };

                // Start the recursive process
                add_recursive(0);
            }
        };

        // Log Node
        template<typename T>
        class LogNode : public Node<T> {
        public:
            LogNode(std::shared_ptr<Variable<T>> input)
                : input_(input) {
                this->set_parents({ input });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of log(x) is 1/x * grad_output
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        T x = input_->data().data()[i];
                        grad_input.data()[i] = grad_output.data()[i] / x;
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "Log";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
        };

        // Exp Node
        template<typename T>
        class ExpNode : public Node<T> {
        public:
            ExpNode(std::shared_ptr<Variable<T>> input)
                : input_(input), output_data_() {
                this->set_parents({ input });
            }

            void set_output_data(const Tensor<T>& output) {
                output_data_ = output;
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of exp(x) is exp(x) * grad_output
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        // Use the cached output value for efficiency
                        grad_input.data()[i] = output_data_.data()[i] * grad_output.data()[i];
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "Exp";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            Tensor<T> output_data_;  // Cache the output for backward pass
        };

        // Power Node
        template<typename T>
        class PowerNode : public Node<T> {
        public:
            PowerNode(std::shared_ptr<Variable<T>> base, T exponent)
                : base_(base), exponent_(exponent) {
                this->set_parents({ base });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of x^n is n * x^(n-1) * grad_output
                if (base_->requires_grad()) {
                    Tensor<T> grad_base(base_->data().shape());

                    for (size_t i = 0; i < grad_base.size(); ++i) {
                        T x = base_->data().data()[i];
                        grad_base.data()[i] = exponent_ * std::pow(x, exponent_ - 1) * grad_output.data()[i];
                    }

                    base_->accumulate_grad(grad_base);
                }
            }

            std::string op_name() const override {
                return "Power";
            }

        private:
            std::shared_ptr<Variable<T>> base_;
            T exponent_;
        };

        // ReLU Node
        template<typename T>
        class ReLUNode : public Node<T> {
        public:
            ReLUNode(std::shared_ptr<Variable<T>> input)
                : input_(input) {
                this->set_parents({ input });
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of ReLU(x) is 1 if x > 0, 0 otherwise
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        T x = input_->data().data()[i];
                        grad_input.data()[i] = x > 0 ? grad_output.data()[i] : 0;
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "ReLU";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
        };

        // Sigmoid Node
        template<typename T>
        class SigmoidNode : public Node<T> {
        public:
            SigmoidNode(std::shared_ptr<Variable<T>> input)
                : input_(input), output_data_() {
                this->set_parents({ input });
            }

            void set_output_data(const Tensor<T>& output) {
                output_data_ = output;
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)) * grad_output
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        T sigmoid_x = output_data_.data()[i];
                        grad_input.data()[i] = sigmoid_x * (1 - sigmoid_x) * grad_output.data()[i];
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "Sigmoid";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            Tensor<T> output_data_;  // Cache the output for backward pass
        };

        // Tanh Node
        template<typename T>
        class TanhNode : public Node<T> {
        public:
            TanhNode(std::shared_ptr<Variable<T>> input)
                : input_(input), output_data_() {
                this->set_parents({ input });
            }

            void set_output_data(const Tensor<T>& output) {
                output_data_ = output;
            }

            void backward(const Tensor<T>& grad_output) override {
                // Gradient of tanh(x) is (1 - tanh(x)^2) * grad_output
                if (input_->requires_grad()) {
                    Tensor<T> grad_input(input_->data().shape());

                    for (size_t i = 0; i < grad_input.size(); ++i) {
                        T tanh_x = output_data_.data()[i];
                        grad_input.data()[i] = (1 - tanh_x * tanh_x) * grad_output.data()[i];
                    }

                    input_->accumulate_grad(grad_input);
                }
            }

            std::string op_name() const override {
                return "Tanh";
            }

        private:
            std::shared_ptr<Variable<T>> input_;
            Tensor<T> output_data_;  // Cache the output for backward pass
        };

        // -------------------- OPERATOR OVERLOADS --------------------

        // Addition
        template<typename T>
        std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& a,
            const std::shared_ptr<Variable<T>>& b) {
            // Perform the addition
            Tensor<T> result_data = a->data() + b->data();

            // Create the result variable
            bool requires_grad = a->requires_grad() || b->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<AddNode<T>>(a, b);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Subtraction
        template<typename T>
        std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& a,
            const std::shared_ptr<Variable<T>>& b) {
            // Perform the subtraction
            Tensor<T> result_data = a->data() - b->data();

            // Create the result variable
            bool requires_grad = a->requires_grad() || b->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<SubtractNode<T>>(a, b);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Element-wise multiplication
        template<typename T>
        std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& a,
            const std::shared_ptr<Variable<T>>& b) {
            // Perform the multiplication
            Tensor<T> result_data = a->data() * b->data();

            // Create the result variable
            bool requires_grad = a->requires_grad() || b->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<MultiplyNode<T>>(a, b);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Division
        template<typename T>
        std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& a,
            const std::shared_ptr<Variable<T>>& b) {
            // Perform the division
            Tensor<T> result_data = a->data() / b->data();

            // Create the result variable
            bool requires_grad = a->requires_grad() || b->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<DivideNode<T>>(a, b);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Scalar addition
        template<typename T>
        std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& a, T scalar) {
            // Perform the addition
            Tensor<T> result_data = a->data() + scalar;

            // Create the result variable
            bool requires_grad = a->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<ScalarAddNode<T>>(a, scalar);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Scalar multiplication
        template<typename T>
        std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& a, T scalar) {
            // Perform the multiplication
            Tensor<T> result_data = a->data() * scalar;

            // Create the result variable
            bool requires_grad = a->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<ScalarMultiplyNode<T>>(a, scalar);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Matrix multiplication
        template<typename T>
        std::shared_ptr<Variable<T>> matmul(const std::shared_ptr<Variable<T>>& a,
            const std::shared_ptr<Variable<T>>& b) {
            // Perform the matrix multiplication
            Tensor<T> result_data = tensor::matmul(a->data(), b->data());

            // Create the result variable
            bool requires_grad = a->requires_grad() || b->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<MatMulNode<T>>(a, b);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Exponential function
        template<typename T>
        std::shared_ptr<Variable<T>> exp(const std::shared_ptr<Variable<T>>& input) {
            // Perform the exponential operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = std::exp(input->data().data()[i]);
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<ExpNode<T>>(input);
                node->set_output_data(result_data);  // Cache output for backward pass
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Logarithm function
        template<typename T>
        std::shared_ptr<Variable<T>> log(const std::shared_ptr<Variable<T>>& input) {
            // Perform the logarithm operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = std::log(input->data().data()[i]);
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<LogNode<T>>(input);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Power function
        template<typename T>
        std::shared_ptr<Variable<T>> pow(const std::shared_ptr<Variable<T>>& input, T exponent) {
            // Perform the power operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = std::pow(input->data().data()[i], exponent);
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<PowerNode<T>>(input, exponent);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // ReLU activation function
        template<typename T>
        std::shared_ptr<Variable<T>> relu(const std::shared_ptr<Variable<T>>& input) {
            // Perform the ReLU operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = std::max(T(0), input->data().data()[i]);
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<ReLUNode<T>>(input);
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Sigmoid activation function
        template<typename T>
        std::shared_ptr<Variable<T>> sigmoid(const std::shared_ptr<Variable<T>>& input) {
            // Perform the sigmoid operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = T(1) / (T(1) + std::exp(-input->data().data()[i]));
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<SigmoidNode<T>>(input);
                node->set_output_data(result_data);  // Cache output for backward pass
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // Tanh activation function
        template<typename T>
        std::shared_ptr<Variable<T>> tanh(const std::shared_ptr<Variable<T>>& input) {
            // Perform the tanh operation
            Tensor<T> result_data(input->data().shape());

            for (size_t i = 0; i < result_data.size(); ++i) {
                result_data.data()[i] = std::tanh(input->data().data()[i]);
            }

            // Create the result variable
            bool requires_grad = input->requires_grad();
            auto result = std::make_shared<Variable<T>>(result_data, requires_grad);

            if (requires_grad) {
                // Create and record the operation node
                auto node = std::make_shared<TanhNode<T>>(input);
                node->set_output_data(result_data);  // Cache output for backward pass
                result->set_grad_fn(node);
                ComputationGraph<T>::record_operation(node);
            }

            return result;
        }

        // -------------------- HIGHER ORDER AUTOGRAD --------------------

        // Module for second-order derivatives (Hessian)
        template<typename T>
        class HigherOrderAutograd {
        public:
            // Compute gradient of gradient (Hessian-vector product)
            static std::shared_ptr<Variable<T>> grad_grad(
                std::shared_ptr<Variable<T>> output,
                std::shared_ptr<Variable<T>> input,
                const Tensor<T>& v) {

                // First-order backward pass
                output->backward();

                // Save the first-order gradients
                Tensor<T> first_grad = input->grad().clone();

                // Create a new computation graph
                ComputationGraph<T>::init();

                // Create a new variable from the gradient
                auto grad_var = std::make_shared<Variable<T>>(first_grad, true);

                // Second-order backward pass with vector v
                grad_var->backward(v);

                // Return the second-order gradient
                return grad_var;
            }

            // Compute full Hessian matrix
            static Tensor<T> hessian(
                std::shared_ptr<Variable<T>> output,
                std::shared_ptr<Variable<T>> input) {

                // Ensure input is flattened
                size_t n = input->data().size();

                // Create Hessian matrix
                std::vector<size_t> hessian_shape = { n, n };
                Tensor<T> H(hessian_shape);

                // For each dimension i, compute gradient of gradient w.r.t dimension i
                for (size_t i = 0; i < n; ++i) {
                    // Create unit vector
                    Tensor<T> v(input->data().shape());
                    v.fill(T(0));
                    v.data()[i] = T(1);

                    // Compute Hessian-vector product
                    auto hv = grad_grad(output, input, v);

                    // Copy the i-th column of the Hessian
                    for (size_t j = 0; j < n; ++j) {
                        H.at({ j, i }) = hv->grad().data()[j];
                    }
                }

                return H;
            }
        };

        // -------------------- GRADIENT CHECKING --------------------

        // Helper for numerical gradient verification
        template<typename T>
        class GradientChecker {
        public:
            // Check if analytical gradient matches numerical gradient
            static bool check_gradient(
                std::function<std::shared_ptr<Variable<T>>(std::shared_ptr<Variable<T>>)> func,
                std::shared_ptr<Variable<T>> input,
                T epsilon = 1e-5,
                T tolerance = 1e-5) {

                // Get analytical gradient
                auto output = func(input);
                output->backward();
                Tensor<T> analytical_grad = input->grad().clone();

                // Compute numerical gradient
                Tensor<T> numerical_grad(input->data().shape());
                Tensor<T> original_data = input->data().clone();

                for (size_t i = 0; i < input->data().size(); ++i) {
                    // Compute f(x + epsilon)
                    T* data_ptr = input->data().data();
                    data_ptr[i] += epsilon;
                    auto output_plus = func(input);
                    T f_plus = output_plus->data().data()[0];

                    // Compute f(x - epsilon)
                    data_ptr[i] = original_data.data()[i] - epsilon;
                    auto output_minus = func(input);
                    T f_minus = output_minus->data().data()[0];

                    // Restore original data
                    data_ptr[i] = original_data.data()[i];

                    // Compute numerical gradient
                    numerical_grad.data()[i] = (f_plus - f_minus) / (2 * epsilon);
                }

                // Check if gradients match
                T max_diff = 0;
                for (size_t i = 0; i < analytical_grad.size(); ++i) {
                    T diff = std::abs(analytical_grad.data()[i] - numerical_grad.data()[i]);
                    max_diff = std::max(max_diff, diff);
                }

                return max_diff <= tolerance;
            }
        };

        // -------------------- MEMORY OPTIMIZATION --------------------

        // In-place operations for memory optimization
        template<typename T>
        class InPlaceOps {
        public:
            // In-place addition (a += b)
            static void add_(std::shared_ptr<Variable<T>>& a, const std::shared_ptr<Variable<T>>& b) {
                if (a->requires_grad() || b->requires_grad()) {
                    throw std::runtime_error("In-place operations can't be used on variables that require gradients");
                }

                // Perform in-place addition
                for (size_t i = 0; i < a->data().size(); ++i) {
                    a->data().data()[i] += b->data().data()[i];
                }
            }

            // In-place scalar multiplication (a *= scalar)
            static void mul_(std::shared_ptr<Variable<T>>& a, T scalar) {
                if (a->requires_grad()) {
                    throw std::runtime_error("In-place operations can't be used on variables that require gradients");
                }

                // Perform in-place multiplication
                for (size_t i = 0; i < a->data().size(); ++i) {
                    a->data().data()[i] *= scalar;
                }
            }

            // In-place ReLU
            static void relu_(std::shared_ptr<Variable<T>>& a) {
                if (a->requires_grad()) {
                    throw std::runtime_error("In-place operations can't be used on variables that require gradients");
                }

                // Perform in-place ReLU
                for (size_t i = 0; i < a->data().size(); ++i) {
                    a->data().data()[i] = std::max(T(0), a->data().data()[i]);
                }
            }
        };

    } // namespace autograd
} // namespace tensor

#endif // AUTOGRAD_H