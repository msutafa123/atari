// TensorGrad.h - v0.2.0
// Automatic differentiation support for tensors

#ifndef TENSOR_GRAD_H
#define TENSOR_GRAD_H

#include "Tensor.h"
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>

namespace tensor {

    template<typename T>
    class TensorGrad;

    // Forward operation record for backward pass
    template<typename T>
    struct GradOperation {
        virtual ~GradOperation() = default;

        // Compute gradients for inputs from gradient of output
        virtual void backward(const TensorGrad<T>& grad_output) = 0;

        // Name of the operation (for debugging)
        virtual std::string name() const = 0;
    };

    // Computation graph for tracking operations
    template<typename T>
    class ComputationGraph {
    public:
        // Get singleton instance
        static ComputationGraph& instance() {
            static ComputationGraph instance;
            return instance;
        }

        // Record an operation in the graph
        void record_operation(std::shared_ptr<GradOperation<T>> op) {
            operations_.push_back(op);
        }

        // Clear the graph (used after backward pass)
        void clear() {
            operations_.clear();
        }

        // Get all recorded operations
        const std::vector<std::shared_ptr<GradOperation<T>>>& operations() const {
            return operations_;
        }

    private:
        std::vector<std::shared_ptr<GradOperation<T>>> operations_;

        // Private constructor for singleton
        ComputationGraph() = default;

        // Prevent copying
        ComputationGraph(const ComputationGraph&) = delete;
        ComputationGraph& operator=(const ComputationGraph&) = delete;
    };

    template<typename T>
    class TensorGrad : public Tensor<T> {
    public:
        // Constructors that mirror Tensor constructors
        TensorGrad() : Tensor<T>(), requires_grad_(false), grad_fn_(nullptr) {}

        explicit TensorGrad(const TensorShape& shape, bool requires_grad = false)
            : Tensor<T>(shape), requires_grad_(requires_grad), grad_fn_(nullptr) {}

        TensorGrad(const TensorShape& shape, T value, bool requires_grad = false)
            : Tensor<T>(shape, value), requires_grad_(requires_grad), grad_fn_(nullptr) {}

        // Copy constructor from Tensor
        TensorGrad(const Tensor<T>& tensor, bool requires_grad = false)
            : Tensor<T>(tensor), requires_grad_(requires_grad), grad_fn_(nullptr) {}

        // Copy constructor from TensorGrad
        TensorGrad(const TensorGrad& other)
            : Tensor<T>(other),
            requires_grad_(other.requires_grad_),
            grad_fn_(other.grad_fn_),
            grad_tensor_(other.grad_tensor_) {
        }

        // Move constructor
        TensorGrad(TensorGrad&& other) noexcept
            : Tensor<T>(std::move(other)),
            requires_grad_(other.requires_grad_),
            grad_fn_(std::move(other.grad_fn_)),
            grad_tensor_(std::move(other.grad_tensor_)) {

            other.requires_grad_ = false;
        }

        // Copy assignment
        TensorGrad& operator=(const TensorGrad& other) {
            if (this != &other) {
                Tensor<T>::operator=(other);
                requires_grad_ = other.requires_grad_;
                grad_fn_ = other.grad_fn_;
                grad_tensor_ = other.grad_tensor_;
            }
            return *this;
        }

        // Move assignment
        TensorGrad& operator=(TensorGrad&& other) noexcept {
            if (this != &other) {
                Tensor<T>::operator=(std::move(other));
                requires_grad_ = other.requires_grad_;
                grad_fn_ = std::move(other.grad_fn_);
                grad_tensor_ = std::move(other.grad_tensor_);

                other.requires_grad_ = false;
            }
            return *this;
        }

        // Gradient related functions
        bool requires_grad() const { return requires_grad_; }

        void set_requires_grad(bool requires_grad) {
            requires_grad_ = requires_grad;
            if (!requires_grad && grad_tensor_) {
                grad_tensor_.reset();
            }
        }

        // Check if tensor has gradient
        bool has_grad() const {
            return grad_tensor_ != nullptr;
        }

        // Get gradient tensor
        TensorGrad<T>& grad() {
            if (!grad_tensor_) {
                grad_tensor_ = std::make_shared<TensorGrad<T>>(this->shape(), T(0));
            }
            return *grad_tensor_;
        }

        const TensorGrad<T>& grad() const {
            if (!grad_tensor_) {
                throw std::runtime_error("Tensor has no gradient");
            }
            return *grad_tensor_;
        }

        // Zero out gradients
        void zero_grad() {
            if (grad_tensor_) {
                grad_tensor_->fill(T(0));
            }
            else if (requires_grad_) {
                grad_tensor_ = std::make_shared<TensorGrad<T>>(this->shape(), T(0));
            }
        }

        // Backward pass
        void backward(const TensorGrad<T>& grad_output = TensorGrad<T>()) {
            if (!requires_grad_) {
                throw std::runtime_error("Tensor does not require gradients");
            }

            // If no gradient is provided, create default gradient
            if (grad_output.size() == 0) {
                // Default gradient is 1.0 for scalar tensors, otherwise error
                if (this->size() != 1) {
                    throw std::runtime_error("Gradient can be implicitly created only for scalar results");
                }

                if (!grad_tensor_) {
                    grad_tensor_ = std::make_shared<TensorGrad<T>>(this->shape(), T(1));
                }
                else {
                    grad_tensor_->fill(T(1));
                }
            }
            else {
                // Check shape compatibility
                if (grad_output.shape() != this->shape()) {
                    throw std::invalid_argument("Gradient shape must match tensor shape");
                }

                // Set gradient
                if (!grad_tensor_) {
                    grad_tensor_ = std::make_shared<TensorGrad<T>>(grad_output);
                }
                else {
                    *grad_tensor_ = grad_output;
                }
            }

            // Propagate gradients backward
            if (grad_fn_) {
                grad_fn_->backward(*grad_tensor_);
            }
        }

        // Set gradient function
        void set_grad_fn(std::shared_ptr<GradOperation<T>> grad_fn) {
            grad_fn_ = grad_fn;
        }

        // Get gradient function
        std::shared_ptr<GradOperation<T>> grad_fn() const {
            return grad_fn_;
        }

        // Detach from computation graph (creates a copy with no grad tracking)
        TensorGrad<T> detach() const {
            TensorGrad<T> result(*this, false);
            return result;
        }

        // Clone tensor with same grad requirements
        TensorGrad<T> clone() const {
            return TensorGrad<T>(*this, requires_grad_);
        }

        // Static methods for creating tensors with gradients
        static TensorGrad<T> ones_like(const TensorGrad<T>& other, bool requires_grad = false) {
            return TensorGrad<T>(Tensor<T>::ones(other.shape()), requires_grad);
        }

        static TensorGrad<T> zeros_like(const TensorGrad<T>& other, bool requires_grad = false) {
            return TensorGrad<T>(Tensor<T>::zeros(other.shape()), requires_grad);
        }

        static TensorGrad<T> random_like(const TensorGrad<T>& other, T min = T(0), T max = T(1),
            bool requires_grad = false) {
            return TensorGrad<T>(Tensor<T>::random(other.shape(), min, max), requires_grad);
        }

    private:
        bool requires_grad_;
        std::shared_ptr<GradOperation<T>> grad_fn_;
        std::shared_ptr<TensorGrad<T>> grad_tensor_;

        friend class ComputationGraph<T>;
    };

    // Basic operations with gradient support
    namespace autograd {

        // Addition operation with gradient tracking
        template<typename T>
        class AddOperation : public GradOperation<T> {
        public:
            AddOperation(TensorGrad<T>& a, TensorGrad<T>& b) : a_(a), b_(b) {}

            void backward(const TensorGrad<T>& grad_output) override {
                // Gradient of addition is passed directly to both inputs
                if (a_.requires_grad()) {
                    if (a_.has_grad()) {
                        // Accumulate gradient if it exists
                        a_.grad() = a_.grad() + grad_output;
                    }
                    else {
                        a_.grad() = grad_output.clone();
                    }

                    // Continue backpropagation if needed
                    if (a_.grad_fn()) {
                        a_.grad_fn()->backward(a_.grad());
                    }
                }

                if (b_.requires_grad()) {
                    if (b_.has_grad()) {
                        // Accumulate gradient if it exists
                        b_.grad() = b_.grad() + grad_output;
                    }
                    else {
                        b_.grad() = grad_output.clone();
                    }

                    // Continue backpropagation if needed
                    if (b_.grad_fn()) {
                        b_.grad_fn()->backward(b_.grad());
                    }
                }
            }

            std::string name() const override {
                return "Add";
            }

        private:
            TensorGrad<T>& a_;
            TensorGrad<T>& b_;
        };

        // Addition with gradient support
        template<typename T>
        TensorGrad<T> add(const TensorGrad<T>& a, const TensorGrad<T>& b) {
            // Perform the actual addition
            TensorGrad<T> result = static_cast<Tensor<T>>(a) + static_cast<Tensor<T>>(b);

            // Set up gradient tracking if needed
            bool requires_grad = a.requires_grad() || b.requires_grad();
            result.set_requires_grad(requires_grad);

            if (requires_grad) {
                // Create non-const references (will be captured in the operation)
                TensorGrad<T>& a_ref = const_cast<TensorGrad<T>&>(a);
                TensorGrad<T>& b_ref = const_cast<TensorGrad<T>&>(b);

                // Create and register the operation
                auto op = std::make_shared<AddOperation<T>>(a_ref, b_ref);
                result.set_grad_fn(op);

                // Record in computation graph
                ComputationGraph<T>::instance().record_operation(op);
            }

            return result;
        }

        // Multiplication operation with gradient tracking
        template<typename T>
        class MulOperation : public GradOperation<T> {
        public:
            MulOperation(TensorGrad<T>& a, TensorGrad<T>& b) : a_(a), b_(b) {}

            void backward(const TensorGrad<T>& grad_output) override {
                // Gradient of a * b:
                // da = grad_output * b
                // db = grad_output * a

                if (a_.requires_grad()) {
                    TensorGrad<T> grad_a = grad_output * static_cast<Tensor<T>>(b_);

                    if (a_.has_grad()) {
                        a_.grad() = a_.grad() + grad_a;
                    }
                    else {
                        a_.grad() = grad_a;
                    }

                    if (a_.grad_fn()) {
                        a_.grad_fn()->backward(a_.grad());
                    }
                }

                if (b_.requires_grad()) {
                    TensorGrad<T> grad_b = grad_output * static_cast<Tensor<T>>(a_);

                    if (b_.has_grad()) {
                        b_.grad() = b_.grad() + grad_b;
                    }
                    else {
                        b_.grad() = grad_b;
                    }

                    if (b_.grad_fn()) {
                        b_.grad_fn()->backward(b_.grad());
                    }
                }
            }

            std::string name() const override {
                return "Multiply";
            }

        private:
            TensorGrad<T>& a_;
            TensorGrad<T>& b_;
        };

        // Multiplication with gradient support
        template<typename T>
        TensorGrad<T> multiply(const TensorGrad<T>& a, const TensorGrad<T>& b) {
            // Perform the actual multiplication
            TensorGrad<T> result = static_cast<Tensor<T>>(a) * static_cast<Tensor<T>>(b);

            // Set up gradient tracking if needed
            bool requires_grad = a.requires_grad() || b.requires_grad();
            result.set_requires_grad(requires_grad);

            if (requires_grad) {
                // Create non-const references (will be captured in the operation)
                TensorGrad<T>& a_ref = const_cast<TensorGrad<T>&>(a);
                TensorGrad<T>& b_ref = const_cast<TensorGrad<T>&>(b);

                // Create and register the operation
                auto op = std::make_shared<MulOperation<T>>(a_ref, b_ref);
                result.set_grad_fn(op);

                // Record in computation graph
                ComputationGraph<T>::instance().record_operation(op);
            }

            return result;
        }

        // More operations would be implemented similarly
        // Here are templates for a few more common operations

        // Subtraction operation with gradient tracking
        template<typename T>
        class SubOperation : public GradOperation<T> {
        public:
            SubOperation(TensorGrad<T>& a, TensorGrad<T>& b) : a_(a), b_(b) {}

            void backward(const TensorGrad<T>& grad_output) override {
                if (a_.requires_grad()) {
                    if (a_.has_grad()) {
                        a_.grad() = a_.grad() + grad_output;
                    }
                    else {
                        a_.grad() = grad_output.clone();
                    }

                    if (a_.grad_fn()) {
                        a_.grad_fn()->backward(a_.grad());
                    }
                }

                if (b_.requires_grad()) {
                    TensorGrad<T> neg_grad = grad_output * T(-1);

                    if (b_.has_grad()) {
                        b_.grad() = b_.grad() + neg_grad;
                    }
                    else {
                        b_.grad() = neg_grad;
                    }

                    if (b_.grad_fn()) {
                        b_.grad_fn()->backward(b_.grad());
                    }
                }
            }

            std::string name() const override {
                return "Subtract";
            }

        private:
            TensorGrad<T>& a_;
            TensorGrad<T>& b_;
        };

        // Division operation with gradient tracking
        template<typename T>
        class DivOperation : public GradOperation<T> {
        public:
            DivOperation(TensorGrad<T>& a, TensorGrad<T>& b) : a_(a), b_(b) {}

            void backward(const TensorGrad<T>& grad_output) override {
                // Gradient of a / b:
                // da = grad_output / b
                // db = -grad_output * a / (b * b)

                if (a_.requires_grad()) {
                    TensorGrad<T> grad_a = grad_output / static_cast<Tensor<T>>(b_);

                    if (a_.has_grad()) {
                        a_.grad() = a_.grad() + grad_a;
                    }
                    else {
                        a_.grad() = grad_a;
                    }

                    if (a_.grad_fn()) {
                        a_.grad_fn()->backward(a_.grad());
                    }
                }

                if (b_.requires_grad()) {
                    TensorGrad<T> b_squared = static_cast<Tensor<T>>(b_) * static_cast<Tensor<T>>(b_);
                    TensorGrad<T> grad_b = -grad_output * static_cast<Tensor<T>>(a_) / b_squared;

                    if (b_.has_grad()) {
                        b_.grad() = b_.grad() + grad_b;
                    }
                    else {
                        b_.grad() = grad_b;
                    }

                    if (b_.grad_fn()) {
                        b_.grad_fn()->backward(b_.grad());
                    }
                }
            }

            std::string name() const override {
                return "Divide";
            }

        private:
            TensorGrad<T>& a_;
            TensorGrad<T>& b_;
        };

        // Matrix multiplication operation with gradient tracking
        template<typename T>
        class MatMulOperation : public GradOperation<T> {
        public:
            MatMulOperation(TensorGrad<T>& a, TensorGrad<T>& b) : a_(a), b_(b) {}

            void backward(const TensorGrad<T>& grad_output) override {
                // Gradient of a @ b:
                // da = grad_output @ b.T
                // db = a.T @ grad_output

                if (a_.requires_grad()) {
                    // Transpose b
                    Tensor<T> b_t = transpose(static_cast<Tensor<T>>(b_));

                    // Compute grad_a = grad_output @ b_t
                    TensorGrad<T> grad_a = matmul(grad_output, b_t);

                    if (a_.has_grad()) {
                        a_.grad() = a_.grad() + grad_a;
                    }
                    else {
                        a_.grad() = grad_a;
                    }

                    if (a_.grad_fn()) {
                        a_.grad_fn()->backward(a_.grad());
                    }
                }

                if (b_.requires_grad()) {
                    // Transpose a
                    Tensor<T> a_t = transpose(static_cast<Tensor<T>>(a_));

                    // Compute grad_b = a_t @ grad_output
                    TensorGrad<T> grad_b = matmul(a_t, grad_output);

                    if (b_.has_grad()) {
                        b_.grad() = b_.grad() + grad_b;
                    }
                    else {
                        b_.grad() = grad_b;
                    }

                    if (b_.grad_fn()) {
                        b_.grad_fn()->backward(b_.grad());
                    }
                }
            }

            std::string name() const override {
                return "MatMul";
            }

        private:
            TensorGrad<T>& a_;
            TensorGrad<T>& b_;
        };

        // Implement additional operations here...
    } // namespace autograd

} // namespace tensor

#endif // TENSOR_GRAD_H