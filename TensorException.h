// TensorException.h - v0.1.0
// Custom exceptions for tensor operations

#ifndef TENSOR_EXCEPTION_H
#define TENSOR_EXCEPTION_H

#include <stdexcept>
#include <string>

namespace tensor {

    // Base class for all tensor exceptions
    class TensorException : public std::runtime_error {
    public:
        explicit TensorException(const std::string& message)
            : std::runtime_error("TensorException: " + message) {}
    };

    // Shape-related exceptions
    class ShapeException : public TensorException {
    public:
        explicit ShapeException(const std::string& message)
            : TensorException("Shape error: " + message) {
        }
    };

    // Index-related exceptions
    class IndexException : public TensorException {
    public:
        explicit IndexException(const std::string& message)
            : TensorException("Index error: " + message) {
        }
    };

    // Memory-related exceptions
    class MemoryException : public TensorException {
    public:
        explicit MemoryException(const std::string& message)
            : TensorException("Memory error: " + message) {
        }
    };

    // Device-related exceptions
    class DeviceException : public TensorException {
    public:
        explicit DeviceException(const std::string& message)
            : TensorException("Device error: " + message) {
        }
    };

    // Type-related exceptions
    class TypeError : public TensorException {
    public:
        explicit TypeError(const std::string& message)
            : TensorException("Type error: " + message) {
        }
    };

} // namespace tensor

#endif // TENSOR_EXCEPTION_H