// Tensor.h - v2.0.0
// Modern C++17 tensor implementation with simplified interface

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace tensor {

    // Forward declarations
    template<typename T> class TensorView;

    /**
     * @brief Class representing a multi-dimensional array
     */
    template<typename T>
    class Tensor {
    public:
        // Type traits assertions for valid data types
        static_assert(std::is_arithmetic<T>::value || std::is_same<T, bool>::value,
            "Tensor only supports arithmetic types and bool");

        //==========================================================================
        // CONSTRUCTORS & ASSIGNMENT OPERATORS
        //==========================================================================

        /**
         * @brief Default constructor - creates an empty tensor
         */
        Tensor() : shape_(), data_(nullptr), size_(0) {}

        /**
         * @brief Create a tensor with the given shape, zero-initialized
         */
        explicit Tensor(const std::vector<size_t>& shape)
            : shape_(shape),
            size_(std::accumulate(shape.begin(), shape.end(),
                size_t(1), std::multiplies<size_t>()))
        {
            allocate_memory();
            std::fill(data_.get(), data_.get() + size_, T(0));
        }

        /**
         * @brief Create a tensor with the given shape, initialized to a specific value
         */
        Tensor(const std::vector<size_t>& shape, T value)
            : shape_(shape),
            size_(std::accumulate(shape.begin(), shape.end(),
                size_t(1), std::multiplies<size_t>()))
        {
            allocate_memory();
            std::fill(data_.get(), data_.get() + size_, value);
        }

        /**
         * @brief Create a tensor with the given shape, initialized with a user-provided array
         */
        Tensor(const std::vector<size_t>& shape, const T* data)
            : shape_(shape),
            size_(std::accumulate(shape.begin(), shape.end(),
                size_t(1), std::multiplies<size_t>()))
        {
            allocate_memory();
            if (data) {
                std::copy(data, data + size_, data_.get());
            }
            else {
                std::fill(data_.get(), data_.get() + size_, T(0));
            }
        }

        /**
         * @brief Copy constructor
         */
        Tensor(const Tensor& other)
            : shape_(other.shape_), size_(other.size_)
        {
            allocate_memory();
            std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
        }

        /**
         * @brief Move constructor
         */
        Tensor(Tensor&& other) noexcept
            : shape_(std::move(other.shape_)),
            data_(std::move(other.data_)),
            size_(other.size_)
        {
            other.size_ = 0;
        }

        /**
         * @brief Copy assignment operator
         */
        Tensor& operator=(const Tensor& other)
        {
            if (this != &other) {
                shape_ = other.shape_;
                size_ = other.size_;

                allocate_memory();
                std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
            }
            return *this;
        }

        /**
         * @brief Move assignment operator
         */
        Tensor& operator=(Tensor&& other) noexcept
        {
            if (this != &other) {
                shape_ = std::move(other.shape_);
                data_ = std::move(other.data_);
                size_ = other.size_;

                other.size_ = 0;
            }
            return *this;
        }

        //==========================================================================
        // BASIC PROPERTIES & ACCESSORS
        //==========================================================================

        /**
         * @brief Get the shape of the tensor
         */
        const std::vector<size_t>& shape() const {
            return shape_;
        }

        /**
         * @brief Get the number of dimensions of the tensor
         */
        size_t ndim() const {
            return shape_.size();
        }

        /**
         * @brief Get the total number of elements in the tensor
         */
        size_t size() const {
            return size_;
        }

        /**
         * @brief Get a specific dimension size
         */
        size_t dim(size_t index) const {
            if (index >= shape_.size()) {
                throw std::out_of_range("Dimension index out of range");
            }
            return shape_[index];
        }

        /**
         * @brief Get raw data pointer
         */
        T* data() {
            return data_.get();
        }

        /**
         * @brief Get raw data pointer (const version)
         */
        const T* data() const {
            return data_.get();
        }

        //==========================================================================
        // TENSOR ACCESS AND MANIPULATION
        //==========================================================================

        /**
         * @brief Access element using a vector of indices (bounds-checked)
         */
        T& at(const std::vector<size_t>& indices) {
            size_t offset = calculate_offset(indices);
            return data_.get()[offset];
        }

        /**
         * @brief Access element using a vector of indices (bounds-checked, const version)
         */
        const T& at(const std::vector<size_t>& indices) const {
            size_t offset = calculate_offset(indices);
            return data_.get()[offset];
        }

        /**
         * @brief Access element using initializer list (convenience wrapper)
         */
        T& at(std::initializer_list<size_t> indices) {
            return at(std::vector<size_t>(indices));
        }

        /**
         * @brief Access element using initializer list (convenience wrapper, const version)
         */
        const T& at(std::initializer_list<size_t> indices) const {
            return at(std::vector<size_t>(indices));
        }

        /**
         * @brief Access element in a 1D tensor (optimized)
         */
        T& at(size_t i) {
            if (shape_.size() != 1) {
                throw std::invalid_argument("Expected 1D tensor for this access pattern");
            }
            if (i >= shape_[0]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[i];
        }

        /**
         * @brief Access element in a 1D tensor (optimized, const version)
         */
        const T& at(size_t i) const {
            if (shape_.size() != 1) {
                throw std::invalid_argument("Expected 1D tensor for this access pattern");
            }
            if (i >= shape_[0]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[i];
        }

        /**
         * @brief Access element in a 2D tensor (optimized)
         */
        T& at(size_t i, size_t j) {
            if (shape_.size() != 2) {
                throw std::invalid_argument("Expected 2D tensor for this access pattern");
            }
            if (i >= shape_[0] || j >= shape_[1]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[i * shape_[1] + j];
        }

        /**
         * @brief Access element in a 2D tensor (optimized, const version)
         */
        const T& at(size_t i, size_t j) const {
            if (shape_.size() != 2) {
                throw std::invalid_argument("Expected 2D tensor for this access pattern");
            }
            if (i >= shape_[0] || j >= shape_[1]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[i * shape_[1] + j];
        }

        /**
         * @brief Access element in a 3D tensor (optimized)
         */
        T& at(size_t i, size_t j, size_t k) {
            if (shape_.size() != 3) {
                throw std::invalid_argument("Expected 3D tensor for this access pattern");
            }
            if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[(i * shape_[1] + j) * shape_[2] + k];
        }

        /**
         * @brief Access element in a 3D tensor (optimized, const version)
         */
        const T& at(size_t i, size_t j, size_t k) const {
            if (shape_.size() != 3) {
                throw std::invalid_argument("Expected 3D tensor for this access pattern");
            }
            if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
                throw std::out_of_range("Index out of range");
            }
            return data_.get()[(i * shape_[1] + j) * shape_[2] + k];
        }

        /**
         * @brief Fill tensor with a specific value
         */
        void fill(T value) {
            if (data_) {
                std::fill(data_.get(), data_.get() + size_, value);
            }
        }

        /**
         * @brief Apply a function to each element in the tensor
         */
        Tensor& apply(const std::function<T(const T&)>& func) {
            if (data_) {
                std::transform(data_.get(), data_.get() + size_, data_.get(), func);
            }
            return *this;
        }

        /**
         * @brief Create a deep copy of this tensor
         */
        Tensor clone() const {
            return *this;  // The copy constructor will handle the deep copy
        }

        /**
         * @brief Reshape tensor to a new shape
         */
        Tensor reshape(const std::vector<size_t>& new_shape) const {
            // Calculate new size
            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                size_t(1), std::multiplies<size_t>());

            // Check if shapes are compatible
            if (new_size != size_) {
                throw std::invalid_argument("New shape must have same total size");
            }

            // Create new tensor with the new shape
            Tensor result(new_shape);
            std::copy(data_.get(), data_.get() + size_, result.data_.get());

            return result;
        }

        /**
         * @brief Reshape using initializer list (convenience wrapper)
         */
        Tensor reshape(std::initializer_list<size_t> new_shape) const {
            return reshape(std::vector<size_t>(new_shape));
        }

        //==========================================================================
        // BASIC ARITHMETIC OPERATIONS
        //==========================================================================

        /**
         * @brief Add another tensor to this one (element-wise)
         */
        Tensor& operator+=(const Tensor& other) {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Cannot add tensors with different shapes");
            }

            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] += other.data_.get()[i];
            }

            return *this;
        }

        /**
         * @brief Subtract another tensor from this one (element-wise)
         */
        Tensor& operator-=(const Tensor& other) {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Cannot subtract tensors with different shapes");
            }

            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] -= other.data_.get()[i];
            }

            return *this;
        }

        /**
         * @brief Multiply this tensor by another (element-wise)
         */
        Tensor& operator*=(const Tensor& other) {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Cannot multiply tensors with different shapes");
            }

            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] *= other.data_.get()[i];
            }

            return *this;
        }

        /**
         * @brief Divide this tensor by another (element-wise)
         */
        Tensor& operator/=(const Tensor& other) {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Cannot divide tensors with different shapes");
            }

            for (size_t i = 0; i < size_; ++i) {
                if (other.data_.get()[i] == T(0)) {
                    throw std::domain_error("Division by zero");
                }
                data_.get()[i] /= other.data_.get()[i];
            }

            return *this;
        }

        /**
         * @brief Add a scalar to this tensor (element-wise)
         */
        Tensor& operator+=(T scalar) {
            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] += scalar;
            }
            return *this;
        }

        /**
         * @brief Subtract a scalar from this tensor (element-wise)
         */
        Tensor& operator-=(T scalar) {
            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] -= scalar;
            }
            return *this;
        }

        /**
         * @brief Multiply this tensor by a scalar (element-wise)
         */
        Tensor& operator*=(T scalar) {
            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] *= scalar;
            }
            return *this;
        }

        /**
         * @brief Divide this tensor by a scalar (element-wise)
         */
        Tensor& operator/=(T scalar) {
            if (scalar == T(0)) {
                throw std::domain_error("Division by zero");
            }
            for (size_t i = 0; i < size_; ++i) {
                data_.get()[i] /= scalar;
            }
            return *this;
        }

        //==========================================================================
        // STATIC FACTORY METHODS
        //==========================================================================

        /**
         * @brief Create a tensor filled with ones
         */
        static Tensor ones(const std::vector<size_t>& shape) {
            return Tensor(shape, T(1));
        }

        /**
         * @brief Create a tensor filled with zeros
         */
        static Tensor zeros(const std::vector<size_t>& shape) {
            return Tensor(shape, T(0));
        }

        /**
         * @brief Create a tensor with random values
         */
        static Tensor random(const std::vector<size_t>& shape, T min = T(0), T max = T(1)) {
            Tensor result(shape);
            std::random_device rd;
            std::mt19937 gen(rd());

            if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(min, max);
                for (size_t i = 0; i < result.size_; ++i) {
                    result.data_.get()[i] = dist(gen);
                }
            }
            else {
                std::uniform_real_distribution<T> dist(min, max);
                for (size_t i = 0; i < result.size_; ++i) {
                    result.data_.get()[i] = dist(gen);
                }
            }

            return result;
        }

        /**
         * @brief Create an identity matrix (2D tensor with ones on the diagonal)
         */
        static Tensor eye(size_t n) {
            Tensor result({ n, n }, T(0));
            for (size_t i = 0; i < n; ++i) {
                result.at(i, i) = T(1);
            }
            return result;
        }

        /**
         * @brief Create a tensor with values from start to end with step
         */
        static Tensor arange(T start, T end, T step = T(1)) {
            if (step == T(0)) {
                throw std::invalid_argument("Step cannot be zero");
            }

            // Calculate size
            size_t size;
            if constexpr (std::is_integral<T>::value) {
                // For integers, ensure proper handling of the endpoint
                if (step > T(0)) {
                    size = static_cast<size_t>((end - start + step - T(1)) / step);
                }
                else {
                    size = static_cast<size_t>((start - end - step - T(1)) / -step);
                }
            }
            else {
                // For floating point, count number of steps
                size = static_cast<size_t>(std::ceil((end - start) / step));
            }

            Tensor result({ size });
            for (size_t i = 0; i < size; ++i) {
                result.data_.get()[i] = start + static_cast<T>(i) * step;
            }

            return result;
        }

        /**
         * @brief Create a tensor with linearly spaced values
         */
        static Tensor linspace(T start, T end, size_t num) {
            if (num < 2) {
                throw std::invalid_argument("Number of points must be at least 2");
            }

            Tensor result({ num });
            T step = (end - start) / static_cast<T>(num - 1);

            for (size_t i = 0; i < num; ++i) {
                result.data_.get()[i] = start + static_cast<T>(i) * step;
            }

            return result;
        }

        /**
         * @brief Convert tensor to string representation
         */
        std::string to_string(int precision = 4) const {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision);

            oss << "Tensor(shape=[";
            for (size_t i = 0; i < shape_.size(); ++i) {
                oss << shape_[i];
                if (i < shape_.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], data=";

            // Print tensor data
            print_data(oss, 0, {}, precision);

            oss << ")";
            return oss.str();
        }

    private:
        std::vector<size_t> shape_;      // Dimensions of the tensor
        std::shared_ptr<T[]> data_;      // Shared pointer to data
        size_t size_;                    // Total number of elements

        /**
         * @brief Allocate memory for tensor data
         */
        void allocate_memory() {
            if (size_ > 0) {
                data_ = std::make_shared<T[]>(size_);
            }
            else {
                data_ = nullptr;
            }
        }

        /**
         * @brief Calculate linear offset from multidimensional indices
         */
        size_t calculate_offset(const std::vector<size_t>& indices) const {
            if (indices.size() != shape_.size()) {
                throw std::invalid_argument(
                    "Number of indices (" + std::to_string(indices.size()) +
                    ") doesn't match tensor dimensions (" + std::to_string(shape_.size()) + ")");
            }

            size_t offset = 0;
            size_t stride = 1;

            // Calculate row-major (C-style) offset
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range("Index " + std::to_string(indices[i]) +
                        " is out of bounds for dimension " + std::to_string(i) +
                        " with size " + std::to_string(shape_[i]));
                }
                offset += indices[i] * stride;
                stride *= shape_[i];
            }

            return offset;
        }

        /**
         * @brief Recursively print tensor data
         */
        void print_data(std::ostream& os, size_t dim, std::vector<size_t> indices, int precision) const {
            // Base case: print value
            if (dim == shape_.size()) {
                os << at(indices);
                return;
            }

            os << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                // Add current dimension index
                indices.push_back(i);

                // Recursively print next dimension
                print_data(os, dim + 1, indices, precision);

                // Remove current dimension index for next iteration
                indices.pop_back();

                // Add separator
                if (i < shape_[dim] - 1) {
                    os << ", ";

                    // Add newlines for better readability in higher dimensions
                    if (dim < shape_.size() - 2) {
                        os << "\n" << std::string(dim + 1, ' ');
                    }
                }
            }
            os << "]";
        }

        // Friend classes
        friend class TensorView<T>;
    };

    //==============================================================================
    // NON-MEMBER OPERATOR OVERLOADS
    //==============================================================================

    /**
     * @brief Add two tensors element-wise
     */
    template<typename T>
    Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot add tensors with different shapes");
        }

        Tensor<T> result = a.clone();
        result += b;
        return result;
    }

    /**
     * @brief Subtract two tensors element-wise
     */
    template<typename T>
    Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot subtract tensors with different shapes");
        }

        Tensor<T> result = a.clone();
        result -= b;
        return result;
    }

    /**
     * @brief Multiply two tensors element-wise
     */
    template<typename T>
    Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot multiply tensors with different shapes");
        }

        Tensor<T> result = a.clone();
        result *= b;
        return result;
    }

    /**
     * @brief Divide two tensors element-wise
     */
    template<typename T>
    Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot divide tensors with different shapes");
        }

        Tensor<T> result = a.clone();
        result /= b;
        return result;
    }

    /**
     * @brief Add scalar to tensor
     */
    template<typename T>
    Tensor<T> operator+(const Tensor<T>& tensor, T scalar) {
        Tensor<T> result = tensor.clone();
        result += scalar;
        return result;
    }

    /**
     * @brief Add tensor to scalar
     */
    template<typename T>
    Tensor<T> operator+(T scalar, const Tensor<T>& tensor) {
        return tensor + scalar;
    }

    /**
     * @brief Subtract scalar from tensor
     */
    template<typename T>
    Tensor<T> operator-(const Tensor<T>& tensor, T scalar) {
        Tensor<T> result = tensor.clone();
        result -= scalar;
        return result;
    }

    /**
     * @brief Subtract tensor from scalar
     */
    template<typename T>
    Tensor<T> operator-(T scalar, const Tensor<T>& tensor) {
        Tensor<T> result(tensor.shape());
        const T* tensor_data = tensor.data();
        T* result_data = result.data();

        for (size_t i = 0; i < tensor.size(); ++i) {
            result_data[i] = scalar - tensor_data[i];
        }

        return result;
    }

    /**
     * @brief Multiply tensor by scalar
     */
    template<typename T>
    Tensor<T> operator*(const Tensor<T>& tensor, T scalar) {
        Tensor<T> result = tensor.clone();
        result *= scalar;
        return result;
    }

    /**
     * @brief Multiply scalar by tensor
     */
    template<typename T>
    Tensor<T> operator*(T scalar, const Tensor<T>& tensor) {
        return tensor * scalar;
    }

    /**
     * @brief Divide tensor by scalar
     */
    template<typename T>
    Tensor<T> operator/(const Tensor<T>& tensor, T scalar) {
        Tensor<T> result = tensor.clone();
        result /= scalar;
        return result;
    }

    /**
     * @brief Divide scalar by tensor
     */
    template<typename T>
    Tensor<T> operator/(T scalar, const Tensor<T>& tensor) {
        Tensor<T> result(tensor.shape());
        const T* tensor_data = tensor.data();
        T* result_data = result.data();

        for (size_t i = 0; i < tensor.size(); ++i) {
            if (tensor_data[i] == T(0)) {
                throw std::domain_error("Division by zero");
            }
            result_data[i] = scalar / tensor_data[i];
        }

        return result;
    }

    /**
     * @brief Output tensor to stream
     */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        os << tensor.to_string();
        return os;
    }

} // namespace tensor

#endif // TENSOR_H