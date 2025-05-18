// TensorPrinter.h - v0.2.0
// Pretty printing for tensors

#ifndef TENSOR_PRINTER_H
#define TENSOR_PRINTER_H

#include "Tensor.h"
#include "TensorIterator.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

namespace tensor {

    template<typename T>
    class TensorPrinter {
    public:
        // Constructor with configuration options
        TensorPrinter(int precision = 4, int max_line_width = 75, int max_items = 1000)
            : precision_(precision), max_line_width_(max_line_width), max_items_(max_items) {
        }

        // Print tensor to stream
        void print(const Tensor<T>& tensor, std::ostream& os = std::cout) const {
            os << "Tensor(shape=" << tensor.shape().to_string() << ", data=";

            // Handle scalar tensors
            if (tensor.size() == 1) {
                os << tensor.at({ 0 }) << ")";
                return;
            }

            // Handle empty tensors
            if (tensor.size() == 0) {
                os << "[]" << ")";
                return;
            }

            // Print tensor data with proper indentation based on dimensions
            print_tensor_data(tensor, os, 0);

            os << ")";
        }

        // Convert tensor to string
        std::string to_string(const Tensor<T>& tensor) const {
            std::stringstream ss;
            print(tensor, ss);
            return ss.str();
        }

    private:
        int precision_;
        int max_line_width_;
        int max_items_;

        // Recursive helper for pretty printing - DÜZELTME: tamamen yeniden yazýldý
        void print_tensor_data(const Tensor<T>& tensor, std::ostream& os, int indent_level,
            std::vector<size_t>& current_indices, size_t current_dim = 0) const {
            if (current_dim >= tensor.shape().ndim()) {
                return;
            }

            // Print opening bracket for current dimension
            os << "[";

            // Handle last dimension specially (actual values)
            if (current_dim == tensor.shape().ndim() - 1) {
                size_t dim_size = tensor.shape().dim(current_dim);
                size_t limit = std::min(dim_size, static_cast<size_t>(max_items_));

                for (size_t i = 0; i < limit; ++i) {
                    // Set this dimension's index
                    current_indices[current_dim] = i;

                    // Print value with precision
                    os << std::fixed << std::setprecision(precision_) << tensor.at(current_indices);

                    // Add separator if needed
                    if (i < limit - 1) {
                        os << ", ";
                    }
                }

                // If we limited the output, indicate there are more elements
                if (limit < dim_size) {
                    os << ", ...";
                }
            }
            else {
                // For non-leaf dimensions, recurse
                size_t dim_size = tensor.shape().dim(current_dim);
                size_t limit = std::min(dim_size, static_cast<size_t>(max_items_));

                for (size_t i = 0; i < limit; ++i) {
                    // Set this dimension's index
                    current_indices[current_dim] = i;

                    // Recurse to next dimension
                    print_tensor_data(tensor, os, indent_level + 1, current_indices, current_dim + 1);

                    // Add separator if needed
                    if (i < limit - 1) {
                        os << ", ";

                        // Add newline for better readability in higher dimensions
                        if (current_dim < tensor.shape().ndim() - 2) {
                            os << "\n" << std::string(indent_level + 1, ' ');
                        }
                    }
                }

                // If we limited the output, indicate there are more elements
                if (limit < dim_size) {
                    os << ", ...";
                }
            }

            // Print closing bracket
            os << "]";
        }

        // Overload that creates the initial indices vector
        void print_tensor_data(const Tensor<T>& tensor, std::ostream& os, int indent_level,
            size_t current_dim = 0) const {

            std::vector<size_t> indices(tensor.shape().ndim(), 0);
            print_tensor_data(tensor, os, indent_level, indices, current_dim);
        }
    };

    // Convenience function for printing
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        TensorPrinter<T> printer;
        printer.print(tensor, os);
        return os;
    }

} // namespace tensor

#endif // TENSOR_PRINTER_H