// TensorReduction.h - v0.2.0
// Reduction operations for tensors (sum, mean, etc.)

#ifndef TENSOR_REDUCTION_H
#define TENSOR_REDUCTION_H

#include "Tensor.h"
#include <functional>
#include <limits>
#include <numeric>
#include <algorithm>

namespace tensor {

    // Reduction operation types
    enum class ReductionOp {
        SUM,
        MEAN,
        MAX,
        MIN,
        PROD,
        ALL,       // Logical AND for boolean tensors
        ANY,       // Logical OR for boolean tensors
        VARIANCE,  // Statistical variance
        STD,       // Standard deviation
        NORM_1,    // L1 norm
        NORM_2,    // L2 norm
        NORM_INF   // Infinity norm
    };

    template<typename T>
    class TensorReducer {
    public:
        // Reduce tensor along specified dimensions
        static Tensor<T> reduce(const Tensor<T>& input, const std::vector<size_t>& dims,
            ReductionOp op = ReductionOp::SUM, bool keep_dims = false) {
            // Get input shape
            const TensorShape& input_shape = input.shape();

            // Create mask of dimensions to reduce
            std::vector<bool> reduce_dims(input_shape.ndim(), false);
            for (auto dim : dims) {
                if (dim >= input_shape.ndim()) {
                    throw std::out_of_range("Dimension out of range");
                }
                reduce_dims[dim] = true;
            }

            // Create output shape
            std::vector<size_t> output_dims;
            if (keep_dims) {
                output_dims.resize(input_shape.ndim());
                for (size_t i = 0; i < input_shape.ndim(); ++i) {
                    output_dims[i] = reduce_dims[i] ? 1 : input_shape.dim(i);
                }
            }
            else {
                for (size_t i = 0; i < input_shape.ndim(); ++i) {
                    if (!reduce_dims[i]) {
                        output_dims.push_back(input_shape.dim(i));
                    }
                }
                // Handle scalar output case
                if (output_dims.empty()) {
                    output_dims.push_back(1);
                }
            }

            // Create output tensor
            TensorShape output_shape(output_dims);
            Tensor<T> output(output_shape);

            // Initialize output based on operation
            T init_value;
            switch (op) {
            case ReductionOp::SUM:
            case ReductionOp::MEAN:
            case ReductionOp::VARIANCE:
            case ReductionOp::STD:
            case ReductionOp::NORM_1:
            case ReductionOp::NORM_2:
                init_value = T(0);
                break;
            case ReductionOp::MAX:
            case ReductionOp::NORM_INF:
                init_value = std::numeric_limits<T>::lowest();
                break;
            case ReductionOp::MIN:
                init_value = std::numeric_limits<T>::max();
                break;
            case ReductionOp::PROD:
                init_value = T(1);
                break;
            case ReductionOp::ALL:
                init_value = true;
                break;
            case ReductionOp::ANY:
                init_value = false;
                break;
            }
            output.fill(init_value);

            // Helper to map input index to output index
            auto map_index = [&](const std::vector<size_t>& input_idx) {
                std::vector<size_t> output_idx;
                if (keep_dims) {
                    output_idx.resize(input_idx.size());
                    for (size_t i = 0; i < input_idx.size(); ++i) {
                        output_idx[i] = reduce_dims[i] ? 0 : input_idx[i];
                    }
                }
                else {
                    for (size_t i = 0; i < input_idx.size(); ++i) {
                        if (!reduce_dims[i]) {
                            output_idx.push_back(input_idx[i]);
                        }
                    }
                }
                return output_idx;
                };

            // Counter for mean and variance calculation
            std::vector<size_t> counts(output.size(), 0);

            // For variance, we need to store intermediate means
            Tensor<T> means;
            if (op == ReductionOp::VARIANCE || op == ReductionOp::STD) {
                // First pass: calculate means
                means = reduce(input, dims, ReductionOp::MEAN, keep_dims);
            }

            // Iterate over all input elements
            // This is a naive implementation; real code would use optimized algorithms
            std::vector<size_t> idx(input_shape.ndim(), 0);
            bool done = false;

            while (!done) {
                // Get value from input
                T value = input.at(idx);

                // Map to output index
                std::vector<size_t> out_idx = map_index(idx);

                // Update count for this output element
                size_t flat_out_idx = 0;
                for (size_t i = 0; i < out_idx.size(); ++i) {
                    flat_out_idx = flat_out_idx * output_shape.dim(i) + out_idx[i];
                }
                counts[flat_out_idx]++;

                // Update output based on operation
                switch (op) {
                case ReductionOp::SUM:
                    output.at(out_idx) += value;
                    break;
                case ReductionOp::MEAN:
                    output.at(out_idx) += value;
                    break;
                case ReductionOp::MAX:
                    output.at(out_idx) = std::max(output.at(out_idx), value);
                    break;
                case ReductionOp::MIN:
                    output.at(out_idx) = std::min(output.at(out_idx), value);
                    break;
                case ReductionOp::PROD:
                    output.at(out_idx) *= value;
                    break;
                case ReductionOp::ALL:
                    output.at(out_idx) = output.at(out_idx) && (value != T(0));
                    break;
                case ReductionOp::ANY:
                    output.at(out_idx) = output.at(out_idx) || (value != T(0));
                    break;
                case ReductionOp::VARIANCE:
                {
                    T mean = means.at(out_idx);
                    T diff = value - mean;
                    output.at(out_idx) += diff * diff;
                }
                break;
                case ReductionOp::STD:
                {
                    T mean = means.at(out_idx);
                    T diff = value - mean;
                    output.at(out_idx) += diff * diff;
                }
                break;
                case ReductionOp::NORM_1:
                    output.at(out_idx) += std::abs(value);
                    break;
                case ReductionOp::NORM_2:
                    output.at(out_idx) += value * value;
                    break;
                case ReductionOp::NORM_INF:
                    output.at(out_idx) = std::max(output.at(out_idx), std::abs(value));
                    break;
                }

                // Increment index
                for (int i = idx.size() - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < input_shape.dim(i)) {
                        break;
                    }
                    idx[i] = 0;
                    if (i == 0) {
                        done = true;
                    }
                }
            }

            // Post-processing for certain operations
            if (op == ReductionOp::MEAN) {
                // Divide each element by count
                for (size_t i = 0; i < output.size(); ++i) {
                    size_t count = counts[i];
                    if (count > 0) {
                        output.data()[i] /= static_cast<T>(count);
                    }
                }
            }
            else if (op == ReductionOp::VARIANCE) {
                // Divide sum of squares by count
                for (size_t i = 0; i < output.size(); ++i) {
                    size_t count = counts[i];
                    if (count > 0) {
                        output.data()[i] /= static_cast<T>(count);
                    }
                }
            }
            else if (op == ReductionOp::STD) {
                // Calculate square root of variance
                for (size_t i = 0; i < output.size(); ++i) {
                    size_t count = counts[i];
                    if (count > 0) {
                        output.data()[i] = std::sqrt(output.data()[i] / static_cast<T>(count));
                    }
                }
            }
            else if (op == ReductionOp::NORM_2) {
                // Take square root of sum of squares
                for (size_t i = 0; i < output.size(); ++i) {
                    output.data()[i] = std::sqrt(output.data()[i]);
                }
            }

            return output;
        }

        // Reduce across all dimensions
        static T reduce_all(const Tensor<T>& input, ReductionOp op = ReductionOp::SUM) {
            // Create vector of all dimensions
            std::vector<size_t> dims(input.shape().ndim());
            std::iota(dims.begin(), dims.end(), 0);

            // Perform reduction to scalar
            Tensor<T> result = reduce(input, dims, op, false);

            // Return the single value
            return result.at({ 0 });
        }

        // Convenience methods for common reductions
        static Tensor<T> sum(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::SUM, keep_dims);
        }

        static Tensor<T> mean(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::MEAN, keep_dims);
        }

        static Tensor<T> max(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::MAX, keep_dims);
        }

        static Tensor<T> min(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::MIN, keep_dims);
        }

        static Tensor<T> prod(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::PROD, keep_dims);
        }

        static Tensor<T> var(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::VARIANCE, keep_dims);
        }

        static Tensor<T> std(const Tensor<T>& input, const std::vector<size_t>& dims, bool keep_dims = false) {
            return reduce(input, dims, ReductionOp::STD, keep_dims);
        }

        static Tensor<T> norm(const Tensor<T>& input, const std::vector<size_t>& dims,
            int p = 2, bool keep_dims = false) {
            switch (p) {
            case 1:
                return reduce(input, dims, ReductionOp::NORM_1, keep_dims);
            case 2:
                return reduce(input, dims, ReductionOp::NORM_2, keep_dims);
            case std::numeric_limits<int>::max():
                return reduce(input, dims, ReductionOp::NORM_INF, keep_dims);
            default:
                throw std::invalid_argument("Unsupported norm order: " + std::to_string(p));
            }
        }

        // Scalar reductions over entire tensor
        static T sum_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::SUM);
        }

        static T mean_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::MEAN);
        }

        static T max_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::MAX);
        }

        static T min_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::MIN);
        }

        static T prod_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::PROD);
        }

        static T var_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::VARIANCE);
        }

        static T std_all(const Tensor<T>& input) {
            return reduce_all(input, ReductionOp::STD);
        }

        static T norm_all(const Tensor<T>& input, int p = 2) {
            switch (p) {
            case 1:
                return reduce_all(input, ReductionOp::NORM_1);
            case 2:
                return reduce_all(input, ReductionOp::NORM_2);
            case std::numeric_limits<int>::max():
                return reduce_all(input, ReductionOp::NORM_INF);
            default:
                throw std::invalid_argument("Unsupported norm order: " + std::to_string(p));
            }
        }

        // Argmax/Argmin - return indices of maximum/minimum values
        static Tensor<size_t> argmax(const Tensor<T>& input, size_t dim) {
            if (dim >= input.shape().ndim()) {
                throw std::out_of_range("Dimension out of range");
            }

            // Create output shape by removing the specified dimension
            std::vector<size_t> output_dims;
            for (size_t i = 0; i < input.shape().ndim(); ++i) {
                if (i != dim) {
                    output_dims.push_back(input.shape().dim(i));
                }
            }

            TensorShape output_shape(output_dims);
            Tensor<size_t> output(output_shape);

            // Initialize with zeros
            output.fill(0);

            // Create helper tensor to store max values
            Tensor<T> max_values(output_shape);
            max_values.fill(std::numeric_limits<T>::lowest());

            // Iterate over all input elements
            std::vector<size_t> idx(input.shape().ndim(), 0);
            bool done = false;

            while (!done) {
                // Get value from input
                T value = input.at(idx);

                // Map to output index by removing the dimension
                std::vector<size_t> out_idx;
                for (size_t i = 0; i < idx.size(); ++i) {
                    if (i != dim) {
                        out_idx.push_back(idx[i]);
                    }
                }

                // Update max value and index
                if (value > max_values.at(out_idx)) {
                    max_values.at(out_idx) = value;
                    output.at(out_idx) = idx[dim];
                }

                // Increment index
                for (int i = idx.size() - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < input.shape().dim(i)) {
                        break;
                    }
                    idx[i] = 0;
                    if (i == 0) {
                        done = true;
                    }
                }
            }

            return output;
        }

        static Tensor<size_t> argmin(const Tensor<T>& input, size_t dim) {
            if (dim >= input.shape().ndim()) {
                throw std::out_of_range("Dimension out of range");
            }

            // Create output shape by removing the specified dimension
            std::vector<size_t> output_dims;
            for (size_t i = 0; i < input.shape().ndim(); ++i) {
                if (i != dim) {
                    output_dims.push_back(input.shape().dim(i));
                }
            }

            TensorShape output_shape(output_dims);
            Tensor<size_t> output(output_shape);

            // Initialize with zeros
            output.fill(0);

            // Create helper tensor to store min values
            Tensor<T> min_values(output_shape);
            min_values.fill(std::numeric_limits<T>::max());

            // Iterate over all input elements
            std::vector<size_t> idx(input.shape().ndim(), 0);
            bool done = false;

            while (!done) {
                // Get value from input
                T value = input.at(idx);

                // Map to output index by removing the dimension
                std::vector<size_t> out_idx;
                for (size_t i = 0; i < idx.size(); ++i) {
                    if (i != dim) {
                        out_idx.push_back(idx[i]);
                    }
                }

                // Update min value and index
                if (value < min_values.at(out_idx)) {
                    min_values.at(out_idx) = value;
                    output.at(out_idx) = idx[dim];
                }

                // Increment index
                for (int i = idx.size() - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < input.shape().dim(i)) {
                        break;
                    }
                    idx[i] = 0;
                    if (i == 0) {
                        done = true;
                    }
                }
            }

            return output;
        }
    };

} // namespace tensor

#endif // TENSOR_REDUCTION_H