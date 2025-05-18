// TensorSerializer.h - v0.1.0
// Save and load tensors to/from files

#ifndef TENSOR_SERIALIZER_H
#define TENSOR_SERIALIZER_H

#include "Tensor.h"
#include "DataType.h"
#include <fstream>
#include <string>
#include <cstdint>

namespace tensor {

    class TensorSerializer {
    public:
        // Save tensor to file
        template<typename T>
        static bool save(const Tensor<T>& tensor, const std::string& filename) {
            // Open file for binary writing
            std::ofstream file(filename, std::ios::binary);
            if (!file) {
                return false;
            }

            // Write magic number
            const uint32_t magic = 0x54534E54; // "TSNT"
            file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

            // Write version
            const uint32_t version = 1;
            file.write(reinterpret_cast<const char*>(&version), sizeof(version));

            // Write data type
            DataType data_type = DataType::from_type<T>();
            uint32_t type_enum = static_cast<uint32_t>(data_type.type());
            file.write(reinterpret_cast<const char*>(&type_enum), sizeof(type_enum));

            // Write shape
            const TensorShape& shape = tensor.shape();
            uint32_t ndim = static_cast<uint32_t>(shape.ndim());
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

            for (size_t i = 0; i < shape.ndim(); ++i) {
                uint64_t dim = shape.dim(i);
                file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }

            // Write data
            size_t data_size = tensor.size() * sizeof(T);
            file.write(reinterpret_cast<const char*>(tensor.data()), data_size);

            return file.good();
        }

        // Load tensor from file
        template<typename T>
        static bool load(Tensor<T>& tensor, const std::string& filename) {
            // Open file for binary reading
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                return false;
            }

            // Read and check magic number
            uint32_t magic;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic != 0x54534E54) {
                return false;
            }

            // Read version
            uint32_t version;
            file.read(reinterpret_cast<char*>(&version), sizeof(version));
            if (version != 1) {
                return false;
            }

            // Read data type
            uint32_t type_enum;
            file.read(reinterpret_cast<char*>(&type_enum), sizeof(type_enum));

            // Verify data type matches
            DataType expected_type = DataType::from_type<T>();
            if (static_cast<DataTypeEnum>(type_enum) != expected_type.type()) {
                return false;
            }

            // Read shape
            uint32_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

            std::vector<size_t> dims(ndim);
            for (uint32_t i = 0; i < ndim; ++i) {
                uint64_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                dims[i] = static_cast<size_t>(dim);
            }

            // Create tensor with correct shape
            TensorShape shape(dims);
            tensor = Tensor<T>(shape);

            // Read data
            size_t data_size = tensor.size() * sizeof(T);
            file.read(reinterpret_cast<char*>(tensor.data()), data_size);

            return file.good();
        }
    };

} // namespace tensor

#endif // TENSOR_SERIALIZER_H