// DataType.h - v0.2.0
// Defines data types for tensors

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <string>
#include <cstdint>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <stdexcept>

namespace tensor {

    enum class DataTypeEnum {
        FLOAT16,  // Added half-precision
        FLOAT32,
        FLOAT64,
        INT8,     // Added more integer types
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        BOOL,
        COMPLEX64,  // Added complex number support
        COMPLEX128
    };

    class DataType {
    public:
        // Default constructor (FLOAT32)
        DataType() : type_(DataTypeEnum::FLOAT32), size_(4) {}

        // Create specific data type
        explicit DataType(DataTypeEnum type) : type_(type) {
            switch (type) {
            case DataTypeEnum::FLOAT16: size_ = 2; break;
            case DataTypeEnum::FLOAT32: size_ = 4; break;
            case DataTypeEnum::FLOAT64: size_ = 8; break;
            case DataTypeEnum::INT8: size_ = 1; break;
            case DataTypeEnum::INT16: size_ = 2; break;
            case DataTypeEnum::INT32: size_ = 4; break;
            case DataTypeEnum::INT64: size_ = 8; break;
            case DataTypeEnum::UINT8: size_ = 1; break;
            case DataTypeEnum::UINT16: size_ = 2; break;
            case DataTypeEnum::UINT32: size_ = 4; break;
            case DataTypeEnum::UINT64: size_ = 8; break;
            case DataTypeEnum::BOOL: size_ = 1; break;
            case DataTypeEnum::COMPLEX64: size_ = 8; break;
            case DataTypeEnum::COMPLEX128: size_ = 16; break;
            }
        }

        // Get type enum
        DataTypeEnum type() const { return type_; }

        // Get size in bytes
        size_t size() const { return size_; }

        // Get name as string
        std::string name() const {
            switch (type_) {
            case DataTypeEnum::FLOAT16: return "float16";
            case DataTypeEnum::FLOAT32: return "float32";
            case DataTypeEnum::FLOAT64: return "float64";
            case DataTypeEnum::INT8: return "int8";
            case DataTypeEnum::INT16: return "int16";
            case DataTypeEnum::INT32: return "int32";
            case DataTypeEnum::INT64: return "int64";
            case DataTypeEnum::UINT8: return "uint8";
            case DataTypeEnum::UINT16: return "uint16";
            case DataTypeEnum::UINT32: return "uint32";
            case DataTypeEnum::UINT64: return "uint64";
            case DataTypeEnum::BOOL: return "bool";
            case DataTypeEnum::COMPLEX64: return "complex64";
            case DataTypeEnum::COMPLEX128: return "complex128";
            default: return "unknown";
            }
        }

        // Check if type is floating point
        bool is_floating_point() const {
            return type_ == DataTypeEnum::FLOAT16 ||
                type_ == DataTypeEnum::FLOAT32 ||
                type_ == DataTypeEnum::FLOAT64;
        }

        // Check if type is integer
        bool is_integer() const {
            return type_ == DataTypeEnum::INT8 ||
                type_ == DataTypeEnum::INT16 ||
                type_ == DataTypeEnum::INT32 ||
                type_ == DataTypeEnum::INT64 ||
                type_ == DataTypeEnum::UINT8 ||
                type_ == DataTypeEnum::UINT16 ||
                type_ == DataTypeEnum::UINT32 ||
                type_ == DataTypeEnum::UINT64;
        }

        // Check if type is signed
        bool is_signed() const {
            return type_ == DataTypeEnum::INT8 ||
                type_ == DataTypeEnum::INT16 ||
                type_ == DataTypeEnum::INT32 ||
                type_ == DataTypeEnum::INT64 ||
                type_ == DataTypeEnum::FLOAT16 ||
                type_ == DataTypeEnum::FLOAT32 ||
                type_ == DataTypeEnum::FLOAT64 ||
                type_ == DataTypeEnum::COMPLEX64 ||
                type_ == DataTypeEnum::COMPLEX128;
        }

        // Check if type is complex
        bool is_complex() const {
            return type_ == DataTypeEnum::COMPLEX64 ||
                type_ == DataTypeEnum::COMPLEX128;
        }

        // Get minimum representable value
        template<typename T>
        T min() const {
            // Implementation would depend on type
            throw std::runtime_error("Not implemented");
        }

        // Get maximum representable value
        template<typename T>
        T max() const {
            // Implementation would depend on type
            throw std::runtime_error("Not implemented");
        }

        // Equality operators
        bool operator==(const DataType& other) const {
            return type_ == other.type_;
        }

        bool operator!=(const DataType& other) const {
            return !(*this == other);
        }

        // Helper functions for type matching
        template<typename T> static DataType from_type();

        // Create from type name string
        static DataType from_string(const std::string& type_name) {
            static const std::unordered_map<std::string, DataTypeEnum> type_map = {
                {"float16", DataTypeEnum::FLOAT16},
                {"float32", DataTypeEnum::FLOAT32},
                {"float64", DataTypeEnum::FLOAT64},
                {"int8", DataTypeEnum::INT8},
                {"int16", DataTypeEnum::INT16},
                {"int32", DataTypeEnum::INT32},
                {"int64", DataTypeEnum::INT64},
                {"uint8", DataTypeEnum::UINT8},
                {"uint16", DataTypeEnum::UINT16},
                {"uint32", DataTypeEnum::UINT32},
                {"uint64", DataTypeEnum::UINT64},
                {"bool", DataTypeEnum::BOOL},
                {"complex64", DataTypeEnum::COMPLEX64},
                {"complex128", DataTypeEnum::COMPLEX128}
            };

            auto it = type_map.find(type_name);
            if (it == type_map.end()) {
                throw std::invalid_argument("Unknown data type: " + type_name);
            }

            return DataType(it->second);
        }

    private:
        DataTypeEnum type_;
        size_t size_;
    };

    // Template specializations for common types
    // Note: We're adding more specializations for the expanded type set
    template<> inline DataType DataType::from_type<float>() { return DataType(DataTypeEnum::FLOAT32); }
    template<> inline DataType DataType::from_type<double>() { return DataType(DataTypeEnum::FLOAT64); }
    template<> inline DataType DataType::from_type<int8_t>() { return DataType(DataTypeEnum::INT8); }
    template<> inline DataType DataType::from_type<int16_t>() { return DataType(DataTypeEnum::INT16); }
    template<> inline DataType DataType::from_type<int32_t>() { return DataType(DataTypeEnum::INT32); }
    template<> inline DataType DataType::from_type<int64_t>() { return DataType(DataTypeEnum::INT64); }
    template<> inline DataType DataType::from_type<uint8_t>() { return DataType(DataTypeEnum::UINT8); }
    template<> inline DataType DataType::from_type<uint16_t>() { return DataType(DataTypeEnum::UINT16); }
    template<> inline DataType DataType::from_type<uint32_t>() { return DataType(DataTypeEnum::UINT32); }
    template<> inline DataType DataType::from_type<uint64_t>() { return DataType(DataTypeEnum::UINT64); }
    template<> inline DataType DataType::from_type<bool>() { return DataType(DataTypeEnum::BOOL); }
    // Note: Complex number specializations would be added if there's a suitable C++ complex number type

} // namespace tensor

#endif // DATA_TYPE_H