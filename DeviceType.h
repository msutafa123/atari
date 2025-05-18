// DeviceType.h - v0.2.0
// Defines where tensor data is stored

#ifndef DEVICE_TYPE_H
#define DEVICE_TYPE_H

#include <string>
#include <vector>

namespace tensor {

    enum class DeviceType {
        CPU,
        CUDA,       // NVIDIA GPU
        OPENCL,     // OpenCL devices
        METAL,      // Apple Metal
        TPU,        // Tensor Processing Unit
        VULKAN,     // Vulkan compute
        CUSTOM      // For user-defined devices
    };

    class Device {
    public:
        // Default constructor creates CPU device
        Device() : type_(DeviceType::CPU), id_(-1), name_("CPU") {}

        // Create device with type and optional device ID
        explicit Device(DeviceType type, int id = 0, const std::string& name = "")
            : type_(type), id_(id) {
            if (name.empty()) {
                // Generate default name
                switch (type) {
                case DeviceType::CPU: name_ = "CPU"; break;
                case DeviceType::CUDA: name_ = "CUDA:" + std::to_string(id); break;
                case DeviceType::OPENCL: name_ = "OpenCL:" + std::to_string(id); break;
                case DeviceType::METAL: name_ = "Metal:" + std::to_string(id); break;
                case DeviceType::TPU: name_ = "TPU:" + std::to_string(id); break;
                case DeviceType::VULKAN: name_ = "Vulkan:" + std::to_string(id); break;
                case DeviceType::CUSTOM: name_ = "Custom:" + std::to_string(id); break;
                }
            }
            else {
                name_ = name;
            }
        }

        // Get device type
        DeviceType type() const { return type_; }

        // Get device ID
        int id() const { return id_; }

        // Get device name
        const std::string& name() const { return name_; }

        // Check device type
        bool is_cpu() const { return type_ == DeviceType::CPU; }
        bool is_cuda() const { return type_ == DeviceType::CUDA; }
        bool is_opencl() const { return type_ == DeviceType::OPENCL; }
        bool is_metal() const { return type_ == DeviceType::METAL; }
        bool is_tpu() const { return type_ == DeviceType::TPU; }
        bool is_vulkan() const { return type_ == DeviceType::VULKAN; }
        bool is_custom() const { return type_ == DeviceType::CUSTOM; }

        // Check if device supports GPU acceleration
        bool is_accelerator() const {
            return type_ != DeviceType::CPU;
        }

        // String representation
        std::string to_string() const {
            return name_;
        }

        // Equality operators
        bool operator==(const Device& other) const {
            return type_ == other.type_ && id_ == other.id_;
        }

        bool operator!=(const Device& other) const {
            return !(*this == other);
        }

        // Static device instances
        static Device cpu() { return Device(DeviceType::CPU); }
        static Device cuda(int id = 0) { return Device(DeviceType::CUDA, id); }
        static Device opencl(int id = 0) { return Device(DeviceType::OPENCL, id); }
        static Device metal(int id = 0) { return Device(DeviceType::METAL, id); }
        static Device tpu(int id = 0) { return Device(DeviceType::TPU, id); }
        static Device vulkan(int id = 0) { return Device(DeviceType::VULKAN, id); }
        static Device custom(int id = 0, const std::string& name = "Custom") {
            return Device(DeviceType::CUSTOM, id, name);
        }

    private:
        DeviceType type_;
        int id_;
        std::string name_;
    };

    // Global CPU device instance
    static const Device CPU = Device::cpu();

} // namespace tensor

#endif // DEVICE_TYPE_H