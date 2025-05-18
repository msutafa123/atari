// device.h - v1.0.0
// CPU ve GPU cihazlarý için basitleþtirilmiþ yönetim katmaný
// C++17 standartlarýna uygun

#ifndef DEVICE_H
#define DEVICE_H

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <iostream>

namespace tensor {

    // Ýleri tanýmlamalar
    template<typename T> class Tensor;

    /**
     * @brief Desteklenen cihaz türleri
     */
    enum class DeviceType {
        CPU,     // CPU (AMD Ryzen gibi)
        CUDA,    // NVIDIA GPU
    };

    /**
     * @brief Cihaz için temel sýnýf
     */
    class Device {
    public:
        /**
         * @brief Cihaz oluþturucu
         * @param type Cihaz türü (CPU veya CUDA)
         * @param index Cihaz indeksi (ayný türden birden fazla cihaz için)
         */
        Device(DeviceType type, int index = 0)
            : type_(type), index_(index), name_(get_device_name(type, index)) {
        }

        virtual ~Device() = default;

        // Cihaz bilgileri
        DeviceType type() const { return type_; }
        int index() const { return index_; }
        const std::string& name() const { return name_; }

        // Cihaz türü yardýmcýlarý
        bool is_cpu() const { return type_ == DeviceType::CPU; }
        bool is_cuda() const { return type_ == DeviceType::CUDA; }
        bool is_gpu() const { return type_ == DeviceType::CUDA; }

        // Cihaz özellikleri
        virtual int get_core_count() const = 0;
        virtual size_t get_memory_size() const = 0;
        virtual bool supports_avx2() const = 0; // CPU için önemli
        virtual bool supports_cuda() const = 0; // GPU için önemli

        // Bellek yönetimi
        virtual void* allocate(size_t bytes) = 0;
        virtual void deallocate(void* ptr) = 0;
        virtual size_t get_free_memory() const = 0;

        // Senkronizasyon
        virtual void synchronize() = 0;

        // Veri transferi
        template<typename T>
        virtual void copy_to_device(const T* host_data, T* device_data, size_t count) = 0;

        template<typename T>
        virtual void copy_from_device(const T* device_data, T* host_data, size_t count) = 0;

        // Cihaz karþýlaþtýrma
        bool operator==(const Device& other) const {
            return type_ == other.type_ && index_ == other.index_;
        }

        bool operator!=(const Device& other) const {
            return !(*this == other);
        }

        // Cihaz oluþturma metodlarý
        static std::shared_ptr<Device> create(DeviceType type, int index = 0);
        static std::shared_ptr<Device> get_default_device();

        // Cihaz keþfi
        static std::vector<std::shared_ptr<Device>> get_all_devices();
        static int get_device_count(DeviceType type);

        // Cihaz adý alma
        static std::string get_device_name(DeviceType type, int index);

        // Temel tensor iþlemleri için optimizasyon bilgisi
        virtual bool has_optimized_gemm() const = 0;  // Matris çarpýmý
        virtual bool has_optimized_convolution() const = 0;  // Konvolüsyon

    protected:
        DeviceType type_;
        int index_;
        std::string name_;
    };

    /**
     * @brief CPU cihazý (Ryzen 9 5900x gibi)
     */
    class CPUDevice : public Device {
    public:
        CPUDevice(int index = 0) : Device(DeviceType::CPU, index) {}

        int get_core_count() const override {
            return std::thread::hardware_concurrency(); // Ryzen 9 5900x: 12 çekirdek, 24 thread
        }

        size_t get_memory_size() const override {
            // Basitleþtirilmiþ - gerçek uygulamada sistem bellek boyutunu sorgulamalý
            return 32ULL * 1024 * 1024 * 1024; // Varsayýlan 32GB
        }

        bool supports_avx2() const override {
            // Ryzen 9 5900x AVX2'yi destekler
#ifdef __AVX2__
            return true;
#else
    // Runtime kontrolü de eklenebilir
            return false;
#endif
        }

        bool supports_cuda() const override {
            return false; // CPU CUDA'yý desteklemez
        }

        void* allocate(size_t bytes) override {
            return ::operator new(bytes);
        }

        void deallocate(void* ptr) override {
            ::operator delete(ptr);
        }

        size_t get_free_memory() const override {
            // Basitleþtirilmiþ - gerçek uygulamada sistem bilgilerini sorgulamalý
            return get_memory_size() / 2; // Tahmini deðer
        }

        void synchronize() override {
            // CPU'da senkronizasyon hemen gerçekleþir
        }

        template<typename T>
        void copy_to_device(const T* host_data, T* device_data, size_t count) override {
            std::memcpy(device_data, host_data, count * sizeof(T));
        }

        template<typename T>
        void copy_from_device(const T* device_data, T* host_data, size_t count) override {
            std::memcpy(host_data, device_data, count * sizeof(T));
        }

        bool has_optimized_gemm() const override {
            // Ryzen 9 5900x için optimize edilmiþ BLAS kullanýlabilir
            return true;
        }

        bool has_optimized_convolution() const override {
            // Ryzen 9 5900x için AVX2 kullanarak optimize edilebilir
            return supports_avx2();
        }
    };

    /**
     * @brief CUDA GPU cihazý
     */
    class CUDADevice : public Device {
    public:
        CUDADevice(int index = 0) : Device(DeviceType::CUDA, index) {
            // CUDA kullanýlabilirliðini kontrol et
            if (!is_cuda_available()) {
                throw std::runtime_error("CUDA desteklenmiyor veya kullanýlamýyor");
            }
        }

        int get_core_count() const override {
            // Basitleþtirilmiþ - gerçek uygulamada CUDA API'sini sorgulamalý
            return 5000; // Örnek CUDA çekirdek sayýsý
        }

        size_t get_memory_size() const override {
            // Basitleþtirilmiþ - gerçek uygulamada CUDA API'sini sorgulamalý
            return 8ULL * 1024 * 1024 * 1024; // Varsayýlan 8GB
        }

        bool supports_avx2() const override {
            return false; // GPU AVX2'yi desteklemez
        }

        bool supports_cuda() const override {
            return is_cuda_available();
        }

        void* allocate(size_t bytes) override {
            // CUDA bellek tahsisi - basitleþtirilmiþ
            void* ptr = nullptr;
            // cudaMalloc(&ptr, bytes) benzeri bir çaðrý yapýlmalý
            return ptr;
        }

        void deallocate(void* ptr) override {
            // CUDA bellek serbest býrakma - basitleþtirilmiþ
            // cudaFree(ptr) benzeri bir çaðrý yapýlmalý
        }

        size_t get_free_memory() const override {
            // Basitleþtirilmiþ - gerçek uygulamada CUDA API'sini sorgulamalý
            return get_memory_size() / 2; // Tahmini deðer
        }

        void synchronize() override {
            // CUDA akýþlarýný senkronize et - basitleþtirilmiþ
            // cudaDeviceSynchronize() benzeri bir çaðrý yapýlmalý
        }

        template<typename T>
        void copy_to_device(const T* host_data, T* device_data, size_t count) override {
            // Host'tan device'a kopyalama - basitleþtirilmiþ
            // cudaMemcpy(device_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice) benzeri bir çaðrý yapýlmalý
        }

        template<typename T>
        void copy_from_device(const T* device_data, T* host_data, size_t count) override {
            // Device'dan host'a kopyalama - basitleþtirilmiþ
            // cudaMemcpy(host_data, device_data, count * sizeof(T), cudaMemcpyDeviceToHost) benzeri bir çaðrý yapýlmalý
        }

        bool has_optimized_gemm() const override {
            // CUDA için cuBLAS kullanýlabilir
            return true;
        }

        bool has_optimized_convolution() const override {
            // CUDA için cuDNN kullanýlabilir
            return true;
        }

    private:
        bool is_cuda_available() const {
            // CUDA kontrolü - basitleþtirilmiþ
            // Gerçek uygulamada CUDA API'sini sorgulamalý
            return true; // Varsayýlan olarak mevcut kabul edildi
        }
    };

    // Cihaz oluþturma fonksiyonu implementasyonu
    inline std::shared_ptr<Device> Device::create(DeviceType type, int index) {
        switch (type) {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>(index);
        case DeviceType::CUDA:
            try {
                return std::make_shared<CUDADevice>(index);
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA cihazý oluþturulamadý: " << e.what() << std::endl;
                std::cerr << "CPU cihazýna geri dönülüyor..." << std::endl;
                return std::make_shared<CPUDevice>(0);
            }
        default:
            throw std::invalid_argument("Desteklenmeyen cihaz türü");
        }
    }

    // Varsayýlan cihazý al
    inline std::shared_ptr<Device> Device::get_default_device() {
        // Önce CUDA'yý dene, mevcut deðilse CPU'ya dön
        try {
            return create(DeviceType::CUDA, 0);
        }
        catch (const std::exception&) {
            return create(DeviceType::CPU, 0);
        }
    }

    // Cihaz adýný al
    inline std::string Device::get_device_name(DeviceType type, int index) {
        switch (type) {
        case DeviceType::CPU:
            return "CPU:" + std::to_string(index);
        case DeviceType::CUDA:
            return "CUDA:" + std::to_string(index);
        default:
            return "Bilinmeyen:" + std::to_string(index);
        }
    }

} // namespace tensor

#endif // DEVICE_H