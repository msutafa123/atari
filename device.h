// device.h - v1.0.0
// CPU ve GPU cihazlar� i�in basitle�tirilmi� y�netim katman�
// C++17 standartlar�na uygun

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

    // �leri tan�mlamalar
    template<typename T> class Tensor;

    /**
     * @brief Desteklenen cihaz t�rleri
     */
    enum class DeviceType {
        CPU,     // CPU (AMD Ryzen gibi)
        CUDA,    // NVIDIA GPU
    };

    /**
     * @brief Cihaz i�in temel s�n�f
     */
    class Device {
    public:
        /**
         * @brief Cihaz olu�turucu
         * @param type Cihaz t�r� (CPU veya CUDA)
         * @param index Cihaz indeksi (ayn� t�rden birden fazla cihaz i�in)
         */
        Device(DeviceType type, int index = 0)
            : type_(type), index_(index), name_(get_device_name(type, index)) {
        }

        virtual ~Device() = default;

        // Cihaz bilgileri
        DeviceType type() const { return type_; }
        int index() const { return index_; }
        const std::string& name() const { return name_; }

        // Cihaz t�r� yard�mc�lar�
        bool is_cpu() const { return type_ == DeviceType::CPU; }
        bool is_cuda() const { return type_ == DeviceType::CUDA; }
        bool is_gpu() const { return type_ == DeviceType::CUDA; }

        // Cihaz �zellikleri
        virtual int get_core_count() const = 0;
        virtual size_t get_memory_size() const = 0;
        virtual bool supports_avx2() const = 0; // CPU i�in �nemli
        virtual bool supports_cuda() const = 0; // GPU i�in �nemli

        // Bellek y�netimi
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

        // Cihaz kar��la�t�rma
        bool operator==(const Device& other) const {
            return type_ == other.type_ && index_ == other.index_;
        }

        bool operator!=(const Device& other) const {
            return !(*this == other);
        }

        // Cihaz olu�turma metodlar�
        static std::shared_ptr<Device> create(DeviceType type, int index = 0);
        static std::shared_ptr<Device> get_default_device();

        // Cihaz ke�fi
        static std::vector<std::shared_ptr<Device>> get_all_devices();
        static int get_device_count(DeviceType type);

        // Cihaz ad� alma
        static std::string get_device_name(DeviceType type, int index);

        // Temel tensor i�lemleri i�in optimizasyon bilgisi
        virtual bool has_optimized_gemm() const = 0;  // Matris �arp�m�
        virtual bool has_optimized_convolution() const = 0;  // Konvol�syon

    protected:
        DeviceType type_;
        int index_;
        std::string name_;
    };

    /**
     * @brief CPU cihaz� (Ryzen 9 5900x gibi)
     */
    class CPUDevice : public Device {
    public:
        CPUDevice(int index = 0) : Device(DeviceType::CPU, index) {}

        int get_core_count() const override {
            return std::thread::hardware_concurrency(); // Ryzen 9 5900x: 12 �ekirdek, 24 thread
        }

        size_t get_memory_size() const override {
            // Basitle�tirilmi� - ger�ek uygulamada sistem bellek boyutunu sorgulamal�
            return 32ULL * 1024 * 1024 * 1024; // Varsay�lan 32GB
        }

        bool supports_avx2() const override {
            // Ryzen 9 5900x AVX2'yi destekler
#ifdef __AVX2__
            return true;
#else
    // Runtime kontrol� de eklenebilir
            return false;
#endif
        }

        bool supports_cuda() const override {
            return false; // CPU CUDA'y� desteklemez
        }

        void* allocate(size_t bytes) override {
            return ::operator new(bytes);
        }

        void deallocate(void* ptr) override {
            ::operator delete(ptr);
        }

        size_t get_free_memory() const override {
            // Basitle�tirilmi� - ger�ek uygulamada sistem bilgilerini sorgulamal�
            return get_memory_size() / 2; // Tahmini de�er
        }

        void synchronize() override {
            // CPU'da senkronizasyon hemen ger�ekle�ir
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
            // Ryzen 9 5900x i�in optimize edilmi� BLAS kullan�labilir
            return true;
        }

        bool has_optimized_convolution() const override {
            // Ryzen 9 5900x i�in AVX2 kullanarak optimize edilebilir
            return supports_avx2();
        }
    };

    /**
     * @brief CUDA GPU cihaz�
     */
    class CUDADevice : public Device {
    public:
        CUDADevice(int index = 0) : Device(DeviceType::CUDA, index) {
            // CUDA kullan�labilirli�ini kontrol et
            if (!is_cuda_available()) {
                throw std::runtime_error("CUDA desteklenmiyor veya kullan�lam�yor");
            }
        }

        int get_core_count() const override {
            // Basitle�tirilmi� - ger�ek uygulamada CUDA API'sini sorgulamal�
            return 5000; // �rnek CUDA �ekirdek say�s�
        }

        size_t get_memory_size() const override {
            // Basitle�tirilmi� - ger�ek uygulamada CUDA API'sini sorgulamal�
            return 8ULL * 1024 * 1024 * 1024; // Varsay�lan 8GB
        }

        bool supports_avx2() const override {
            return false; // GPU AVX2'yi desteklemez
        }

        bool supports_cuda() const override {
            return is_cuda_available();
        }

        void* allocate(size_t bytes) override {
            // CUDA bellek tahsisi - basitle�tirilmi�
            void* ptr = nullptr;
            // cudaMalloc(&ptr, bytes) benzeri bir �a�r� yap�lmal�
            return ptr;
        }

        void deallocate(void* ptr) override {
            // CUDA bellek serbest b�rakma - basitle�tirilmi�
            // cudaFree(ptr) benzeri bir �a�r� yap�lmal�
        }

        size_t get_free_memory() const override {
            // Basitle�tirilmi� - ger�ek uygulamada CUDA API'sini sorgulamal�
            return get_memory_size() / 2; // Tahmini de�er
        }

        void synchronize() override {
            // CUDA ak��lar�n� senkronize et - basitle�tirilmi�
            // cudaDeviceSynchronize() benzeri bir �a�r� yap�lmal�
        }

        template<typename T>
        void copy_to_device(const T* host_data, T* device_data, size_t count) override {
            // Host'tan device'a kopyalama - basitle�tirilmi�
            // cudaMemcpy(device_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice) benzeri bir �a�r� yap�lmal�
        }

        template<typename T>
        void copy_from_device(const T* device_data, T* host_data, size_t count) override {
            // Device'dan host'a kopyalama - basitle�tirilmi�
            // cudaMemcpy(host_data, device_data, count * sizeof(T), cudaMemcpyDeviceToHost) benzeri bir �a�r� yap�lmal�
        }

        bool has_optimized_gemm() const override {
            // CUDA i�in cuBLAS kullan�labilir
            return true;
        }

        bool has_optimized_convolution() const override {
            // CUDA i�in cuDNN kullan�labilir
            return true;
        }

    private:
        bool is_cuda_available() const {
            // CUDA kontrol� - basitle�tirilmi�
            // Ger�ek uygulamada CUDA API'sini sorgulamal�
            return true; // Varsay�lan olarak mevcut kabul edildi
        }
    };

    // Cihaz olu�turma fonksiyonu implementasyonu
    inline std::shared_ptr<Device> Device::create(DeviceType type, int index) {
        switch (type) {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>(index);
        case DeviceType::CUDA:
            try {
                return std::make_shared<CUDADevice>(index);
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA cihaz� olu�turulamad�: " << e.what() << std::endl;
                std::cerr << "CPU cihaz�na geri d�n�l�yor..." << std::endl;
                return std::make_shared<CPUDevice>(0);
            }
        default:
            throw std::invalid_argument("Desteklenmeyen cihaz t�r�");
        }
    }

    // Varsay�lan cihaz� al
    inline std::shared_ptr<Device> Device::get_default_device() {
        // �nce CUDA'y� dene, mevcut de�ilse CPU'ya d�n
        try {
            return create(DeviceType::CUDA, 0);
        }
        catch (const std::exception&) {
            return create(DeviceType::CPU, 0);
        }
    }

    // Cihaz ad�n� al
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