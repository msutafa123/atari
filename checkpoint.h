// checkpoint.h - v1.0.0
// Model durumunu kaydetme ve y�kleme i�lemleri i�in ara�lar
// C++17 standartlar�na uygun

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <fstream>
#include <future>
#include <optional>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <variant>
#include <mutex>
#include <iostream>

namespace tensor {

    // �leri tan�mlamalar
    template<typename T> class Tensor;
    template<typename T> class Model;
    template<typename T> class Optimizer;

    /**
     * @brief Desteklenen checkpoint formatlar�
     */
    enum class CheckpointFormat {
        BINARY,    // �zel ikili format
        HDF5,      // HDF5 format�
        JSON,      // JSON format� (insan taraf�ndan okunabilir)
        PROTOBUF   // Protocol Buffers format�
    };

    /**
     * @brief Kaydetme se�enekleri ve ayarlar�
     */
    struct SaveOptions {
        CheckpointFormat format = CheckpointFormat::BINARY;
        bool compression = true;
        int compression_level = 6;  // 0-9 aras� (9 en y�ksek s�k��t�rma)
        bool include_optimizer = true;
        bool save_metadata = true;
        bool save_async = false;
    };

    /**
     * @brief Y�kleme se�enekleri ve ayarlar�
     */
    struct LoadOptions {
        bool strict_version_check = false;
        bool load_optimizer = true;
        bool map_location_to_cpu = false;
        bool verbose = false;
    };

    /**
     * @brief Meta veri s�n�f�
     */
    class Metadata {
    public:
        // Meta veri ekleme
        template<typename T>
        void set(const std::string& key, const T& value) {
            if constexpr (std::is_same_v<T, std::string>) {
                string_data_[key] = value;
            }
            else if constexpr (std::is_integral_v<T>) {
                int_data_[key] = static_cast<int64_t>(value);
            }
            else if constexpr (std::is_floating_point_v<T>) {
                float_data_[key] = static_cast<double>(value);
            }
            else if constexpr (std::is_same_v<T, bool>) {
                bool_data_[key] = value;
            }
            else {
                // Desteklenmeyen tip, bir hata g�ster veya string d�n���m� yap
                string_data_[key] = std::to_string(value);
            }
        }

        // Meta veri alma
        template<typename T>
        std::optional<T> get(const std::string& key) const {
            if constexpr (std::is_same_v<T, std::string>) {
                auto it = string_data_.find(key);
                if (it != string_data_.end()) {
                    return it->second;
                }
            }
            else if constexpr (std::is_integral_v<T>) {
                auto it = int_data_.find(key);
                if (it != int_data_.end()) {
                    return static_cast<T>(it->second);
                }
            }
            else if constexpr (std::is_floating_point_v<T>) {
                auto it = float_data_.find(key);
                if (it != float_data_.end()) {
                    return static_cast<T>(it->second);
                }
            }
            else if constexpr (std::is_same_v<T, bool>) {
                auto it = bool_data_.find(key);
                if (it != bool_data_.end()) {
                    return it->second;
                }
            }

            return std::nullopt;
        }

        // Bir anahtar�n varl���n� kontrol etme
        bool contains(const std::string& key) const {
            return string_data_.count(key) > 0 ||
                int_data_.count(key) > 0 ||
                float_data_.count(key) > 0 ||
                bool_data_.count(key) > 0;
        }

        // T�m anahtarlar� alma
        std::vector<std::string> keys() const {
            std::vector<std::string> result;

            for (const auto& [key, _] : string_data_) result.push_back(key);
            for (const auto& [key, _] : int_data_) result.push_back(key);
            for (const auto& [key, _] : float_data_) result.push_back(key);
            for (const auto& [key, _] : bool_data_) result.push_back(key);

            return result;
        }

        // JSON format�na d�n��t�rme
        std::string to_json() const;

        // JSON format�ndan olu�turma
        static Metadata from_json(const std::string& json_str);

    private:
        std::unordered_map<std::string, std::string> string_data_;
        std::unordered_map<std::string, int64_t> int_data_;
        std::unordered_map<std::string, double> float_data_;
        std::unordered_map<std::string, bool> bool_data_;
    };

    /**
     * @brief Checkpoint s�n�f�
     */
    class Checkpoint {
    public:
        Checkpoint() = default;

        /**
         * @brief Yeni bir checkpoint olu�tur
         * @param version Checkpoint versiyonu
         */
        explicit Checkpoint(const std::string& version)
            : version_(version),
            timestamp_(std::chrono::system_clock::now()) {
        }

        /**
         * @brief Model durumunu ekle
         * @param name Model ad� veya tan�mlay�c�s�
         * @param state Model durumu (tensors ve parametreler)
         */
        template<typename T>
        void add_model_state(const std::string& name, const std::unordered_map<std::string, Tensor<T>>& state) {
            models_[name] = state;
        }

        /**
         * @brief Optimizer durumunu ekle
         * @param name Optimizer ad� veya tan�mlay�c�s�
         * @param state Optimizer durumu
         */
        template<typename T>
        void add_optimizer_state(const std::string& name, const std::unordered_map<std::string, Tensor<T>>& state) {
            optimizers_[name] = state;
        }

        /**
         * @brief Meta veri ekle veya g�ncelle
         * @param metadata Metadata nesnesi
         */
        void set_metadata(const Metadata& metadata) {
            metadata_ = metadata;
        }

        /**
         * @brief Meta veri al
         * @return Metadata nesnesi
         */
        const Metadata& metadata() const {
            return metadata_;
        }

        /**
         * @brief Checkpoint versiyonunu al
         * @return Versiyon string'i
         */
        const std::string& version() const {
            return version_;
        }

        /**
         * @brief Checkpoint olu�turulma zaman�n� al
         * @return Zaman damgas�
         */
        auto timestamp() const {
            return timestamp_;
        }

        /**
         * @brief Belirli bir modelin durumunu al
         * @param name Model ad�
         * @return Model durumu veya nullptr e�er model bulunamazsa
         */
        template<typename T>
        const std::unordered_map<std::string, Tensor<T>>* get_model_state(const std::string& name) const {
            auto it = models_.find(name);
            if (it != models_.end()) {
                return &(std::get<std::unordered_map<std::string, Tensor<T>>>(it->second));
            }
            return nullptr;
        }

        /**
         * @brief Belirli bir optimizer'�n durumunu al
         * @param name Optimizer ad�
         * @return Optimizer durumu veya nullptr e�er optimizer bulunamazsa
         */
        template<typename T>
        const std::unordered_map<std::string, Tensor<T>>* get_optimizer_state(const std::string& name) const {
            auto it = optimizers_.find(name);
            if (it != optimizers_.end()) {
                return &(std::get<std::unordered_map<std::string, Tensor<T>>>(it->second));
            }
            return nullptr;
        }

    private:
        std::string version_;
        std::chrono::system_clock::time_point timestamp_;
        Metadata metadata_;

        // Model ve optimizer durumlar� - tip agnostik depolama i�in variant kullan�yoruz
        using StateVariant = std::variant
            std::unordered_map<std::string, Tensor<float>>,
            std::unordered_map<std::string, Tensor<double>>,
            std::unordered_map<std::string, Tensor<int>>,
            std::unordered_map<std::string, Tensor<int8_t>>
    > ;

    std::unordered_map<std::string, StateVariant> models_;
    std::unordered_map<std::string, StateVariant> optimizers_;
    };

    /**
     * @brief Checkpoint Y�neticisi
     */
    class CheckpointManager {
    public:
        /**
         * @brief Yeni bir CheckpointManager olu�tur
         * @param version K�t�phane versiyonu
         */
        explicit CheckpointManager(const std::string& version = "1.0.0")
            : version_(version) {
        }

        /**
         * @brief Model ve optimizer durumunu kaydet
         * @param path Dosya yolu
         * @param model Kaydedilecek model
         * @param optimizer Kaydedilecek optimizer (opsiyonel)
         * @param metadata Ek meta veri (opsiyonel)
         * @param options Kaydetme se�enekleri
         * @return Ba�ar�l� olup olmad���
         */
        template<typename T>
        bool save(const std::string& path,
            const Model<T>& model,
            const Optimizer<T>* optimizer = nullptr,
            const Metadata& metadata = Metadata(),
            const SaveOptions& options = SaveOptions()) {

            // Dosya yolunu do�rula
            std::filesystem::path file_path(path);
            auto dir = file_path.parent_path();

            if (!dir.empty() && !std::filesystem::exists(dir)) {
                try {
                    std::filesystem::create_directories(dir);
                }
                catch (const std::exception& e) {
                    std::cerr << "Dizin olu�turma hatas�: " << e.what() << std::endl;
                    return false;
                }
            }

            // Yeni checkpoint olu�tur
            Checkpoint checkpoint(version_);

            // Model durumunu ekle
            auto model_state = extract_model_state(model);
            checkpoint.add_model_state("model", model_state);

            // Optimizer durumunu ekle (e�er varsa)
            if (optimizer && options.include_optimizer) {
                auto optimizer_state = extract_optimizer_state(*optimizer);
                checkpoint.add_optimizer_state("optimizer", optimizer_state);
            }

            // Meta veriyi ayarla
            Metadata combined_metadata = metadata;

            // Otomatik meta veri ekle
            if (options.save_metadata) {
                auto now = std::chrono::system_clock::now();
                auto now_time_t = std::chrono::system_clock::to_time_t(now);
                std::string timestamp = std::ctime(&now_time_t);
                timestamp.pop_back(); // Son sat�r sonu karakterini kald�r

                combined_metadata.set("created_at", timestamp);
                combined_metadata.set("version", version_);
                combined_metadata.set("format", format_to_string(options.format));

                // Model bilgilerini ekle (s�n�f ad�, parametre say�s�, vb.)
                add_model_metadata(combined_metadata, model);
            }

            checkpoint.set_metadata(combined_metadata);

            // Asenkron kaydetme
            if (options.save_async) {
                auto save_task = std::async(std::launch::async,
                    &CheckpointManager::save_checkpoint_to_file,
                    this,
                    checkpoint,
                    path,
                    options);

                // Asenkron g�revin sonucunu izlemek i�in bir ba�lam olu�tur
                auto task_id = next_task_id_++;
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    active_tasks_[task_id] = std::move(save_task);
                }

                return true; // Asenkron ba�lat�ld�
            }
            else {
                // Senkron kaydet
                return save_checkpoint_to_file(checkpoint, path, options);
            }
        }

        /**
         * @brief Checkpoint dosyas�ndan y�kleme yap
         * @param path Dosya yolu
         * @param model Hedef model
         * @param optimizer Hedef optimizer (opsiyonel)
         * @param options Y�kleme se�enekleri
         * @return Y�klenen metadata veya hata durumunda std::nullopt
         */
        template<typename T>
        std::optional<Metadata> load(const std::string& path,
            Model<T>& model,
            Optimizer<T>* optimizer = nullptr,
            const LoadOptions& options = LoadOptions()) {

            // Dosya varl���n� kontrol et
            if (!std::filesystem::exists(path)) {
                std::cerr << "Checkpoint dosyas� bulunamad�: " << path << std::endl;
                return std::nullopt;
            }

            try {
                // Checkpoint'i dosyadan y�kle
                auto checkpoint_opt = load_checkpoint_from_file(path);
                if (!checkpoint_opt) {
                    std::cerr << "Checkpoint y�klenemedi: " << path << std::endl;
                    return std::nullopt;
                }

                auto& checkpoint = *checkpoint_opt;

                // Versiyon kontrol�
                if (options.strict_version_check && checkpoint.version() != version_) {
                    std::cerr << "Versiyon uyumsuzlu�u. Checkpoint: "
                        << checkpoint.version() << ", Beklenen: " << version_ << std::endl;
                    return std::nullopt;
                }

                // Model durumunu uygula
                const auto* model_state = checkpoint.get_model_state<T>("model");
                if (!model_state) {
                    std::cerr << "Model durumu checkpoint'te bulunamad�" << std::endl;
                    return std::nullopt;
                }

                apply_model_state(model, *model_state);

                // Optimizer durumunu uygula (e�er varsa ve isteniyorsa)
                if (optimizer && options.load_optimizer) {
                    const auto* optimizer_state = checkpoint.get_optimizer_state<T>("optimizer");
                    if (optimizer_state) {
                        apply_optimizer_state(*optimizer, *optimizer_state);
                    }
                    else if (options.verbose) {
                        std::cerr << "Uyar�: Optimizer durumu checkpoint'te bulunamad�" << std::endl;
                    }
                }

                return checkpoint.metadata();
            }
            catch (const std::exception& e) {
                std::cerr << "Checkpoint y�kleme hatas�: " << e.what() << std::endl;
                return std::nullopt;
            }
        }

        /**
         * @brief Dosyadan sadece meta veriyi y�kle
         * @param path Dosya yolu
         * @return Meta veri nesnesi veya hata durumunda std::nullopt
         */
        std::optional<Metadata> get_metadata(const std::string& path) {
            // Dosya varl���n� kontrol et
            if (!std::filesystem::exists(path)) {
                std::cerr << "Checkpoint dosyas� bulunamad�: " << path << std::endl;
                return std::nullopt;
            }

            try {
                // Sadece meta veriyi y�kle (daha verimli olabilir)
                auto metadata_opt = load_metadata_from_file(path);
                return metadata_opt;
            }
            catch (const std::exception& e) {
                std::cerr << "Meta veri y�kleme hatas�: " << e.what() << std::endl;
                return std::nullopt;
            }
        }

        /**
         * @brief Model i�in k�smi y�kleme
         * @param path Dosya yolu
         * @param model Hedef model
         * @param parameter_names Sadece y�klenecek parametre adlar�
         * @param options Y�kleme se�enekleri
         * @return Ba�ar�l� olup olmad���
         */
        template<typename T>
        bool load_partial(const std::string& path,
            Model<T>& model,
            const std::vector<std::string>& parameter_names,
            const LoadOptions& options = LoadOptions()) {

            // Dosya varl���n� kontrol et
            if (!std::filesystem::exists(path)) {
                std::cerr << "Checkpoint dosyas� bulunamad�: " << path << std::endl;
                return false;
            }

            try {
                // Checkpoint'i dosyadan y�kle
                auto checkpoint_opt = load_checkpoint_from_file(path);
                if (!checkpoint_opt) {
                    std::cerr << "Checkpoint y�klenemedi: " << path << std::endl;
                    return false;
                }

                auto& checkpoint = *checkpoint_opt;

                // Model durumunu al
                const auto* model_state = checkpoint.get_model_state<T>("model");
                if (!model_state) {
                    std::cerr << "Model durumu checkpoint'te bulunamad�" << std::endl;
                    return false;
                }

                // Se�ilen parametreleri uygula
                std::unordered_map<std::string, Tensor<T>> filtered_state;
                for (const auto& name : parameter_names) {
                    auto it = model_state->find(name);
                    if (it != model_state->end()) {
                        filtered_state[name] = it->second;
                    }
                    else if (options.verbose) {
                        std::cerr << "Uyar�: Parametre bulunamad�: " << name << std::endl;
                    }
                }

                apply_model_state(model, filtered_state);
                return true;

            }
            catch (const std::exception& e) {
                std::cerr << "K�smi y�kleme hatas�: " << e.what() << std::endl;
                return false;
            }
        }

        /**
         * @brief Asenkron g�revlerin durumunu kontrol et
         * @return Tamamlanan g�rev say�s�
         */
        int check_async_tasks() {
            std::lock_guard<std::mutex> lock(tasks_mutex_);

            std::vector<uint64_t> completed_tasks;
            int count = 0;

            for (auto& [id, task] : active_tasks_) {
                if (task.valid() && task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    // Sonucu kontrol et
                    try {
                        bool result = task.get();
                        if (!result) {
                            std::cerr << "Asenkron g�rev ba�ar�s�z oldu: " << id << std::endl;
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Asenkron g�rev hatas�: " << e.what() << std::endl;
                    }

                    completed_tasks.push_back(id);
                    count++;
                }
            }

            // Tamamlanan g�revleri kald�r
            for (auto id : completed_tasks) {
                active_tasks_.erase(id);
            }

            return count;
        }

        /**
         * @brief T�m asenkron g�revlerin tamamlanmas�n� bekle
         * @param timeout_ms Zaman a��m� (milisaniye, 0=sonsuz)
         * @return Tamamlanan g�rev say�s�
         */
        int wait_for_async_tasks(uint64_t timeout_ms = 0) {
            auto start_time = std::chrono::steady_clock::now();
            int completed = 0;

            while (true) {
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    if (active_tasks_.empty()) {
                        return completed;
                    }
                }

                completed += check_async_tasks();

                // Zaman a��m�n� kontrol et
                if (timeout_ms > 0) {
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        current_time - start_time).count();

                    if (elapsed >= timeout_ms) {
                        return completed;
                    }
                }

                // CPU kullan�m�n� azaltmak i�in biraz bekle
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

    private:
        std::string version_;
        uint64_t next_task_id_ = 1;
        std::unordered_map<uint64_t, std::future<bool>> active_tasks_;
        std::mutex tasks_mutex_;

        // Format d�n���mleri
        static std::string format_to_string(CheckpointFormat format) {
            switch (format) {
            case CheckpointFormat::BINARY: return "binary";
            case CheckpointFormat::HDF5: return "hdf5";
            case CheckpointFormat::JSON: return "json";
            case CheckpointFormat::PROTOBUF: return "protobuf";
            default: return "unknown";
            }
        }

        // Modelden parametre ��karma
        template<typename T>
        std::unordered_map<std::string, Tensor<T>> extract_model_state(const Model<T>& model) {
            // Bu k�s�mda ger�ek modelden parametreleri ��karma kodu olacak
            // Basitle�tirilmi� prototip:
            std::unordered_map<std::string, Tensor<T>> state;

            // Model s�n�f�nda get_parameters() gibi bir metod oldu�unu varsay�yoruz
            // model.get_parameters() kullan�labilir

            return state;
        }

        // Optimizerdan durum ��karma
        template<typename T>
        std::unordered_map<std::string, Tensor<T>> extract_optimizer_state(const Optimizer<T>& optimizer) {
            // Bu k�s�mda ger�ek optimizerdan durumu ��karma kodu olacak
            // Basitle�tirilmi� prototip:
            std::unordered_map<std::string, Tensor<T>> state;

            // Optimizer s�n�f�nda get_state() gibi bir metod oldu�unu varsay�yoruz
            // optimizer.get_state() kullan�labilir

            return state;
        }

        // Modele parametre uygulama
        template<typename T>
        void apply_model_state(Model<T>& model,
            const std::unordered_map<std::string, Tensor<T>>& state) {
            // Bu k�s�mda ger�ek modele parametreleri uygulama kodu olacak
            // Basitle�tirilmi� prototip:

            // Model s�n�f�nda load_parameters() gibi bir metod oldu�unu varsay�yoruz
            // model.load_parameters(state) kullan�labilir
        }

        // Optimizera durum uygulama
        template<typename T>
        void apply_optimizer_state(Optimizer<T>& optimizer,
            const std::unordered_map<std::string, Tensor<T>>& state) {
            // Bu k�s�mda ger�ek optimizera durumu uygulama kodu olacak
            // Basitle�tirilmi� prototip:

            // Optimizer s�n�f�nda load_state() gibi bir metod oldu�unu varsay�yoruz
            // optimizer.load_state(state) kullan�labilir
        }

        // Model meta verisi ekleme
        template<typename T>
        void add_model_metadata(Metadata& metadata, const Model<T>& model) {
            // Model hakk�nda bilgileri meta veriye ekler
            // Model s�n�f�nda get_name(), get_parameter_count() gibi metodlar oldu�unu varsay�yoruz

            // �rnek:
            // metadata.set("model_name", model.get_name());
            // metadata.set("parameter_count", model.get_parameter_count());
        }

        // Checkpoint'i dosyaya kaydetme
        bool save_checkpoint_to_file(const Checkpoint& checkpoint,
            const std::string& path,
            const SaveOptions& options) {
            // Bu k�s�mda ger�ek dosya kaydetme kodu olacak
            // Format se�imine g�re uygun serile�tirme yap�l�r

            switch (options.format) {
            case CheckpointFormat::BINARY:
                return save_binary(checkpoint, path, options);
            case CheckpointFormat::HDF5:
                return save_hdf5(checkpoint, path, options);
            case CheckpointFormat::JSON:
                return save_json(checkpoint, path, options);
            case CheckpointFormat::PROTOBUF:
                return save_protobuf(checkpoint, path, options);
            default:
                std::cerr << "Desteklenmeyen format: "
                    << format_to_string(options.format) << std::endl;
                return false;
            }
        }

        // Dosyadan checkpoint y�kleme
        std::optional<Checkpoint> load_checkpoint_from_file(const std::string& path) {
            // Bu k�s�mda ger�ek dosyadan y�kleme kodu olacak
            // Dosya uzant�s�na bakarak format� tespit edebiliriz

            std::filesystem::path file_path(path);
            std::string extension = file_path.extension().string();

            if (extension == ".bin") {
                return load_binary(path);
            }
            else if (extension == ".h5" || extension == ".hdf5") {
                return load_hdf5(path);
            }
            else if (extension == ".json") {
                return load_json(path);
            }
            else if (extension == ".pb") {
                return load_protobuf(path);
            }
            else {
                // Varsay�lan olarak binary dene
                return load_binary(path);
            }
        }

        // Dosyadan sadece meta veriyi y�kleme
        std::optional<Metadata> load_metadata_from_file(const std::string& path) {
            // Bu k�s�mda sadece meta veriyi okuma kodu olacak
            // Daha verimli olmas� i�in t�m checkpoint'i y�klemeden

            // Basitle�tirilmi� prototip - ger�ek uygulamada optimize edilmeli
            auto checkpoint_opt = load_checkpoint_from_file(path);
            if (checkpoint_opt) {
                return checkpoint_opt->metadata();
            }

            return std::nullopt;
        }

        // Binary format implementasyonlar�
        bool save_binary(const Checkpoint& checkpoint,
            const std::string& path,
            const SaveOptions& options) {
            // Binary formatta kaydetme kodu
            return false; // Hen�z implemente edilmedi
        }

        std::optional<Checkpoint> load_binary(const std::string& path) {
            // Binary formattan y�kleme kodu
            return std::nullopt; // Hen�z implemente edilmedi
        }

        // HDF5 format implementasyonlar�
        bool save_hdf5(const Checkpoint& checkpoint,
            const std::string& path,
            const SaveOptions& options) {
            // HDF5 formatta kaydetme kodu
            return false; // Hen�z implemente edilmedi
        }

        std::optional<Checkpoint> load_hdf5(const std::string& path) {
            // HDF5 formattan y�kleme kodu
            return std::nullopt; // Hen�z implemente edilmedi
        }

        // JSON format implementasyonlar�
        bool save_json(const Checkpoint& checkpoint,
            const std::string& path,
            const SaveOptions& options) {
            // JSON formatta kaydetme kodu
            return false; // Hen�z implemente edilmedi
        }

        std::optional<Checkpoint> load_json(const std::string& path) {
            // JSON formattan y�kleme kodu
            return std::nullopt; // Hen�z implemente edilmedi
        }

        // Protobuf format implementasyonlar�
        bool save_protobuf(const Checkpoint& checkpoint,
            const std::string& path,
            const SaveOptions& options) {
            // Protobuf formatta kaydetme kodu
            return false; // Hen�z implemente edilmedi
        }

        std::optional<Checkpoint> load_protobuf(const std::string& path) {
            // Protobuf formattan y�kleme kodu
            return std::nullopt; // Hen�z implemente edilmedi
        }
    };

    /**
     * @brief Otomatik checkpoint y�neticisi
     */
    template<typename T>
    class AutoCheckpoint {
    public:
        /**
         * @brief Otomatik checkpoint y�neticisi olu�tur
         * @param manager CheckpointManager nesnesi
         * @param model Kaydedilecek model
         * @param optimizer Kaydedilecek optimizer (opsiyonel)
         * @param directory Checkpoint dizini
         * @param interval Kaydetme aral��� (iterasyon/epoch say�s�)
         * @param max_to_keep Saklanacak maksimum checkpoint say�s�
         * @param options Kaydetme se�enekleri
         */
        AutoCheckpoint(CheckpointManager& manager,
            Model<T>& model,
            Optimizer<T>* optimizer = nullptr,
            const std::string& directory = "checkpoints",
            int interval = 1,
            int max_to_keep = 5,
            const SaveOptions& options = SaveOptions())
            : manager_(manager),
            model_(model),
            optimizer_(optimizer),
            directory_(directory),
            interval_(interval),
            max_to_keep_(max_to_keep),
            options_(options),
            step_counter_(0) {

            // Dizin varl���n� kontrol et
            std::filesystem::path dir_path(directory);
            if (!std::filesystem::exists(dir_path)) {
                std::filesystem::create_directories(dir_path);
            }
        }

        /**
         * @brief Step say�s�n� art�r ve gerekirse checkpoint kaydet
         * @param metadata Ek meta veri (opsiyonel)
         * @return Checkpoint kaydedildi mi
         */
        bool step(const Metadata& metadata = Metadata()) {
            step_counter_++;

            if (step_counter_ % interval_ == 0) {
                return save_checkpoint(metadata);
            }

            return false;
        }

        /**
         * @brief Manuel olarak checkpoint kaydet
         * @param metadata Ek meta veri (opsiyonel)
         * @return Ba�ar�l� olup olmad���
         */
        bool save_checkpoint(const Metadata& metadata = Metadata()) {
            std::string checkpoint_path = generate_checkpoint_path();

            // Metadata'ya ad�m bilgisini ekle
            Metadata enhanced_metadata = metadata;
            enhanced_metadata.set("step", step_counter_);
            enhanced_metadata.set("epoch", step_counter_ / steps_per_epoch_);

            bool success = manager_.save(checkpoint_path, model_, optimizer_, enhanced_metadata, options_);

            if (success) {
                // Checkpoint listesini g�ncelle
                checkpoints_.push_back(checkpoint_path);

                // Eski checkpoint'leri kald�r
                cleanup_old_checkpoints();
            }

            return success;
        }

        /**
         * @brief En son checkpoint'i y�kle
         * @param options Y�kleme se�enekleri
         * @return Y�klenen meta veri veya std::nullopt
         */
        std::optional<Metadata> restore_latest(const LoadOptions& options = LoadOptions()) {
            if (checkpoints_.empty()) {
                // Dizini tara ve mevcut checkpoint'leri bul
                scan_checkpoint_directory();
            }

            if (checkpoints_.empty()) {
                std::cerr << "Y�klenecek checkpoint bulunamad�" << std::endl;
                return std::nullopt;
            }

            // En son checkpoint'i al
            std::string latest = checkpoints_.back();

            // Y�kle
            return manager_.load(latest, model_, optimizer_, options);
        }

        /**
         * @brief Epoch boyutu ayarla
         * @param steps_per_epoch Bir epoch'taki ad�m say�s�
         */
        void set_steps_per_epoch(int steps_per_epoch) {
            steps_per_epoch_ = steps_per_epoch;
        }

        /**
         * @brief Kaydetme aral���n� de�i�tir
         * @param interval Yeni aral�k
         */
        void set_interval(int interval) {
            interval_ = interval;
        }

        /**
         * @brief Checkpoint adland�rma format�n� de�i�tir
         * @param format Format string (�rn. "model_{step}.ckpt")
         */
        void set_checkpoint_format(const std::string& format) {
            checkpoint_format_ = format;
        }

        /**
         * @brief Saklanacak maksimum checkpoint say�s�n� de�i�tir
         * @param max_to_keep Yeni de�er
         */
        void set_max_to_keep(int max_to_keep) {
            max_to_keep_ = max_to_keep;
            cleanup_old_checkpoints();
        }

        /**
         * @brief Ad�m sayac�n� ayarla
         * @param step Yeni ad�m say�s�
         */
        void set_step(int step) {
            step_counter_ = step;
        }

    private:
        CheckpointManager& manager_;
        Model<T>& model_;
        Optimizer<T>* optimizer_;
        std::string directory_;
        int interval_;
        int max_to_keep_;
        SaveOptions options_;
        int step_counter_ = 0;
        int steps_per_epoch_ = 1;
        std::string checkpoint_format_ = "model_{step}.ckpt";
        std::vector<std::string> checkpoints_;

        // Checkpoint yolu olu�tur
        std::string generate_checkpoint_path() {
            std::string path = checkpoint_format_;

            // {step} k�sm�n� ger�ek de�erle de�i�tir
            size_t pos = path.find("{step}");
            if (pos != std::string::npos) {
                path.replace(pos, 6, std::to_string(step_counter_));
            }

            // {epoch} k�sm�n� ger�ek de�erle de�i�tir
            pos = path.find("{epoch}");
            if (pos != std::string::npos) {
                path.replace(pos, 7, std::to_string(step_counter_ / steps_per_epoch_));
            }

            // Dizin ekle
            return directory_ + "/" + path;
        }

        // Eski checkpoint'leri temizle
        void cleanup_old_checkpoints() {
            if (max_to_keep_ <= 0 || checkpoints_.size() <= max_to_keep_) {
                return;
            }

            while (checkpoints_.size() > max_to_keep_) {
                // En eski checkpoint'i sil
                std::string oldest = checkpoints_.front();
                checkpoints_.erase(checkpoints_.begin());

                try {
                    std::filesystem::remove(oldest);
                }
                catch (const std::exception& e) {
                    std::cerr << "Eski checkpoint silme hatas�: " << e.what() << std::endl;
                }
            }
        }

        // Dizindeki checkpoint'leri tara
        void scan_checkpoint_directory() {
            checkpoints_.clear();

            try {
                for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
                    std::string path = entry.path().string();

                    // Sadece .ckpt, .bin, .h5, .hdf5, .pb, .json uzant�l� dosyalar� al
                    std::string ext = entry.path().extension().string();
                    if (ext == ".ckpt" || ext == ".bin" || ext == ".h5" ||
                        ext == ".hdf5" || ext == ".pb" || ext == ".json") {
                        checkpoints_.push_back(path);
                    }
                }

                // Dosyalar� s�rala (alfabetik, genellikle zaman damgal�)
                std::sort(checkpoints_.begin(), checkpoints_.end());
            }
            catch (const std::exception& e) {
                std::cerr << "Dizin tarama hatas�: " << e.what() << std::endl;
            }
        }
    };

} // namespace tensor

#endif // CHECKPOINT_H