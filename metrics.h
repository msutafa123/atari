// metrics.h - v1.0.0
// Yapay sinir aðlarý ve makine öðrenmesi modelleri için performans metrikleri
// C++17 standartlarýna uygun

#ifndef METRICS_H
#define METRICS_H

#include "Tensor.h"
#include "TensorOps.h"
#include "TensorMath.h"
#include "math_utils.h"
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <iomanip>

namespace tensor {
    namespace metrics {

        // Sayýsal kararlýlýk için küçük epsilon deðeri
        template<typename T>
        constexpr T epsilon = std::numeric_limits<T>::epsilon() * T(100);

        // Temel metrik sýnýfý - tüm metrikler için arayüz
        template<typename T>
        class Metric {
        public:
            virtual ~Metric() = default;

            // Metrik hesaplama
            virtual T compute(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;

            // Metriði güncelle (birikimli hesaplamalar için)
            virtual void update(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;

            // Birikmiþ sonucu al
            virtual T result() const = 0;

            // Birikimli hesaplamalarý sýfýrla
            virtual void reset() = 0;

            // Metrik adýný al
            virtual std::string name() const = 0;

            // Daha iyi deðerin daha yüksek mü yoksa daha düþük mü olduðunu belirle
            virtual bool higher_is_better() const = 0;
        };

        //==========================================================================
        // SINIFLANDIRMA METRÝKLERÝ
        //==========================================================================

        // Doðruluk (Accuracy) metriði
        template<typename T>
        class Accuracy : public Metric<T> {
        public:
            Accuracy() : correct_predictions_(0), total_predictions_(0) {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t correct = 0;
                size_t total = predictions.shape().dim(0);

                // Tahminlerin ve hedeflerin boyutlarýna göre hesaplama yöntemi belirle
                if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                    // Çok sýnýflý durum: predictions [batch_size, num_classes]
                    for (size_t i = 0; i < total; ++i) {
                        // En yüksek olasýlýklý sýnýfý bul
                        size_t pred_class = 0;
                        T max_prob = predictions.at({ i, 0 });

                        for (size_t j = 1; j < predictions.shape().dim(1); ++j) {
                            if (predictions.at({ i, j }) > max_prob) {
                                max_prob = predictions.at({ i, j });
                                pred_class = j;
                            }
                        }

                        // Hedefin formatýna göre kontrol et
                        if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                            // One-hot encoding hedefler
                            size_t true_class = 0;
                            T max_val = targets.at({ i, 0 });

                            for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                if (targets.at({ i, j }) > max_val) {
                                    max_val = targets.at({ i, j });
                                    true_class = j;
                                }
                            }

                            if (pred_class == true_class) {
                                correct++;
                            }
                        }
                        else {
                            // Sýnýf indeks hedefleri
                            size_t true_class = static_cast<size_t>(targets.at({ i }));
                            if (pred_class == true_class) {
                                correct++;
                            }
                        }
                    }
                }
                else {
                    // Ýkili sýnýflandýrma: predictions [batch_size]
                    for (size_t i = 0; i < total; ++i) {
                        // Tahmin eþik deðerine göre sýnýflandýrma (0.5)
                        size_t pred_class = predictions.at({ i }) >= T(0.5) ? 1 : 0;
                        size_t true_class = targets.at({ i }) >= T(0.5) ? 1 : 0;

                        if (pred_class == true_class) {
                            correct++;
                        }
                    }
                }

                return static_cast<T>(correct) / static_cast<T>(total);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                T batch_accuracy = compute(predictions, targets);
                size_t batch_size = predictions.shape().dim(0);

                // Doðru tahminleri ve toplam sayýyý güncelle
                correct_predictions_ += static_cast<size_t>(batch_accuracy * batch_size);
                total_predictions_ += batch_size;
            }

            T result() const override {
                if (total_predictions_ == 0) {
                    return T(0);
                }
                return static_cast<T>(correct_predictions_) / static_cast<T>(total_predictions_);
            }

            void reset() override {
                correct_predictions_ = 0;
                total_predictions_ = 0;
            }

            std::string name() const override {
                return "Accuracy";
            }

            bool higher_is_better() const override {
                return true; // Doðruluk için yüksek deðerler daha iyidir
            }

        private:
            size_t correct_predictions_;
            size_t total_predictions_;

            // Giriþleri doðrula
            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.ndim() == 0 || targets.ndim() == 0) {
                    throw std::invalid_argument("Predictions and targets must have at least 1 dimension");
                }

                if (predictions.shape().dim(0) != targets.shape().dim(0)) {
                    throw std::invalid_argument("Predictions and targets must have same batch size");
                }

                if (predictions.ndim() == 2 && targets.ndim() == 2) {
                    if (predictions.shape().dim(1) != targets.shape().dim(1) &&
                        targets.shape().dim(1) != 1) {
                        throw std::invalid_argument("For 2D inputs, predictions and targets must have compatible class dimensions");
                    }
                }
            }
        };

        // Precision (Kesinlik) metriði
        template<typename T>
        class Precision : public Metric<T> {
        public:
            // Tek sýnýf veya tüm sýnýflarýn ortalamasý için
            Precision(size_t class_idx = 0, bool average = true)
                : class_idx_(class_idx), average_(average), true_positives_(0),
                false_positives_(0), class_count_(0) {
            }

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t batch_size = predictions.shape().dim(0);

                // Çok sýnýflý durum için
                if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    // Tüm sýnýflar için ortalama hesapla
                    if (average_) {
                        T total_precision = T(0);
                        size_t valid_classes = 0;

                        for (size_t c = 0; c < num_classes; ++c) {
                            size_t tp = 0, fp = 0;

                            for (size_t i = 0; i < batch_size; ++i) {
                                // En yüksek olasýlýklý sýnýfý bul
                                size_t pred_class = 0;
                                T max_prob = predictions.at({ i, 0 });

                                for (size_t j = 1; j < num_classes; ++j) {
                                    if (predictions.at({ i, j }) > max_prob) {
                                        max_prob = predictions.at({ i, j });
                                        pred_class = j;
                                    }
                                }

                                // Gerçek sýnýfý belirle
                                size_t true_class;
                                if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                                    true_class = 0;
                                    T max_val = targets.at({ i, 0 });

                                    for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                        if (targets.at({ i, j }) > max_val) {
                                            max_val = targets.at({ i, j });
                                            true_class = j;
                                        }
                                    }
                                }
                                else {
                                    true_class = static_cast<size_t>(targets.at({ i }));
                                }

                                // Þu anki sýnýf için TP/FP güncelle
                                if (pred_class == c) {
                                    if (true_class == c) {
                                        tp++;
                                    }
                                    else {
                                        fp++;
                                    }
                                }
                            }

                            // Bu sýnýf için precision hesapla
                            if (tp + fp > 0) {
                                total_precision += static_cast<T>(tp) / static_cast<T>(tp + fp);
                                valid_classes++;
                            }
                        }

                        return valid_classes > 0 ? total_precision / static_cast<T>(valid_classes) : T(0);
                    }
                    // Sadece belirtilen sýnýf için hesapla
                    else {
                        if (class_idx_ >= num_classes) {
                            throw std::invalid_argument("Class index out of range");
                        }

                        size_t tp = 0, fp = 0;

                        for (size_t i = 0; i < batch_size; ++i) {
                            // En yüksek olasýlýklý sýnýfý bul
                            size_t pred_class = 0;
                            T max_prob = predictions.at({ i, 0 });

                            for (size_t j = 1; j < num_classes; ++j) {
                                if (predictions.at({ i, j }) > max_prob) {
                                    max_prob = predictions.at({ i, j });
                                    pred_class = j;
                                }
                            }

                            // Gerçek sýnýfý belirle
                            size_t true_class;
                            if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                                true_class = 0;
                                T max_val = targets.at({ i, 0 });

                                for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                    if (targets.at({ i, j }) > max_val) {
                                        max_val = targets.at({ i, j });
                                        true_class = j;
                                    }
                                }
                            }
                            else {
                                true_class = static_cast<size_t>(targets.at({ i }));
                            }

                            // Belirtilen sýnýf için TP/FP güncelle
                            if (pred_class == class_idx_) {
                                if (true_class == class_idx_) {
                                    tp++;
                                }
                                else {
                                    fp++;
                                }
                            }
                        }

                        return (tp + fp > 0) ? static_cast<T>(tp) / static_cast<T>(tp + fp) : T(0);
                    }
                }
                // Ýkili sýnýflandýrma için
                else {
                    size_t tp = 0, fp = 0;

                    for (size_t i = 0; i < batch_size; ++i) {
                        bool pred = predictions.at({ i }) >= T(0.5);
                        bool target = targets.at({ i }) >= T(0.5);

                        if (pred) {
                            if (target) {
                                tp++;
                            }
                            else {
                                fp++;
                            }
                        }
                    }

                    return (tp + fp > 0) ? static_cast<T>(tp) / static_cast<T>(tp + fp) : T(0);
                }
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Batch için hesapla ve sonuçlarý birikimli olarak güncelle
                if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                    // Çok sýnýflý durum
                    size_t batch_size = predictions.shape().dim(0);
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    for (size_t i = 0; i < batch_size; ++i) {
                        // En yüksek olasýlýklý sýnýfý bul
                        size_t pred_class = 0;
                        T max_prob = predictions.at({ i, 0 });

                        for (size_t j = 1; j < num_classes; ++j) {
                            if (predictions.at({ i, j }) > max_prob) {
                                max_prob = predictions.at({ i, j });
                                pred_class = j;
                            }
                        }

                        // Gerçek sýnýfý belirle
                        size_t true_class;
                        if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                            true_class = 0;
                            T max_val = targets.at({ i, 0 });

                            for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                if (targets.at({ i, j }) > max_val) {
                                    max_val = targets.at({ i, j });
                                    true_class = j;
                                }
                            }
                        }
                        else {
                            true_class = static_cast<size_t>(targets.at({ i }));
                        }

                        if (!average_ && pred_class == class_idx_) {
                            if (true_class == class_idx_) {
                                true_positives_++;
                            }
                            else {
                                false_positives_++;
                            }
                        }
                    }
                }
                else {
                    // Ýkili sýnýflandýrma
                    size_t batch_size = predictions.shape().dim(0);

                    for (size_t i = 0; i < batch_size; ++i) {
                        bool pred = predictions.at({ i }) >= T(0.5);
                        bool target = targets.at({ i }) >= T(0.5);

                        if (pred) {
                            if (target) {
                                true_positives_++;
                            }
                            else {
                                false_positives_++;
                            }
                        }
                    }
                }
            }

            T result() const override {
                if (average_) {
                    // Ortalama sonuç için, compute() metodu her çaðrýldýðýnda hesaplanmalý
                    // Bu sebeple birikimli hesaplama burada kullanýlamaz
                    return T(0);
                }
                else {
                    // Belirli bir sýnýf için birikimli hesaplama
                    return (true_positives_ + false_positives_ > 0) ?
                        static_cast<T>(true_positives_) / static_cast<T>(true_positives_ + false_positives_) :
                        T(0);
                }
            }

            void reset() override {
                true_positives_ = 0;
                false_positives_ = 0;
            }

            std::string name() const override {
                if (average_) {
                    return "Average Precision";
                }
                else {
                    return "Precision (Class " + std::to_string(class_idx_) + ")";
                }
            }

            bool higher_is_better() const override {
                return true; // Precision için yüksek deðerler daha iyidir
            }

        private:
            size_t class_idx_;
            bool average_;
            size_t true_positives_;
            size_t false_positives_;
            size_t class_count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.ndim() == 0 || targets.ndim() == 0) {
                    throw std::invalid_argument("Predictions and targets must have at least 1 dimension");
                }

                if (predictions.shape().dim(0) != targets.shape().dim(0)) {
                    throw std::invalid_argument("Predictions and targets must have same batch size");
                }
            }
        };

        // Recall (Duyarlýlýk) metriði
        template<typename T>
        class Recall : public Metric<T> {
        public:
            // Tek sýnýf veya tüm sýnýflarýn ortalamasý için
            Recall(size_t class_idx = 0, bool average = true)
                : class_idx_(class_idx), average_(average), true_positives_(0),
                false_negatives_(0), class_count_(0) {
            }

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t batch_size = predictions.shape().dim(0);

                // Çok sýnýflý durum için
                if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    // Tüm sýnýflar için ortalama hesapla
                    if (average_) {
                        T total_recall = T(0);
                        size_t valid_classes = 0;

                        for (size_t c = 0; c < num_classes; ++c) {
                            size_t tp = 0, fn = 0;

                            for (size_t i = 0; i < batch_size; ++i) {
                                // En yüksek olasýlýklý sýnýfý bul
                                size_t pred_class = 0;
                                T max_prob = predictions.at({ i, 0 });

                                for (size_t j = 1; j < num_classes; ++j) {
                                    if (predictions.at({ i, j }) > max_prob) {
                                        max_prob = predictions.at({ i, j });
                                        pred_class = j;
                                    }
                                }

                                // Gerçek sýnýfý belirle
                                size_t true_class;
                                if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                                    true_class = 0;
                                    T max_val = targets.at({ i, 0 });

                                    for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                        if (targets.at({ i, j }) > max_val) {
                                            max_val = targets.at({ i, j });
                                            true_class = j;
                                        }
                                    }
                                }
                                else {
                                    true_class = static_cast<size_t>(targets.at({ i }));
                                }

                                // Þu anki sýnýf için TP/FN güncelle
                                if (true_class == c) {
                                    if (pred_class == c) {
                                        tp++;
                                    }
                                    else {
                                        fn++;
                                    }
                                }
                            }

                            // Bu sýnýf için recall hesapla
                            if (tp + fn > 0) {
                                total_recall += static_cast<T>(tp) / static_cast<T>(tp + fn);
                                valid_classes++;
                            }
                        }

                        return valid_classes > 0 ? total_recall / static_cast<T>(valid_classes) : T(0);
                    }
                    // Sadece belirtilen sýnýf için hesapla
                    else {
                        if (class_idx_ >= num_classes) {
                            throw std::invalid_argument("Class index out of range");
                        }

                        size_t tp = 0, fn = 0;

                        for (size_t i = 0; i < batch_size; ++i) {
                            // En yüksek olasýlýklý sýnýfý bul
                            size_t pred_class = 0;
                            T max_prob = predictions.at({ i, 0 });

                            for (size_t j = 1; j < num_classes; ++j) {
                                if (predictions.at({ i, j }) > max_prob) {
                                    max_prob = predictions.at({ i, j });
                                    pred_class = j;
                                }
                            }

                            // Gerçek sýnýfý belirle
                            size_t true_class;
                            if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                                true_class = 0;
                                T max_val = targets.at({ i, 0 });

                                for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                    if (targets.at({ i, j }) > max_val) {
                                        max_val = targets.at({ i, j });
                                        true_class = j;
                                    }
                                }
                            }
                            else {
                                true_class = static_cast<size_t>(targets.at({ i }));
                            }

                            // Belirtilen sýnýf için TP/FN güncelle
                            if (true_class == class_idx_) {
                                if (pred_class == class_idx_) {
                                    tp++;
                                }
                                else {
                                    fn++;
                                }
                            }
                        }

                        return (tp + fn > 0) ? static_cast<T>(tp) / static_cast<T>(tp + fn) : T(0);
                    }
                }
                // Ýkili sýnýflandýrma için
                else {
                    size_t tp = 0, fn = 0;

                    for (size_t i = 0; i < batch_size; ++i) {
                        bool pred = predictions.at({ i }) >= T(0.5);
                        bool target = targets.at({ i }) >= T(0.5);

                        if (target) {
                            if (pred) {
                                tp++;
                            }
                            else {
                                fn++;
                            }
                        }
                    }

                    return (tp + fn > 0) ? static_cast<T>(tp) / static_cast<T>(tp + fn) : T(0);
                }
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Batch için hesapla ve sonuçlarý birikimli olarak güncelle
                if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                    // Çok sýnýflý durum
                    size_t batch_size = predictions.shape().dim(0);
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    for (size_t i = 0; i < batch_size; ++i) {
                        // En yüksek olasýlýklý sýnýfý bul
                        size_t pred_class = 0;
                        T max_prob = predictions.at({ i, 0 });

                        for (size_t j = 1; j < num_classes; ++j) {
                            if (predictions.at({ i, j }) > max_prob) {
                                max_prob = predictions.at({ i, j });
                                pred_class = j;
                            }
                        }

                        // Gerçek sýnýfý belirle
                        size_t true_class;
                        if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                            true_class = 0;
                            T max_val = targets.at({ i, 0 });

                            for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                                if (targets.at({ i, j }) > max_val) {
                                    max_val = targets.at({ i, j });
                                    true_class = j;
                                }
                            }
                        }
                        else {
                            true_class = static_cast<size_t>(targets.at({ i }));
                        }

                        if (!average_ && true_class == class_idx_) {
                            if (pred_class == class_idx_) {
                                true_positives_++;
                            }
                            else {
                                false_negatives_++;
                            }
                        }
                    }
                }
                else {
                    // Ýkili sýnýflandýrma
                    size_t batch_size = predictions.shape().dim(0);

                    for (size_t i = 0; i < batch_size; ++i) {
                        bool pred = predictions.at({ i }) >= T(0.5);
                        bool target = targets.at({ i }) >= T(0.5);

                        if (target) {
                            if (pred) {
                                true_positives_++;
                            }
                            else {
                                false_negatives_++;
                            }
                        }
                    }
                }
            }

            T result() const override {
                if (average_) {
                    // Ortalama sonuç için, compute() metodu her çaðrýldýðýnda hesaplanmalý
                    // Bu sebeple birikimli hesaplama burada kullanýlamaz
                    return T(0);
                }
                else {
                    // Belirli bir sýnýf için birikimli hesaplama
                    return (true_positives_ + false_negatives_ > 0) ?
                        static_cast<T>(true_positives_) / static_cast<T>(true_positives_ + false_negatives_) :
                        T(0);
                }
            }

            void reset() override {
                true_positives_ = 0;
                false_negatives_ = 0;
            }

            std::string name() const override {
                if (average_) {
                    return "Average Recall";
                }
                else {
                    return "Recall (Class " + std::to_string(class_idx_) + ")";
                }
            }

            bool higher_is_better() const override {
                return true; // Recall için yüksek deðerler daha iyidir
            }

        private:
            size_t class_idx_;
            bool average_;
            size_t true_positives_;
            size_t false_negatives_;
            size_t class_count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.ndim() == 0 || targets.ndim() == 0) {
                    throw std::invalid_argument("Predictions and targets must have at least 1 dimension");
                }

                if (predictions.shape().dim(0) != targets.shape().dim(0)) {
                    throw std::invalid_argument("Predictions and targets must have same batch size");
                }
            }
        };

        // F1 Score metriði
        template<typename T>
        class F1Score : public Metric<T> {
        public:
            // Tek sýnýf veya tüm sýnýflarýn ortalamasý için
            F1Score(size_t class_idx = 0, bool average = true)
                : precision_(class_idx, average), recall_(class_idx, average),
                class_idx_(class_idx), average_(average) {
            }

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Precision ve Recall hesapla
                T prec = precision_.compute(predictions, targets);
                T rec = recall_.compute(predictions, targets);

                // F1 Score hesapla: 2 * (precision * recall) / (precision + recall)
                if (prec + rec < epsilon<T>) {
                    return T(0); // Sýfýra bölme hatasýný önle
                }

                return T(2) * prec * rec / (prec + rec);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Precision ve Recall metriklerini güncelle
                precision_.update(predictions, targets);
                recall_.update(predictions, targets);
            }

            T result() const override {
                T prec = precision_.result();
                T rec = recall_.result();

                if (prec + rec < epsilon<T>) {
                    return T(0); // Sýfýra bölme hatasýný önle
                }

                return T(2) * prec * rec / (prec + rec);
            }

            void reset() override {
                precision_.reset();
                recall_.reset();
            }

            std::string name() const override {
                if (average_) {
                    return "Average F1-Score";
                }
                else {
                    return "F1-Score (Class " + std::to_string(class_idx_) + ")";
                }
            }

            bool higher_is_better() const override {
                return true; // F1 Score için yüksek deðerler daha iyidir
            }

        private:
            Precision<T> precision_;
            Recall<T> recall_;
            size_t class_idx_;
            bool average_;
        };

        //==========================================================================
        // REGRESYON METRÝKLERÝ
        //==========================================================================

        // Mean Squared Error (MSE) metriði
        template<typename T>
        class MeanSquaredError : public Metric<T> {
        public:
            MeanSquaredError() : sum_squared_error_(0), count_(0) {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                T sum_squared_error = T(0);
                size_t count = predictions.size();

                // Tüm elemanlar için kare hatayý hesapla
                for (size_t i = 0; i < count; ++i) {
                    T diff = predictions.data()[i] - targets.data()[i];
                    sum_squared_error += diff * diff;
                }

                return sum_squared_error / static_cast<T>(count);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t count = predictions.size();

                // Toplam kare hatayý hesapla
                for (size_t i = 0; i < count; ++i) {
                    T diff = predictions.data()[i] - targets.data()[i];
                    sum_squared_error_ += diff * diff;
                }

                count_ += count;
            }

            T result() const override {
                if (count_ == 0) {
                    return T(0);
                }

                return sum_squared_error_ / static_cast<T>(count_);
            }

            void reset() override {
                sum_squared_error_ = T(0);
                count_ = 0;
            }

            std::string name() const override {
                return "Mean Squared Error";
            }

            bool higher_is_better() const override {
                return false; // MSE için düþük deðerler daha iyidir
            }

        private:
            T sum_squared_error_;
            size_t count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.shape() != targets.shape()) {
                    throw std::invalid_argument("Predictions and targets must have same shape");
                }
            }
        };

        // Mean Absolute Error (MAE) metriði
        template<typename T>
        class MeanAbsoluteError : public Metric<T> {
        public:
            MeanAbsoluteError() : sum_absolute_error_(0), count_(0) {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                T sum_absolute_error = T(0);
                size_t count = predictions.size();

                // Tüm elemanlar için mutlak hatayý hesapla
                for (size_t i = 0; i < count; ++i) {
                    sum_absolute_error += std::abs(predictions.data()[i] - targets.data()[i]);
                }

                return sum_absolute_error / static_cast<T>(count);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t count = predictions.size();

                // Toplam mutlak hatayý hesapla
                for (size_t i = 0; i < count; ++i) {
                    sum_absolute_error_ += std::abs(predictions.data()[i] - targets.data()[i]);
                }

                count_ += count;
            }

            T result() const override {
                if (count_ == 0) {
                    return T(0);
                }

                return sum_absolute_error_ / static_cast<T>(count_);
            }

            void reset() override {
                sum_absolute_error_ = T(0);
                count_ = 0;
            }

            std::string name() const override {
                return "Mean Absolute Error";
            }

            bool higher_is_better() const override {
                return false; // MAE için düþük deðerler daha iyidir
            }

        private:
            T sum_absolute_error_;
            size_t count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.shape() != targets.shape()) {
                    throw std::invalid_argument("Predictions and targets must have same shape");
                }
            }
        };

        // Root Mean Squared Error (RMSE) metriði
        template<typename T>
        class RootMeanSquaredError : public Metric<T> {
        public:
            RootMeanSquaredError() : mse_() {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                T mse_value = mse_.compute(predictions, targets);
                return std::sqrt(mse_value);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                mse_.update(predictions, targets);
            }

            T result() const override {
                return std::sqrt(mse_.result());
            }

            void reset() override {
                mse_.reset();
            }

            std::string name() const override {
                return "Root Mean Squared Error";
            }

            bool higher_is_better() const override {
                return false; // RMSE için düþük deðerler daha iyidir
            }

        private:
            MeanSquaredError<T> mse_;
        };

        //==========================================================================
        // ÝLERÝ METRÝKLER
        //==========================================================================

        // ROC AUC metriði
        template<typename T>
        class ROCAUC : public Metric<T> {
        public:
            ROCAUC() : predictions_(), targets_() {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                // Tüm tahminleri ve hedefleri topla
                size_t size = predictions.size();
                std::vector<std::pair<T, bool>> scores_and_labels;
                scores_and_labels.reserve(size);

                for (size_t i = 0; i < size; ++i) {
                    scores_and_labels.emplace_back(predictions.data()[i], targets.data()[i] > T(0.5));
                }

                // Tahminlere göre sýrala
                std::sort(scores_and_labels.begin(), scores_and_labels.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // ROC eðrisi oluþtur ve AUC hesapla
                size_t total_positive = 0;
                size_t total_negative = 0;

                for (const auto& pair : scores_and_labels) {
                    if (pair.second) {
                        total_positive++;
                    }
                    else {
                        total_negative++;
                    }
                }

                if (total_positive == 0 || total_negative == 0) {
                    return T(0.5); // Tek bir sýnýf varsa, AUC 0.5 olarak kabul edilir
                }

                // AUC hesapla (trapezoidal yöntem)
                T auc = T(0);
                size_t true_positives = 0;
                size_t false_positives = 0;
                T prev_tpr = T(0);
                T prev_fpr = T(0);

                for (const auto& pair : scores_and_labels) {
                    if (pair.second) {
                        true_positives++;
                    }
                    else {
                        false_positives++;
                    }

                    T tpr = static_cast<T>(true_positives) / static_cast<T>(total_positive);
                    T fpr = static_cast<T>(false_positives) / static_cast<T>(total_negative);

                    // Trapezoid alanýný ekle
                    auc += (fpr - prev_fpr) * (tpr + prev_tpr) * T(0.5);

                    prev_tpr = tpr;
                    prev_fpr = fpr;
                }

                return auc;
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                // Tüm örnekleri sakla
                for (size_t i = 0; i < predictions.size(); ++i) {
                    predictions_.push_back(predictions.data()[i]);
                    targets_.push_back(targets.data()[i]);
                }
            }

            T result() const override {
                if (predictions_.empty()) {
                    return T(0.5);
                }

                // Tüm tahminleri ve hedefleri topla
                size_t size = predictions_.size();
                std::vector<std::pair<T, bool>> scores_and_labels;
                scores_and_labels.reserve(size);

                for (size_t i = 0; i < size; ++i) {
                    scores_and_labels.emplace_back(predictions_[i], targets_[i] > T(0.5));
                }

                // Tahminlere göre sýrala
                std::sort(scores_and_labels.begin(), scores_and_labels.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // ROC eðrisi oluþtur ve AUC hesapla
                size_t total_positive = 0;
                size_t total_negative = 0;

                for (const auto& pair : scores_and_labels) {
                    if (pair.second) {
                        total_positive++;
                    }
                    else {
                        total_negative++;
                    }
                }

                if (total_positive == 0 || total_negative == 0) {
                    return T(0.5); // Tek bir sýnýf varsa, AUC 0.5 olarak kabul edilir
                }

                // AUC hesapla (trapezoidal yöntem)
                T auc = T(0);
                size_t true_positives = 0;
                size_t false_positives = 0;
                T prev_tpr = T(0);
                T prev_fpr = T(0);

                for (const auto& pair : scores_and_labels) {
                    if (pair.second) {
                        true_positives++;
                    }
                    else {
                        false_positives++;
                    }

                    T tpr = static_cast<T>(true_positives) / static_cast<T>(total_positive);
                    T fpr = static_cast<T>(false_positives) / static_cast<T>(total_negative);

                    // Trapezoid alanýný ekle
                    auc += (fpr - prev_fpr) * (tpr + prev_tpr) * T(0.5);

                    prev_tpr = tpr;
                    prev_fpr = fpr;
                }

                return auc;
            }

            void reset() override {
                predictions_.clear();
                targets_.clear();
            }

            std::string name() const override {
                return "ROC AUC";
            }

            bool higher_is_better() const override {
                return true; // AUC için yüksek deðerler daha iyidir
            }

        private:
            std::vector<T> predictions_;
            std::vector<T> targets_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.size() != targets.size()) {
                    throw std::invalid_argument("Predictions and targets must have same size");
                }
            }
        };

        // Log Loss metriði
        template<typename T>
        class LogLoss : public Metric<T> {
        public:
            LogLoss() : sum_log_loss_(0), count_(0) {}

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                T sum_log_loss = T(0);
                size_t count = predictions.shape().dim(0);

                // Ýkili sýnýflandýrma durumu
                if (predictions.ndim() == 1 || predictions.shape().dim(1) == 1) {
                    for (size_t i = 0; i < count; ++i) {
                        T pred = std::max(std::min(predictions.at({ i }), T(1) - epsilon<T>), epsilon<T>);
                        T target = targets.at({ i });

                        sum_log_loss -= target * std::log(pred) +
                            (T(1) - target) * std::log(T(1) - pred);
                    }
                }
                // Çok sýnýflý durum
                else {
                    size_t num_classes = predictions.shape().dim(1);

                    for (size_t i = 0; i < count; ++i) {
                        for (size_t j = 0; j < num_classes; ++j) {
                            T pred = std::max(std::min(predictions.at({ i, j }), T(1) - epsilon<T>), epsilon<T>);
                            T target = targets.ndim() == 1 ? (targets.at({ i }) == j ? T(1) : T(0)) :
                                targets.at({ i, j });

                            if (target > epsilon<T>) {
                                sum_log_loss -= target * std::log(pred);
                            }
                        }
                    }
                }

                return sum_log_loss / static_cast<T>(count);
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                T batch_log_loss = compute(predictions, targets);
                size_t batch_size = predictions.shape().dim(0);

                sum_log_loss_ += batch_log_loss * batch_size;
                count_ += batch_size;
            }

            T result() const override {
                if (count_ == 0) {
                    return T(0);
                }

                return sum_log_loss_ / static_cast<T>(count_);
            }

            void reset() override {
                sum_log_loss_ = T(0);
                count_ = 0;
            }

            std::string name() const override {
                return "Log Loss";
            }

            bool higher_is_better() const override {
                return false; // Log Loss için düþük deðerler daha iyidir
            }

        private:
            T sum_log_loss_;
            size_t count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.shape().dim(0) != targets.shape().dim(0)) {
                    throw std::invalid_argument("Predictions and targets must have same batch size");
                }

                if (predictions.ndim() > 1 && targets.ndim() > 1 &&
                    predictions.shape().dim(1) != targets.shape().dim(1)) {
                    throw std::invalid_argument("Predictions and targets must have compatible class dimensions");
                }
            }
        };

        // Intersection over Union (IoU) metriði - Segmentasyon ve Obje Tespiti için
        template<typename T>
        class IoU : public Metric<T> {
        public:
            IoU(size_t class_idx = 0, bool average = true)
                : class_idx_(class_idx), average_(average),
                intersection_(0), union_(0), class_count_(0) {
            }

            T compute(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                size_t batch_size = predictions.shape().dim(0);

                // Çok sýnýflý durum
                if (predictions.ndim() > 2 && predictions.shape().dim(1) > 1) {
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    // Tüm sýnýflar için ortalama hesapla
                    if (average_) {
                        T total_iou = T(0);
                        size_t valid_classes = 0;

                        for (size_t c = 0; c < num_classes; ++c) {
                            T intersection = T(0);
                            T union_area = T(0);

                            for (size_t i = 0; i < batch_size; ++i) {
                                // Tahmin ve hedefin her pikseli/hücresi için IoU hesapla
                                for (size_t j = 2; j < predictions.ndim(); ++j) {
                                    for (size_t k = 0; k < predictions.shape().dim(j); ++k) {
                                        std::vector<size_t> idx(predictions.ndim(), 0);
                                        idx[0] = i;
                                        idx[1] = c;
                                        idx[j] = k;

                                        bool pred = predictions.at(idx) >= T(0.5);
                                        bool target = targets.at(idx) >= T(0.5);

                                        if (pred && target) intersection += T(1);
                                        if (pred || target) union_area += T(1);
                                    }
                                }
                            }

                            if (union_area > epsilon<T>) {
                                total_iou += intersection / union_area;
                                valid_classes++;
                            }
                        }

                        return valid_classes > 0 ? total_iou / static_cast<T>(valid_classes) : T(0);
                    }
                    // Sadece belirtilen sýnýf için hesapla
                    else {
                        if (class_idx_ >= num_classes) {
                            throw std::invalid_argument("Class index out of range");
                        }

                        T intersection = T(0);
                        T union_area = T(0);

                        for (size_t i = 0; i < batch_size; ++i) {
                            // Tahmin ve hedefin her pikseli/hücresi için IoU hesapla
                            for (size_t j = 2; j < predictions.ndim(); ++j) {
                                for (size_t k = 0; k < predictions.shape().dim(j); ++k) {
                                    std::vector<size_t> idx(predictions.ndim(), 0);
                                    idx[0] = i;
                                    idx[1] = class_idx_;
                                    idx[j] = k;

                                    bool pred = predictions.at(idx) >= T(0.5);
                                    bool target = targets.at(idx) >= T(0.5);

                                    if (pred && target) intersection += T(1);
                                    if (pred || target) union_area += T(1);
                                }
                            }
                        }

                        return union_area > epsilon<T> ? intersection / union_area : T(0);
                    }
                }
                // Ýkili segmentasyon için
                else {
                    T intersection = T(0);
                    T union_area = T(0);

                    // Her örnek için IoU hesapla
                    for (size_t i = 0; i < predictions.size(); ++i) {
                        bool pred = predictions.data()[i] >= T(0.5);
                        bool target = targets.data()[i] >= T(0.5);

                        if (pred && target) intersection += T(1);
                        if (pred || target) union_area += T(1);
                    }

                    return union_area > epsilon<T> ? intersection / union_area : T(0);
                }
            }

            void update(const Tensor<T>& predictions, const Tensor<T>& targets) override {
                // Giriþ doðrulama
                validate_inputs(predictions, targets);

                // Çok sýnýflý durum
                if (predictions.ndim() > 2 && predictions.shape().dim(1) > 1) {
                    size_t num_classes = predictions.shape().dim(1);
                    class_count_ = num_classes;

                    if (!average_) {
                        // Sadece belirtilen sýnýf için IoU bileþenlerini topla
                        for (size_t i = 0; i < predictions.size() / num_classes; ++i) {
                            std::vector<size_t> idx(predictions.ndim(), 0);
                            idx[1] = class_idx_;

                            bool pred = predictions.at(idx) >= T(0.5);
                            bool target = targets.at(idx) >= T(0.5);

                            if (pred && target) intersection_ += T(1);
                            if (pred || target) union_ += T(1);
                        }
                    }
                }
                // Ýkili segmentasyon için
                else {
                    // IoU bileþenlerini topla
                    for (size_t i = 0; i < predictions.size(); ++i) {
                        bool pred = predictions.data()[i] >= T(0.5);
                        bool target = targets.data()[i] >= T(0.5);

                        if (pred && target) intersection_ += T(1);
                        if (pred || target) union_ += T(1);
                    }
                }
            }

            T result() const override {
                if (average_) {
                    // Ortalama IoU için, compute() metodu her çaðrýldýðýnda hesaplanmalý
                    return T(0);
                }
                else {
                    return union_ > epsilon<T> ? intersection_ / union_ : T(0);
                }
            }

            void reset() override {
                intersection_ = T(0);
                union_ = T(0);
            }

            std::string name() const override {
                if (average_) {
                    return "Mean IoU";
                }
                else {
                    return "IoU (Class " + std::to_string(class_idx_) + ")";
                }
            }

            bool higher_is_better() const override {
                return true; // IoU için yüksek deðerler daha iyidir
            }

        private:
            size_t class_idx_;
            bool average_;
            T intersection_;
            T union_;
            size_t class_count_;

            void validate_inputs(const Tensor<T>& predictions, const Tensor<T>& targets) const {
                if (predictions.shape() != targets.shape()) {
                    throw std::invalid_argument("Predictions and targets must have same shape");
                }
            }
        };

        //==========================================================================
        // METRÝK TOPLAYICI
        //==========================================================================

        // Birden çok metriði takip etmek için koleksiyon
        template<typename T>
        class MetricCollection {
        public:
            // Varsayýlan constructor
            MetricCollection() = default;

            // Metrik ekle
            void add_metric(const std::string& name, std::shared_ptr<Metric<T>> metric) {
                metrics_[name] = metric;
            }

            // Ýsimle metriðe eriþ
            std::shared_ptr<Metric<T>> get_metric(const std::string& name) {
                auto it = metrics_.find(name);
                if (it == metrics_.end()) {
                    throw std::out_of_range("Metric not found: " + name);
                }
                return it->second;
            }

            // Tüm metrikleri hesapla
            std::unordered_map<std::string, T> compute_all(
                const Tensor<T>& predictions, const Tensor<T>& targets) {

                std::unordered_map<std::string, T> results;

                for (const auto& [name, metric] : metrics_) {
                    results[name] = metric->compute(predictions, targets);
                }

                return results;
            }

            // Tüm metrikleri güncelle
            void update_all(const Tensor<T>& predictions, const Tensor<T>& targets) {
                for (auto& [name, metric] : metrics_) {
                    metric->update(predictions, targets);
                }
            }

            // Tüm metriklerin sonuçlarýný al
            std::unordered_map<std::string, T> get_all_results() const {
                std::unordered_map<std::string, T> results;

                for (const auto& [name, metric] : metrics_) {
                    results[name] = metric->result();
                }

                return results;
            }

            // Tüm metrikleri sýfýrla
            void reset_all() {
                for (auto& [name, metric] : metrics_) {
                    metric->reset();
                }
            }

            // Metrik sayýsýný al
            size_t size() const {
                return metrics_.size();
            }

            // Tüm metrik isimlerini al
            std::vector<std::string> get_metric_names() const {
                std::vector<std::string> names;
                names.reserve(metrics_.size());

                for (const auto& [name, _] : metrics_) {
                    names.push_back(name);
                }

                return names;
            }

            // Metriklere iterator ile eriþim
            auto begin() { return metrics_.begin(); }
            auto end() { return metrics_.end(); }
            auto begin() const { return metrics_.begin(); }
            auto end() const { return metrics_.end(); }

        private:
            std::unordered_map<std::string, std::shared_ptr<Metric<T>>> metrics_;
        };

        //==========================================================================
        // YARDIMCI FONKSÝYONLAR
        //==========================================================================

        // Karmaþýklýk matrisi (Confusion Matrix) hesapla
        template<typename T>
        Tensor<size_t> confusion_matrix(const Tensor<T>& predictions, const Tensor<T>& targets, size_t num_classes) {
            // Karmaþýklýk matrisi oluþtur [num_classes, num_classes]
            Tensor<size_t> cm(std::vector<size_t>{num_classes, num_classes}, 0);

            // Tahminlerin ve hedeflerin boyutlarýna göre iþlem yap
            if (predictions.ndim() == 2 && predictions.shape().dim(1) > 1) {
                // Çok sýnýflý tahminler
                size_t batch_size = predictions.shape().dim(0);

                for (size_t i = 0; i < batch_size; ++i) {
                    // En yüksek olasýlýklý sýnýfý bul
                    size_t pred_class = 0;
                    T max_prob = predictions.at({ i, 0 });

                    for (size_t j = 1; j < predictions.shape().dim(1); ++j) {
                        if (predictions.at({ i, j }) > max_prob) {
                            max_prob = predictions.at({ i, j });
                            pred_class = j;
                        }
                    }

                    // Gerçek sýnýfý belirle
                    size_t true_class;
                    if (targets.ndim() == 2 && targets.shape().dim(1) > 1) {
                        true_class = 0;
                        T max_val = targets.at({ i, 0 });

                        for (size_t j = 1; j < targets.shape().dim(1); ++j) {
                            if (targets.at({ i, j }) > max_val) {
                                max_val = targets.at({ i, j });
                                true_class = j;
                            }
                        }
                    }
                    else {
                        true_class = static_cast<size_t>(targets.at({ i }));
                    }

                    // Sýnýf sýnýrlarýný kontrol et
                    if (true_class < num_classes && pred_class < num_classes) {
                        cm.at({ true_class, pred_class })++;
                    }
                }
            }
            else {
                // Ýkili sýnýflandýrma
                size_t batch_size = predictions.shape().dim(0);

                for (size_t i = 0; i < batch_size; ++i) {
                    size_t pred_class = predictions.at({ i }) >= T(0.5) ? 1 : 0;
                    size_t true_class = targets.at({ i }) >= T(0.5) ? 1 : 0;

                    cm.at({ true_class, pred_class })++;
                }
            }

            return cm;
        }

        // Metrik raporu oluþtur (sýnýflandýrma için)
        template<typename T>
        std::string classification_report(const Tensor<T>& predictions, const Tensor<T>& targets,
            const std::vector<std::string>& class_names = {}) {
            // Sýnýf sayýsýný belirle
            size_t num_classes;
            if (predictions.ndim() == 2) {
                num_classes = predictions.shape().dim(1);
            }
            else {
                // Ýkili sýnýflandýrma için
                num_classes = 2;
            }

            // Karmaþýklýk matrisi hesapla
            Tensor<size_t> cm = confusion_matrix(predictions, targets, num_classes);

            // Precision, Recall ve F1 Score hesapla
            std::vector<T> precision(num_classes, T(0));
            std::vector<T> recall(num_classes, T(0));
            std::vector<T> f1(num_classes, T(0));
            std::vector<size_t> support(num_classes, 0);

            // Sýnýf baþýna metrikleri hesapla
            for (size_t i = 0; i < num_classes; ++i) {
                size_t tp = cm.at({ i, i });
                size_t class_total = 0;
                size_t pred_total = 0;

                for (size_t j = 0; j < num_classes; ++j) {
                    class_total += cm.at({ i, j }); // Gerçek i sýnýfý toplamý
                    pred_total += cm.at({ j, i });  // Tahmin i sýnýfý toplamý
                }

                support[i] = class_total;

                if (pred_total > 0) {
                    precision[i] = static_cast<T>(tp) / static_cast<T>(pred_total);
                }

                if (class_total > 0) {
                    recall[i] = static_cast<T>(tp) / static_cast<T>(class_total);
                }

                if (precision[i] + recall[i] > epsilon<T>) {
                    f1[i] = T(2) * precision[i] * recall[i] / (precision[i] + recall[i]);
                }
            }

            // Ortalama deðerleri hesapla
            T avg_precision = T(0);
            T avg_recall = T(0);
            T avg_f1 = T(0);
            size_t total_support = 0;

            for (size_t i = 0; i < num_classes; ++i) {
                if (support[i] > 0) {
                    avg_precision += precision[i] * static_cast<T>(support[i]);
                    avg_recall += recall[i] * static_cast<T>(support[i]);
                    avg_f1 += f1[i] * static_cast<T>(support[i]);
                    total_support += support[i];
                }
            }

            if (total_support > 0) {
                avg_precision /= static_cast<T>(total_support);
                avg_recall /= static_cast<T>(total_support);
                avg_f1 /= static_cast<T>(total_support);
            }

            // Rapor oluþtur
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);

            // Üst baþlýk
            ss << std::setw(15) << "Class" << std::setw(12) << "Precision"
                << std::setw(12) << "Recall" << std::setw(12) << "F1-Score"
                << std::setw(12) << "Support" << std::endl;

            ss << std::string(63, '-') << std::endl;

            // Sýnýf baþýna metrikler
            for (size_t i = 0; i < num_classes; ++i) {
                std::string class_name;
                if (i < class_names.size()) {
                    class_name = class_names[i];
                }
                else {
                    class_name = "Class " + std::to_string(i);
                }

                ss << std::setw(15) << class_name
                    << std::setw(12) << precision[i]
                    << std::setw(12) << recall[i]
                        << std::setw(12) << f1[i]
                            << std::setw(12) << support[i] << std::endl;
            }

            ss << std::string(63, '-') << std::endl;

            // Ortalama deðerler
            ss << std::setw(15) << "Weighted Avg"
                << std::setw(12) << avg_precision
                << std::setw(12) << avg_recall
                << std::setw(12) << avg_f1
                << std::setw(12) << total_support << std::endl;

            return ss.str();
        }

        //==========================================================================
        // HELPER FONKSÝYONLAR (FACTORY PATTERNÝ)
        //==========================================================================

        // Yaygýn metrik koleksiyonlarý için yardýmcý fonksiyonlar

        // Sýnýflandýrma metrikleri koleksiyonu oluþtur
        template<typename T>
        MetricCollection<T> create_classification_metrics() {
            MetricCollection<T> metrics;

            metrics.add_metric("accuracy", std::make_shared<Accuracy<T>>());
            metrics.add_metric("precision", std::make_shared<Precision<T>>(0, true));
            metrics.add_metric("recall", std::make_shared<Recall<T>>(0, true));
            metrics.add_metric("f1", std::make_shared<F1Score<T>>(0, true));

            return metrics;
        }

        // Regresyon metrikleri koleksiyonu oluþtur
        template<typename T>
        MetricCollection<T> create_regression_metrics() {
            MetricCollection<T> metrics;

            metrics.add_metric("mse", std::make_shared<MeanSquaredError<T>>());
            metrics.add_metric("mae", std::make_shared<MeanAbsoluteError<T>>());
            metrics.add_metric("rmse", std::make_shared<RootMeanSquaredError<T>>());

            return metrics;
        }

    } // namespace metrics
} // namespace tensor

#endif // METRICS_H