// metrics_compat.h - Compatibility layer for metrics.h to work with Tensor.h
#ifndef METRICS_COMPAT_H
#define METRICS_COMPAT_H

#include "Tensor.h"

namespace tensor {
    namespace metrics_compat {

        // Adapt�r s�n�f: Tensor<T> i�in .dim(idx) �a�r�lar�n� destekler
        template<typename T>
        class TensorAdapter {
        public:
            explicit TensorAdapter(const Tensor<T>& tensor) : tensor_(tensor) {}

            // metrics.h'nin bekledi�i shape().dim(idx) aray�z�n� sa�l�yor
            class ShapeAdapter {
            public:
                explicit ShapeAdapter(const Tensor<T>& tensor) : tensor_(tensor) {}

                // .dim(idx) �a�r�s�n� Tensor<T>::dim(idx)'e y�nlendir
                size_t dim(size_t idx) const {
                    return tensor_.dim(idx);
                }

                // shape() i�in ek destek (gerekirse)
                const std::vector<size_t>& as_vector() const {
                    return tensor_.shape();
                }

            private:
                const Tensor<T>& tensor_;
            };

            // shape() �a�r�s�n� ShapeAdapter'a d�n��t�r
            ShapeAdapter shape() const {
                return ShapeAdapter(tensor_);
            }

            // Tensor metodlar�na do�rudan eri�im
            const Tensor<T>& get_tensor() const {
                return tensor_;
            }

            // metrics.h'deki at() ve di�er metodlar i�in vekil
            T at(const std::vector<size_t>& indices) const {
                return tensor_.at(indices);
            }

            T at(size_t i) const {
                return tensor_.at(i);
            }

            T at(size_t i, size_t j) const {
                return tensor_.at(i, j);
            }

            size_t ndim() const {
                return tensor_.ndim();
            }

            size_t size() const {
                return tensor_.size();
            }

            const T* data() const {
                return tensor_.data();
            }

        private:
            const Tensor<T>& tensor_;
        };

        // metrics.h ile uyumluluk i�in adapt�r olu�tur
        template<typename T>
        TensorAdapter<T> adapt_tensor(const Tensor<T>& tensor) {
            return TensorAdapter<T>(tensor);
        }

    } // namespace metrics_compat
} // namespace tensor

#endif // METRICS_COMPAT_H