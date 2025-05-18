// metrics_compat.h - Compatibility layer for metrics.h to work with Tensor.h
#ifndef METRICS_COMPAT_H
#define METRICS_COMPAT_H

#include "Tensor.h"

namespace tensor {
    namespace metrics_compat {

        // Adaptör sýnýf: Tensor<T> için .dim(idx) çaðrýlarýný destekler
        template<typename T>
        class TensorAdapter {
        public:
            explicit TensorAdapter(const Tensor<T>& tensor) : tensor_(tensor) {}

            // metrics.h'nin beklediði shape().dim(idx) arayüzünü saðlýyor
            class ShapeAdapter {
            public:
                explicit ShapeAdapter(const Tensor<T>& tensor) : tensor_(tensor) {}

                // .dim(idx) çaðrýsýný Tensor<T>::dim(idx)'e yönlendir
                size_t dim(size_t idx) const {
                    return tensor_.dim(idx);
                }

                // shape() için ek destek (gerekirse)
                const std::vector<size_t>& as_vector() const {
                    return tensor_.shape();
                }

            private:
                const Tensor<T>& tensor_;
            };

            // shape() çaðrýsýný ShapeAdapter'a dönüþtür
            ShapeAdapter shape() const {
                return ShapeAdapter(tensor_);
            }

            // Tensor metodlarýna doðrudan eriþim
            const Tensor<T>& get_tensor() const {
                return tensor_;
            }

            // metrics.h'deki at() ve diðer metodlar için vekil
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

        // metrics.h ile uyumluluk için adaptör oluþtur
        template<typename T>
        TensorAdapter<T> adapt_tensor(const Tensor<T>& tensor) {
            return TensorAdapter<T>(tensor);
        }

    } // namespace metrics_compat
} // namespace tensor

#endif // METRICS_COMPAT_H