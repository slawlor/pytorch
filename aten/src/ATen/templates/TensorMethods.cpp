#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

namespace at {

#define DEFINE_CAST(T, name)                                        \
   template <>                                                       \
   TORCH_API T* Tensor::data_ptr() const {                           \
     TORCH_CHECK(                                                    \
         scalar_type() == ScalarType::name,                          \
         "expected scalar type "                                     \
         #name                                                       \
         " but found ",                                              \
         scalar_type());                                             \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
 #undef DEFINE_ITEM

 } //namespace at
 #include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
namespace at {

// From build/ATen/RegisterCPU.cpp
// Inplace version
struct structured_add_out_inplace final : public at::native::structured_add_out {
    structured_add_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_add_out::set_output(output_idx, sizes, strides, options, names);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

at::Tensor & wrapper_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  structured_add_out_inplace op(self);
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return self;
}

at::Tensor & add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
return wrapper_add__Tensor(self, other, alpha);
}

} //namespace at
