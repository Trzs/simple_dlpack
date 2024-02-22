#include <tuple>
#include <iostream>
#include <pybind11/pybind11.h>
#include "dlpack/dlpack.h"

namespace py = pybind11;

void dlpack_destructor(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
    }

    // If the capsule has not been used, we need to delete it
    DLManagedTensor* dlpackTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    dlpackTensor->deleter(dlpackTensor);
    delete dlpackTensor;
}

struct DLPackAPI {
    static constexpr int size = 12;
    double container[size];

    py::capsule dlpack() {
        // Option to extend to multiple dimensions
        size_t numDims = 1;
        int64_t* shape = new int64_t[numDims];
        shape[0] = size;

        // Create a DLPack tensor
        DLManagedTensor* dlpackTensor = new DLManagedTensor;
        dlpackTensor->dl_tensor.data = static_cast<void*>(&container);
        dlpackTensor->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
        dlpackTensor->dl_tensor.device.device_id = 0;
        dlpackTensor->dl_tensor.ndim = numDims;
        dlpackTensor->dl_tensor.dtype = getDLPackDataType();
        dlpackTensor->dl_tensor.shape = shape;
        dlpackTensor->dl_tensor.strides = nullptr;
        dlpackTensor->dl_tensor.byte_offset = 0;
        dlpackTensor->manager_ctx = nullptr;
        dlpackTensor->deleter = [](DLManagedTensor* tensor) {
            delete[] tensor->dl_tensor.shape;
        };

        // Create a PyCapsule with the DLPack tensor
        return  py::capsule(dlpackTensor, "dltensor", dlpack_destructor);
    }

    DLDataType getDLPackDataType() {
        DLDataType dtype;
        dtype.code = kDLFloat;
        dtype.bits = sizeof(double) * 8;
        dtype.lanes = 1;
        return dtype;
    }

    void print_container() {
        std::cout << "C = [ ";
        for (int i=0; i<size; i++) {
            std::cout << container[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    void set_element(py::int_ index, py::float_ value) {
        if ((index>=size) || (index<0)) {
            throw std::runtime_error("Error: invalid index!");
        }
        container[index] = value;
    }

    std::tuple<int32_t, int32_t> dlpack_device() {
        return std::make_tuple(static_cast<int32_t>(DLDeviceType::kDLCPU), 0);
    }
};

PYBIND11_MODULE(simple_dlpack, m) {
    py::class_<DLPackAPI>(m, "simple_array")
        .def(py::init<>())
        .def("__dlpack__", &DLPackAPI::dlpack, "Part of DLPack API")
        .def("__dlpack_device__", &DLPackAPI::dlpack_device, "Part of DLPack API")
        .def("set", &DLPackAPI::set_element, "Set element[index] to value")
        .def("print", &DLPackAPI::print_container, "Print container")
    ;
}
