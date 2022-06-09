#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "des.hpp"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_PLUGIN(des_cpp)
{
    py::module m("des_cpp", "pybind11 example plugin");

    m.def("multi_server_queue", &multi_server_queue, "A function which adds two numbers");
    return m.ptr();
}