cmake_minimum_required(VERSION 3.12)
project(simple_dlpack)

add_subdirectory(pybind11)

pybind11_add_module(simple_dlpack simple_dlpack.cpp)
