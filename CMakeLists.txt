cmake_minimum_required(VERSION 3.12)
project(simple_dlpack)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(simple_dlpack simple_dlpack.cpp)
