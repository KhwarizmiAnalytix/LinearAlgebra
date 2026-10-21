cmake_minimum_required(VERSION 3.24)
project(LinearAlgebraInstallConsumer LANGUAGES CXX)

find_package(LinearAlgebra CONFIG REQUIRED)

add_executable(install_consumer main.cpp)
target_link_libraries(install_consumer PRIVATE LinearAlgebra::LinearAlgebra)

enable_testing()
add_test(NAME installed_package COMMAND install_consumer)
