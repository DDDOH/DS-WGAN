cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)
set(CMAKE_PREFIX_PATH /opt/homebrew/Caskroom/miniforge/base/envs/dswgan/lib/python3.10/site-packages/torch/share/cmake)
add_executable(example-app main.cpp)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#set(Torch_DIR /opt/homebrew/Caskroom/miniforge/base/envs/dswgan/lib/python3.10/site-packages/torch/share/cmake/Torch)

target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
