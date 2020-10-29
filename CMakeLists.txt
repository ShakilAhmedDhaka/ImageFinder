cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(ImageFinder)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bow bow.cpp)
target_link_libraries(bow ${OpenCV_LIBS})