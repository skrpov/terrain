cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
project(Terrain)

find_package(OpenGL REQUIRED)

add_subdirectory(./vendor/glfw)
add_subdirectory(./vendor/glm)

add_executable(
    app 
    ./vendor/glad/src/glad.c
    ./src/main.cpp)

target_compile_options(
    app 
    PRIVATE 
    -Wall 
    -Wextra 
    -march=native)

target_include_directories(
    app 
    PRIVATE 
    ./vendor/glad/include
    ./vendor/glfw/include/ 
    ./vendor/glm/)

target_link_libraries(
    app 
    PRIVATE 
    glfw 
    glm 
    OpenGL::GL)
