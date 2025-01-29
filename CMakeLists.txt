cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
project(Terrain)

add_subdirectory(./deps/glfw)
add_subdirectory(./deps/glm)

add_executable(main main.cpp)

target_compile_options(main PRIVATE -Wall -Wextra -march=native)
target_include_directories(main PRIVATE ./deps/glfw/include/ ./deps/glm/ ./deps/bgfx/include/)
target_link_libraries(main PRIVATE glfw glm "-framework OpenGL")
