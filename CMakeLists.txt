cmake_minimum_required(VERSION 3.8)

project(VulkanTriangle)

# Find Vulkan
find_package(Vulkan REQUIRED)

set(LIB_DIR ${PROJECT_SOURCE_DIR}/vendor)
# Set up glfw
option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory(${LIB_DIR}/GLFW)

add_executable(hello_triangle src/main.cpp)
target_link_libraries(hello_triangle PRIVATE glfw ${Vulkan_LIBRARIES})

target_include_directories(
  hello_triangle PRIVATE ${LIB_DIR}/glm ${LIB_DIR}/GLFW/include
                         ${Vulkan_INCLUDE_DIRS})
target_compile_definitions(hello_triangle PRIVATE GLFW_INCLUDE_VULKAN)
target_compile_features(hello_triangle PRIVATE cxx_std_17)
