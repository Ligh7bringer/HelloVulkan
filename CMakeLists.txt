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

add_custom_target(
  prepare_shaders
  # Copy shaders to build directory
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/shaders
          ${CMAKE_BINARY_DIR}/shaders
  # Compile shaders
  COMMAND glslc ${CMAKE_BINARY_DIR}/shaders/triangle.vert -o
          ${CMAKE_BINARY_DIR}/shaders/vert.spv
  COMMAND glslc ${CMAKE_BINARY_DIR}/shaders/triangle.frag -o
          ${CMAKE_BINARY_DIR}/shaders/frag.spv
  COMMENT "Compiling shaders..."
  USES_TERMINAL VERBATIM)

add_dependencies(hello_triangle prepare_shaders)
