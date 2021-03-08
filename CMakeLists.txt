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

add_library(tiny_obj_loader STATIC
            ${LIB_DIR}/tiny_obj_loader/tiny_obj_loader.cc)
target_include_directories(tiny_obj_loader INTERFACE ${LIB_DIR}/tiny_obj_loader)

add_executable(hello_triangle src/main.cpp)
target_link_libraries(hello_triangle PRIVATE glfw ${Vulkan_LIBRARIES}
                                             tiny_obj_loader)

target_include_directories(
  hello_triangle PRIVATE ${LIB_DIR}/glm ${LIB_DIR}/GLFW/include
                         ${LIB_DIR}/stb_image ${Vulkan_INCLUDE_DIRS})
target_compile_definitions(hello_triangle PRIVATE GLFW_INCLUDE_VULKAN)
target_compile_features(hello_triangle PRIVATE cxx_std_17)

set(SHADER_DIR assets/shaders)
add_custom_target(
  prepare_shaders
  # Copy shaders to build directory
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/assets
          ${CMAKE_BINARY_DIR}/assets
  # Compile shaders
  COMMAND glslc ${CMAKE_BINARY_DIR}/${SHADER_DIR}/triangle.vert -o
          ${CMAKE_BINARY_DIR}/${SHADER_DIR}/vert.spv
  COMMAND glslc ${CMAKE_BINARY_DIR}/${SHADER_DIR}/triangle.frag -o
          ${CMAKE_BINARY_DIR}/${SHADER_DIR}/frag.spv
  COMMENT "Compiling shaders..."
  USES_TERMINAL VERBATIM)

add_dependencies(hello_triangle prepare_shaders)
