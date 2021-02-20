#pragma once

#include <fstream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <GLFW/glfw3.h>

#define VK_SAFE(FUNC)                                         \
	do                                                        \
	{                                                         \
		auto err = (FUNC);                                    \
		if (err != VK_SUCCESS)                                \
		{                                                     \
			std::string err_msg("@@@ Vulkan error in file "); \
			err_msg += __FILE__;                              \
			err_msg += " on line ";                           \
			err_msg += std::to_string(__LINE__);              \
			throw std::runtime_error(err_msg);                \
		}                                                     \
	} while (0)

inline VkResult CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                             const VkAllocationCallbacks *             pAllocator,
                                             VkDebugUtilsMessengerEXT *pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
	    instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}

	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

inline void DestroyDebugUtilsMessengerEXT(VkInstance                   instance,
                                          VkDebugUtilsMessengerEXT     debugMessenger,
                                          const VkAllocationCallbacks *pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
	    instance, "vkDestroyDebugUtilsMessengerEXT");

	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() const
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR        capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR>   presentModes;
};

inline std::vector<char> readFile(const std::string &filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file!");
	}

	std::size_t       fileSize = (std::size_t) file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
}

template <typename DataTy>
inline size_t sizeOfVectorBytes(const std::vector<DataTy> &array)
{
	return (sizeof(array[0]) * array.size());
}
