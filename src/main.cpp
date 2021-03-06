#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>        // UINT32_MAX
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "UniformBuffer.hpp"
#include "Util.hpp"
#include "Vertex.hpp"

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const uint32_t WIDTH                 = 800;
const uint32_t HEIGHT                = 600;
const int      MAX_CONCURRENT_FRAMES = 2;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

static const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                             {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                             {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                             {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

static const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

class HelloTriangleApplication
{
	GLFWwindow *                 window_;
	VkInstance                   instance_;
	VkDebugUtilsMessengerEXT     debugMessenger_;
	VkSurfaceKHR                 surface_;
	VkPhysicalDevice             physicalDevice_ = VK_NULL_HANDLE;
	VkDevice                     device_;
	VkQueue                      graphicsQueue_, presentQueue_;
	VkSwapchainKHR               swapChain_;
	std::vector<VkImage>         swapChainImages_;
	VkFormat                     swapChainImageFormat_;
	VkExtent2D                   swapChainExtent_;
	std::vector<VkImageView>     swapChainImageViews_;
	VkRenderPass                 renderPass_;
	VkDescriptorSetLayout        descriptorSetLayout_;
	VkPipelineLayout             pipelineLayout_;
	VkPipeline                   graphicsPipeline_;
	std::vector<VkFramebuffer>   swapChainFramebuffers_;
	VkCommandPool                commandPool_;
	std::vector<VkCommandBuffer> commandBuffers_;
	std::vector<VkSemaphore>     imageAvailableSemaphores_, renderFinishedSemaphores_;
	std::vector<VkFence>         concurrentFences_;
	std::vector<VkFence>         concurrentImages_;
	size_t                       currentFrame_{0};
	bool                         framebufferResized_{false};
	VkBuffer                     vertexBuffer_, indexBuffer_;
	VkDeviceMemory               vertexBufferMemory_, indexBufferMemory_;
	std::vector<VkBuffer>        uniformBuffers_;
	std::vector<VkDeviceMemory>  uniformBuffersMemory_;
	VkDescriptorPool             descriptorPool_;
	std::vector<VkDescriptorSet> descriptorSets_;
	VkImage                      textureImage_;
	VkDeviceMemory               textureImageMemory_;

  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	void initWindow()
	{
		glfwInit();

		// Tell GLFW not to create an OpenGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		// Disable window resizing
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window_ = glfwCreateWindow(WIDTH, HEIGHT, "Hello, Vulkan!", nullptr, nullptr);
		glfwSetWindowUserPointer(window_, this);
		glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
		app->framebufferResized_ = true;
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createTextureImage();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncPrimitives();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window_))
		{
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device_);
	}

	void createTextureImage()
	{
		int      texWidth, texHeight, texChannels;
		stbi_uc *pixels = stbi_load(TEXTURE_DIR "texture.jpg", &texWidth, &texHeight, &texChannels,
		                            STBI_rgb_alpha);
		const VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image");
		}

		VkBuffer       stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		allocateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		               stagingBuffer, stagingBufferMemory);

		void *data;
		VK_SAFE(vkMapMemory(device_, stagingBufferMemory, 0, imageSize, 0, &data));
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device_, stagingBufferMemory);

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
		            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage_, textureImageMemory_);

		transitionImageLayout(textureImage_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage_, static_cast<uint32_t>(texWidth),
		                  static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage_, VK_FORMAT_R8G8B8A8_SRGB,
		                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device_, stagingBuffer, nullptr);
		vkFreeMemory(device_, stagingBufferMemory, nullptr);
	}

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
	                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
	                 VkDeviceMemory &imageMemory)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType     = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width  = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth  = 1;
		imageInfo.mipLevels     = 1;
		imageInfo.arrayLayers   = 1;
		imageInfo.format        = format;
		imageInfo.tiling        = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage         = usage;
		imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags         = 0;        // Optional

		VK_SAFE(vkCreateImage(device_, &imageInfo, nullptr, &image));

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device_, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize  = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		VK_SAFE(vkAllocateMemory(device_, &allocInfo, nullptr, &imageMemory));
		vkBindImageMemory(device_, image, imageMemory, /*offset*/ 0);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
	                           VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout           = oldLayout;
		barrier.newLayout           = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;

		barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
		    newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = 0;

			// Transfer don't need to wait on anything, so use earliest possible stage of the
			// pipeline
			sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
		         newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition");
		}

		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0,
		                     nullptr, 1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset      = 0;
		region.bufferRowLength   = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel       = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount     = 1;

		region.imageOffset = {0, 0, 0};
		region.imageExtent = {width, height, 1};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                       /*regionCount*/ 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool        = commandPool_;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		VK_SAFE(vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer));

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_SAFE(vkBeginCommandBuffer(commandBuffer, &beginInfo));

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers    = &commandBuffer;

		VK_SAFE(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE));
		VK_SAFE(vkQueueWaitIdle(graphicsQueue_));

		vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
	}

	void createDescriptorSets()
	{
		const size_t numImages = swapChainImages_.size();

		std::vector<VkDescriptorSetLayout> layouts(numImages, descriptorSetLayout_);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool     = descriptorPool_;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(numImages);
		allocInfo.pSetLayouts        = layouts.data();

		descriptorSets_.resize(numImages);
		VK_SAFE(vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data()));

		for (size_t i = 0; i < numImages; ++i)
		{
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers_[i];
			bufferInfo.offset = 0;
			bufferInfo.range  = VK_WHOLE_SIZE;        // sizeof(UniformBufferObject);

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet           = descriptorSets_[i];
			descriptorWrite.dstBinding       = 0;
			descriptorWrite.dstArrayElement  = 0;
			descriptorWrite.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount  = 1;
			descriptorWrite.pBufferInfo      = &bufferInfo;
			descriptorWrite.pImageInfo       = nullptr;        // Optional
			descriptorWrite.pTexelBufferView = nullptr;        // Optional

			vkUpdateDescriptorSets(device_, 1, &descriptorWrite, 0, nullptr);
		}
	}

	void createDescriptorPool()
	{
		VkDescriptorPoolSize poolSize{};
		poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages_.size());

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes    = &poolSize;
		poolInfo.maxSets       = static_cast<uint32_t>(swapChainImages_.size());

		VK_SAFE(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
	}

	void createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding            = 0;        // location
		uboLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount    = 1;
		uboLayoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings    = &uboLayoutBinding;

		VK_SAFE(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));
	}

	void allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags,
	                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
	                    VkDeviceMemory &bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size        = size;
		bufferInfo.usage       = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		// bufferInfo.flags = 0;

		VK_SAFE(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer));

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize  = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		VK_SAFE(vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory));
		VK_SAFE(vkBindBufferMemory(device_, buffer, bufferMemory, 0));
	}

	void createVertexBuffer()
	{
		initialiseBuffer(vertices, vertexBuffer_, vertexBufferMemory_,
		                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	void createIndexBuffer()
	{
		initialiseBuffer(indices, indexBuffer_, indexBufferMemory_,
		                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		const auto vecSize = swapChainImages_.size();
		uniformBuffers_.resize(vecSize);
		uniformBuffersMemory_.resize(vecSize);

		for (size_t i = 0; i < vecSize; ++i)
		{
			allocateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			               uniformBuffers_[i], uniformBuffersMemory_[i]);
		}
	}

	template <typename DataTy>
	void initialiseBuffer(const std::vector<DataTy> &data, VkBuffer &buffer,
	                      VkDeviceMemory &bufferMemory, VkBufferUsageFlags usageFlags)
	{
		const VkDeviceSize bufferSize = sizeOfVectorBytes(vertices);

		// Temporary buffer to copy the data to a local memory vertex buffer
		VkBuffer       stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		allocateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		               stagingBuffer, stagingBufferMemory);

		void *mappedData;
		VK_SAFE(vkMapMemory(device_, stagingBufferMemory, 0, bufferSize, 0, &mappedData));
		memcpy(mappedData, data.data(), static_cast<size_t>(bufferSize));
		vkUnmapMemory(device_, stagingBufferMemory);

		allocateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usageFlags,
		               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);
		copyBuffer(stagingBuffer, buffer, bufferSize);

		vkDestroyBuffer(device_, stagingBuffer, nullptr);
		vkFreeMemory(device_, stagingBufferMemory, nullptr);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
		{
			if ((typeFilter & (1 << i)) &&
			    (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		const auto  currentTime = std::chrono::high_resolution_clock::now();
		const float time =
		    std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime)
		        .count();

		UniformBufferObject ubo{};
		ubo.model =
		    glm::rotate(glm::mat4(1.f), time * glm::radians(90.f), glm::vec3(0.f, 0.f, 1.f));
		ubo.view = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f),
		                       glm::vec3(0.f, 0.f, 1.f));
		ubo.proj = glm::perspective(glm::radians(45.f),
		                            static_cast<float>(swapChainExtent_.width) /
		                                static_cast<float>(swapChainExtent_.height),
		                            0.1f, 10.f);
		ubo.proj[1][1] *= -1;        // invert y coordinate of scaling factor

		const auto dataSize = sizeof(ubo);
		void *     mappedData;
		VK_SAFE(
		    vkMapMemory(device_, uniformBuffersMemory_[currentImage], 0, dataSize, 0, &mappedData));
		memcpy(mappedData, &ubo, dataSize);
		vkUnmapMemory(device_, uniformBuffersMemory_[currentImage]);
	}

	void drawFrame()
	{
		vkWaitForFences(device_, 1, &concurrentFences_[currentFrame_], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device_, swapChain_, UINT64_MAX,
		                                        imageAvailableSemaphores_[currentFrame_],
		                                        VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// If a previous frame is using this image
		if (concurrentImages_[imageIndex] != VK_NULL_HANDLE)
		{
			// Wait for it to finish
			vkWaitForFences(device_, 1, &concurrentImages_[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark this image as used by the current frame
		concurrentImages_[imageIndex] = concurrentFences_[currentFrame_];

		VkSemaphore          waitSemaphores[]   = {imageAvailableSemaphores_[currentFrame_]};
		VkSemaphore          signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};
		VkPipelineStageFlags waitStages[]{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

		updateUniformBuffer(imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount   = 1;
		submitInfo.pWaitSemaphores      = waitSemaphores;
		submitInfo.pWaitDstStageMask    = waitStages;
		submitInfo.commandBufferCount   = 1;
		submitInfo.pCommandBuffers      = &commandBuffers_[imageIndex];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores    = signalSemaphores;

		vkResetFences(device_, 1, &concurrentFences_[currentFrame_]);

		VK_SAFE(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, concurrentFences_[currentFrame_]));

		VkSwapchainKHR   swapChains[] = {swapChain_};
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores    = signalSemaphores;
		presentInfo.swapchainCount     = 1;
		presentInfo.pSwapchains        = swapChains;
		presentInfo.pImageIndices      = &imageIndex;

		result = vkQueuePresentKHR(presentQueue_, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
		    framebufferResized_)
		{
			framebufferResized_ = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame_ = (currentFrame_ + 1) % MAX_CONCURRENT_FRAMES;
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		// Handle window minimised
		glfwGetFramebufferSize(window_, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window_, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device_);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
	}

	void cleanupSwapChain()
	{
		for (auto framebuffer : swapChainFramebuffers_)
		{
			vkDestroyFramebuffer(device_, framebuffer, nullptr);
		}

		vkFreeCommandBuffers(device_, commandPool_, static_cast<uint32_t>(commandBuffers_.size()),
		                     commandBuffers_.data());

		vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
		vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
		vkDestroyRenderPass(device_, renderPass_, nullptr);

		for (auto imageView : swapChainImageViews_)
		{
			vkDestroyImageView(device_, imageView, nullptr);
		}

		for (size_t i = 0; i < swapChainImages_.size(); ++i)
		{
			vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
			vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
		}

		vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);

		vkDestroySwapchainKHR(device_, swapChain_, nullptr);
	}

	void cleanup()
	{
		cleanupSwapChain();

		vkDestroyImage(device_, textureImage_, nullptr);
		vkFreeMemory(device_, textureImageMemory_, nullptr);

		vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

		vkDestroyBuffer(device_, indexBuffer_, nullptr);
		vkFreeMemory(device_, indexBufferMemory_, nullptr);

		vkDestroyBuffer(device_, vertexBuffer_, nullptr);
		vkFreeMemory(device_, vertexBufferMemory_, nullptr);

		for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
		{
			vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
			vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
			vkDestroyFence(device_, concurrentFences_[i], nullptr);
		}

		vkDestroyCommandPool(device_, commandPool_, nullptr);

		vkDestroyDevice(device_, nullptr);

		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
		}

		vkDestroySurfaceKHR(instance_, surface_, nullptr);
		vkDestroyInstance(instance_, nullptr);

		glfwDestroyWindow(window_);

		glfwTerminate();
	}

	/*---------------- Vulkan setup functions ----------------*/
	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName   = "Hello, Vulkan!";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName        = "No Engine";
		appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion         = VK_API_VERSION_1_0;

		// Setup instance
		VkInstanceCreateInfo createInfo{};
		createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// Pass the extensions required by GLFW
		auto extensions                    = getRequiredExtensions();
		createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		// Global validation layers to enable
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			// Add a separate messenger for the instance creation
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *) &debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
			createInfo.pNext             = nullptr;
		}

		VK_SAFE(vkCreateInstance(&createInfo, nullptr, &instance_));
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		VK_SAFE(CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger_));
	}

	void createSurface()
	{
		VK_SAFE(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("Failed to find device with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

		auto result =
		    std::find_if(devices.begin(), devices.end(),
		                 [&](VkPhysicalDevice dev) -> bool { return isDeviceSuitable(dev); });
		if (result != std::end(devices))
		{
			physicalDevice_ = *result;
		}

		if (physicalDevice_ == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t>                   uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                                  indices.presentFamily.value()};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount       = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};

		VkDeviceCreateInfo createInfo{};
		createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos       = queueCreateInfos.data();
		createInfo.pEnabledFeatures        = &deviceFeatures;
		createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		VK_SAFE(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));

		// Get handle to graphics queue
		vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
		// Get handle to presentation queue
		vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
	}

	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice_);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapChainSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR   presentMode = chooseSwapChainPresentMode(swapChainSupport.presentModes);
		VkExtent2D         extent      = chooseSwapChainExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		// Make sure we are not requesting too many images
		if (swapChainSupport.capabilities.maxImageCount > 0 &&
		    imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface          = surface_;
		createInfo.minImageCount    = imageCount;
		createInfo.imageFormat      = surfaceFormat.format;
		createInfo.imageColorSpace  = surfaceFormat.colorSpace;
		createInfo.imageExtent      = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices              = findQueueFamilies(physicalDevice_);
		uint32_t           queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                         indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily)
		{
			// Images can be used across multiple queue families without explicit ownership
			// transfers
			createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices   = queueFamilyIndices;
		}
		else
		{
			// An image is owned by one queue family at a time and ownership must be explicitly
			// transfered before using it in another queue family
			createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;              // Optional
			createInfo.pQueueFamilyIndices   = nullptr;        // Optional
		}

		// Should a transform be applied to images in the swapchain
		// Note: Maybe useful when trying to mimic OpenGL behaviour?
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		// Alpha channel of the *window*
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode    = presentMode;
		createInfo.clipped        = VK_TRUE;
		createInfo.oldSwapchain   = VK_NULL_HANDLE;

		VK_SAFE(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_));

		// Get swap chain images
		vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
		swapChainImages_.resize(imageCount);
		vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages_.data());
		std::cout << "Swap chain: Using " << imageCount << " images...\n";

		swapChainImageFormat_ = surfaceFormat.format;
		swapChainExtent_      = extent;
	}

	void createImageViews()
	{
		swapChainImageViews_.resize(swapChainImages_.size());

		for (size_t i = 0; i < swapChainImages_.size(); ++i)
		{
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages_[i];
			// Treat images as 2D textures
			createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format                          = swapChainImageFormat_;
			createInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel   = 0;
			createInfo.subresourceRange.levelCount     = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount     = 1;

			VK_SAFE(vkCreateImageView(device_, &createInfo, nullptr, &swapChainImageViews_[i]));
		}
	}

	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format  = swapChainImageFormat_;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		// What to do with the data before and after rendering
		colorAttachment.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		// We are not doing anything with the stencil buffer - ignore the data
		colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		// Index of the attachment in the attachments array - 0 since we only have 1
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		// This will bind the attachment to the shader. It may be referenced with
		// layout(location = 0) out vec4 outColor
		subpass.pColorAttachments = &colorAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass    = 0;
		dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments    = &colorAttachment;
		renderPassInfo.subpassCount    = 1;
		renderPassInfo.pSubpasses      = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies   = &dependency;

		VK_SAFE(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_));
	}

	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile(SHADER_DIR "vert.spv");
		auto fragShaderCode = readFile(SHADER_DIR "frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// Vertex shader stage
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName  = "main";        // Shader entrypoint - main()

		// Fragment shader stage
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName  = "main";        // Shader entrypoint - main()

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto bindingDescription    = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions    = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount =
		    static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// Setup geometry kind & and primitive restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x        = 0.f;
		viewport.y        = 0.f;
		viewport.width    = static_cast<float>(swapChainExtent_.width);
		viewport.height   = static_cast<float>(swapChainExtent_.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		// We want to draw to the entire framebuffer - use a scissor rectangle with the same size
		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapChainExtent_;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports    = &viewport;
		viewportState.scissorCount  = 1;
		viewportState.pScissors     = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth               = 1.f;
		rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;        // y flipping
		rasterizer.depthBiasEnable         = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.f;        // Optional (used if above is enabled)
		rasterizer.depthBiasClamp          = 0.f;        // As above
		rasterizer.depthBiasSlopeFactor    = 0.f;        // As above

		// Multisampling can be used for anitalising
		// Note: enabling it requires a GPU feature
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable   = VK_FALSE;
		multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading      = 1.f;             // Optional
		multisampling.pSampleMask           = nullptr;         // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE;        // Optional
		multisampling.alphaToOneEnable      = VK_FALSE;        // Optional

		// VkPipelineColorBlendAttachmentState -> per attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable         = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;         // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;        // Optional
		colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;             // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;         // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;        // Optional
		colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;             // Optional

		// VkPipelineColorBlendStateCreateInfo -> global colour blending settings
		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable     = VK_FALSE;
		colorBlending.logicOp           = VK_LOGIC_OP_COPY;        // Optional
		colorBlending.attachmentCount   = 1;
		colorBlending.pAttachments      = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;        // Optional
		colorBlending.blendConstants[1] = 0.0f;        // Optional
		colorBlending.blendConstants[2] = 0.0f;        // Optional
		colorBlending.blendConstants[3] = 0.0f;        // Optional

		// Note: Dynamic state can be specified here so that pipeline re-creation is not necessary
		// to change those values

		// Setup shader uniforms - currently none used
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount         = 1;
		pipelineLayoutInfo.pSetLayouts            = &descriptorSetLayout_;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges    = nullptr;

		VK_SAFE(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount          = 2;
		pipelineInfo.pStages             = shaderStages;
		pipelineInfo.pVertexInputState   = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState      = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState   = &multisampling;
		pipelineInfo.pDepthStencilState  = nullptr;        // Optional - not used
		pipelineInfo.pColorBlendState    = &colorBlending;
		pipelineInfo.pDynamicState       = nullptr;        // Optional - not used
		pipelineInfo.layout              = pipelineLayout_;
		pipelineInfo.renderPass          = renderPass_;
		pipelineInfo.subpass             = 0;                     // Index of the subpass
		pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;        // Optional
		pipelineInfo.basePipelineIndex   = -1;                    // Optional

		VK_SAFE(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
		                                  &graphicsPipeline_));

		vkDestroyShaderModule(device_, fragShaderModule, nullptr);
		vkDestroyShaderModule(device_, vertShaderModule, nullptr);
	}

	void createFramebuffers()
	{
		swapChainFramebuffers_.resize(swapChainImageViews_.size());

		for (size_t i = 0; i < swapChainImageViews_.size(); ++i)
		{
			VkImageView attachments[] = {swapChainImageViews_[i]};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass      = renderPass_;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments    = attachments;
			framebufferInfo.width           = swapChainExtent_.width;
			framebufferInfo.height          = swapChainExtent_.height;
			framebufferInfo.layers          = 1;

			VK_SAFE(vkCreateFramebuffer(device_, &framebufferInfo, nullptr,
			                            &swapChainFramebuffers_[i]));
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		poolInfo.flags            = 0;        // Optional

		VK_SAFE(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
	}

	void createCommandBuffers()
	{
		commandBuffers_.resize(swapChainFramebuffers_.size());

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool        = commandPool_;
		allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t) commandBuffers_.size();

		VK_SAFE(vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()));

		for (size_t i = 0; i < commandBuffers_.size(); ++i)
		{
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags            = 0;
			beginInfo.pInheritanceInfo = nullptr;

			VK_SAFE(vkBeginCommandBuffer(commandBuffers_[i], &beginInfo));

			VkClearValue          clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass        = renderPass_;
			renderPassInfo.framebuffer       = swapChainFramebuffers_[i];
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent_;
			renderPassInfo.clearValueCount   = 1;
			renderPassInfo.pClearValues      = &clearColor;

			vkCmdBeginRenderPass(commandBuffers_[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
			                  graphicsPipeline_);

			VkBuffer     vertexBuffers[] = {vertexBuffer_};
			VkDeviceSize offsets[]       = {0};
			vkCmdBindVertexBuffers(commandBuffers_[i], 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffers_[i], indexBuffer_, 0, VK_INDEX_TYPE_UINT16);

			vkCmdBindDescriptorSets(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
			                        pipelineLayout_, 0, 1, &descriptorSets_[i], 0, nullptr);
			vkCmdDrawIndexed(commandBuffers_[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffers_[i]);
			VK_SAFE(vkEndCommandBuffer(commandBuffers_[i]));
		}
	}

	void createSyncPrimitives()
	{
		imageAvailableSemaphores_.resize(MAX_CONCURRENT_FRAMES);
		renderFinishedSemaphores_.resize(MAX_CONCURRENT_FRAMES);
		concurrentFences_.resize(MAX_CONCURRENT_FRAMES);
		concurrentImages_.resize(swapChainImages_.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i)
		{
			VK_SAFE(
			    vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]));
			VK_SAFE(
			    vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]));
			VK_SAFE(vkCreateFence(device_, &fenceInfo, nullptr, &concurrentFences_[i]));
		}
	}

	/*---------------- Helper functions ----------------*/
	int isDeviceSuitable(VkPhysicalDevice device)
	{
		// Check support for graphics & presentation queues
		QueueFamilyIndices indices = findQueueFamilies(device);

		// Check support for swap chain extension
		const bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		// Note: swap chain can be only queried if the extension is supported
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			// Make sure swap chain suppots at least one image format and present mode
			swapChainAdequate =
			    !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		// Check if device supports required queue families
		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// Look for a queue that supports graphics operations
		int i = 0;
		for (const auto &queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}
			if (indices.isComplete())
			{
				break;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
			if (presentSupport)
			{
				indices.presentFamily = i;
			}

			i++;
		}

		return indices;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char *layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto &layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
		                                     availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto &extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		// Get supported capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

		// Get supported surface formats
		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount,
			                                     details.formats.data());
		}

		// Get supported presentation modes
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount,
			                                          details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR
	    chooseSwapChainSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
	{
		for (const auto &availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
			    availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				std::cout << "Swap chain: Using VK_FORMAT_B8G8R8A8_SRGB format and "
				             "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR colour space...\n";
				return availableFormat;
			}
		}

		std::cout
		    << "Swap chain: Preferred format & colour space not supported, using first result...\n";
		return availableFormats[0];
	}

	VkPresentModeKHR
	    chooseSwapChainPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
	{
		for (const auto &availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				std::cout
				    << "Swap chain: Using VK_PRESENT_MODE_MAILBOX_KHR (i.e. triple buffering)...\n";
				return availablePresentMode;
			}
		}

		std::cout << "Swap chain: Using VK_PRESENT_MODE_FIFO_KHR...\n";
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapChainExtent(const VkSurfaceCapabilitiesKHR &capabilities)
	{
		// Setup resolution
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}

		VkExtent2D actualExtent = {WIDTH, HEIGHT};

		actualExtent.width =
		    std::max(capabilities.minImageExtent.width,
		             std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height =
		    std::max(capabilities.minImageExtent.height,
		             std::min(capabilities.maxImageExtent.height, actualExtent.height));

		std::cout << "Swap chain: Resolution " << actualExtent.height << " x " << actualExtent.width
		          << "\n";
		return actualExtent;
	}

	VkShaderModule createShaderModule(const std::vector<char> &code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode    = reinterpret_cast<const uint32_t *>(code.data());
		// Note: The cast above is fine since std::vector ensures the data satisfies the alignment
		// requirements of uint32_t

		VkShaderModule shaderModule;
		VK_SAFE(vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule));

		return shaderModule;
	}

	std::vector<const char *> getRequiredExtensions()
	{
		uint32_t     glfwExtensionCount = 0;
		const char **glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
	{
		createInfo       = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		/*VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT| */
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL
	    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
	                  VkDebugUtilsMessageTypeFlagsEXT             messageType,
	                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
	{
		std::cerr << "Validation layer error: " << pCallbackData->pMessage << "\n";

		return VK_FALSE;
	}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
