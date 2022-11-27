#include <vulkan/vulkan_raii.hpp>

#include <Orochi/Orochi.h>
#include <Test/Common.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN64
#define NOMINMAX
#include <windows.h>

#include <vulkan/vulkan_win32.h>

#include <VersionHelpers.h>
#include <aclapi.h>
#include <dxgi1_2.h>

class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES *operator&();
	~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
		1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
	if (!m_winPSecurityDescriptor) {
		throw std::runtime_error(
			"Failed to allocate memory for security descriptor");
	}

	PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
		SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor,
		SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
		SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
		0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions =
		STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
	return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
	PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
		SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}
#endif

vk::ExternalMemoryHandleTypeFlagBits vkExternalMemoryHandleType() {
#ifdef _WIN64
	return IsWindows8Point1OrGreater()
		? vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
		: vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt;
#else
	return vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif
}

vk::raii::DeviceMemory
vkMalloc(vk::raii::Device const &device,
	vk::PhysicalDeviceMemoryProperties const &memoryProperties,
	vk::MemoryRequirements const &memoryRequirements,
	vk::MemoryPropertyFlags memoryPropertyFlags) {
	uint32_t memoryTypeBits = memoryRequirements.memoryTypeBits;
	uint32_t memoryTypeIndex = uint32_t(~0);
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & 1) &&
			((memoryProperties.memoryTypes[i].propertyFlags &
				memoryPropertyFlags) == memoryPropertyFlags)) {
			memoryTypeIndex = i;
			break;
		}
		memoryTypeBits >>= 1;
	}
	assert(memoryTypeIndex != uint32_t(~0));
	vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size,
		memoryTypeIndex);
	return vk::raii::DeviceMemory(device, memoryAllocateInfo);
}

vk::raii::DeviceMemory
vkExternalMalloc(vk::raii::Device const &device,
	vk::PhysicalDeviceMemoryProperties const &memoryProperties,
	vk::MemoryRequirements const &memoryRequirements,
	vk::MemoryPropertyFlags memoryPropertyFlags) {
	uint32_t memoryTypeBits = memoryRequirements.memoryTypeBits;
	uint32_t memoryTypeIndex = uint32_t(~0);
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & 1) &&
			((memoryProperties.memoryTypes[i].propertyFlags &
				memoryPropertyFlags) == memoryPropertyFlags)) {
			memoryTypeIndex = i;
			break;
		}
		memoryTypeBits >>= 1;
	}
	assert(memoryTypeIndex != uint32_t(~0));
	vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size,
		memoryTypeIndex);
#ifdef _WIN64
	WindowsSecurityAttributes winSecurityAttributes;
	VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
	vulkanExportMemoryWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
	vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
	vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
		DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
	VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
	vulkanExportMemoryAllocateInfoKHR.sType =
		VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
	vulkanExportMemoryAllocateInfoKHR.pNext =
		static_cast<VkExternalMemoryHandleTypeFlags>(
			vkExternalMemoryHandleType()) &
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
		? &vulkanExportMemoryWin32HandleInfoKHR
		: NULL;
	vulkanExportMemoryAllocateInfoKHR.handleTypes =
		static_cast<VkExternalMemoryHandleTypeFlags>(
			vkExternalMemoryHandleType());
#else
	vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
	vulkanExportMemoryAllocateInfoKHR.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
	memoryAllocateInfo.setPNext(&vulkanExportMemoryAllocateInfoKHR);
	return vk::raii::DeviceMemory(device, memoryAllocateInfo);
}

std::vector<uint32_t> vkReadSPV(std::string const &filename) {
	std::vector<uint32_t> data;
	std::ifstream file;
	file.open(filename, std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	file.seekg(0, std::ios::end);
	uint64_t read_count = static_cast<uint64_t>(file.tellg());
	file.seekg(0, std::ios::beg);
	data.resize(static_cast<size_t>(read_count / sizeof(uint32_t)));
	file.read(reinterpret_cast<char *>(data.data()), read_count);
	file.close();
	return data;
}

void *vkGetMemHandle(vk::raii::Device const &device, VkDevice m_device,
	VkDeviceMemory memory,
	VkExternalMemoryHandleTypeFlagBits handleType) {
#ifdef _WIN64
	HANDLE handle = 0;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
	vkMemoryGetWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory = memory;
	vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

	PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
	fpGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)device.getProcAddr(
		"vkGetMemoryWin32HandleKHR");
	if (!fpGetMemoryWin32HandleKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR,
		&handle) != VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}
	return (void *)handle;
#else
	int fd = -1;

	VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
	vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	vkMemoryGetFdInfoKHR.pNext = NULL;
	vkMemoryGetFdInfoKHR.memory = memory;
	vkMemoryGetFdInfoKHR.handleType = handleType;

	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
	fpGetMemoryFdKHR =
		(PFN_vkGetMemoryFdKHR)device.getProcAddr("vkGetMemoryFdKHR");
	if (!fpGetMemoryFdKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetMemoryFdKHR(m_device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}
	return (void *)(uintptr_t)fd;
#endif
}

void importPpExternalMemory( void** ppPtr, oroExternalMemory ppMem,
	vk::raii::Device const &device, VkDevice m_device,
	VkDeviceMemory vkMem, VkDeviceSize size,
	VkExternalMemoryHandleTypeFlagBits handleType) {
	oroExternalMemoryHandleDesc externalMemoryHandleDesc = {};

	if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		externalMemoryHandleDesc.type = oroExternalMemoryHandleTypeOpaqueWin32;
	} else if (handleType &
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		externalMemoryHandleDesc.type = oroExternalMemoryHandleTypeOpaqueWin32Kmt;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		externalMemoryHandleDesc.type = oroExternalMemoryHandleTypeOpaqueFd;
	} else {
		throw std::runtime_error("Unknown handle type requested!");
	}

	externalMemoryHandleDesc.size = size;

#ifdef _WIN64
	externalMemoryHandleDesc.handle.win32.handle =
		(HANDLE)vkGetMemHandle(device, m_device, vkMem, handleType);
#else
	externalMemoryHandleDesc.handle.fd =
		(int)(uintptr_t)vkGetMemHandle(device, m_device, vkMem, handleType);
#endif

	oroImportExternalMemory(&ppMem, &externalMemoryHandleDesc);

	oroExternalMemoryBufferDesc externalMemBufferDesc = {};
	externalMemBufferDesc.offset = 0;
	externalMemBufferDesc.size = size;
	externalMemBufferDesc.flags = 0;

	oroExternalMemoryGetMappedBuffer((oroDeviceptr*)ppPtr, ppMem, &externalMemBufferDesc);
}

static std::string AppName = "App";
static std::string EngineName = "Engine";

int main(int argc, char **argv) {
	int32_t localSize = 8;
	size_t memorySize = localSize * localSize * sizeof(float);
	oroApi api = getApiType( argc, argv );
	int a = oroInitialize(api, 0);
	oroError e;
	e = oroInit(0);
	oroDevice device;
	e = oroDeviceGet(&device, 0);
	oroCtx ctx;
	e = oroCtxCreate(&ctx, 0, device);
	oroDeviceProp props;
	e = oroGetDeviceProperties(&props, device);
	try {
		vk::raii::Context context;
		vk::ApplicationInfo applicationInfo(AppName.c_str(), 1, EngineName.c_str(),
			1, VK_API_VERSION_1_1);
		vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo);
		vk::raii::Instance instance(context, instanceCreateInfo);
		uint32_t physicalDeviceIndex = uint32_t(~0);
		for (uint32_t i = 0; i < vk::raii::PhysicalDevices(instance).size(); ++i) {
			auto physicalDeviceProperties =
				vk::raii::PhysicalDevices(instance)[i]
				.getProperties2<vk::PhysicalDeviceProperties2,
				vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
			vk::PhysicalDevicePCIBusInfoPropertiesEXT
				physicalDevicePCIBusInfoProperties =
				physicalDeviceProperties
				.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
			if (physicalDevicePCIBusInfoProperties.pciDomain == props.pciDomainID &&
				physicalDevicePCIBusInfoProperties.pciBus == props.pciBusID &&
				physicalDevicePCIBusInfoProperties.pciDevice == props.pciDeviceID) {
				physicalDeviceIndex = i;
				break;
			}
		}
		if (physicalDeviceIndex == uint32_t(~0)) {
			throw std::runtime_error("physicalDeviceIndex == uint32_t(~0)");
		}
		vk::raii::PhysicalDevice physicalDevice =
			std::move(vk::raii::PhysicalDevices(instance)[physicalDeviceIndex]);
		auto physicalDeviceProperties =
			physicalDevice
			.getProperties2<vk::PhysicalDeviceProperties2,
			vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
		vk::PhysicalDevicePCIBusInfoPropertiesEXT
			physicalDevicePCIBusInfoProperties =
			physicalDeviceProperties
			.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
		std::cout << "Physical Device Index: " << physicalDeviceIndex << "\n";
		std::cout << "PCI Domain: " << physicalDevicePCIBusInfoProperties.pciDomain
			<< "\n";
		std::cout << "PCI Bus: " << physicalDevicePCIBusInfoProperties.pciBus
			<< "\n";
		std::cout << "PCI Device: " << physicalDevicePCIBusInfoProperties.pciDevice
			<< "\n";
		std::vector<const char *> extensions;
		extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
		extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
		extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
		extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
		extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
		float queuePriority = 0.0f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, 0, 1, &queuePriority);
		vk::DeviceCreateInfo deviceCreateInfo({}, deviceQueueCreateInfo, {},
			extensions);
		vk::raii::Device device(physicalDevice, deviceCreateInfo);
		vk::CommandPoolCreateInfo commandPoolCreateInfo({}, 0);
		vk::raii::CommandPool commandPool(device, commandPoolCreateInfo);
		vk::CommandBufferAllocateInfo commandBufferAllocateInfo(
			*commandPool, vk::CommandBufferLevel::ePrimary, 1);
		vk::raii::CommandBuffer commandBuffer = std::move(
			vk::raii::CommandBuffers(device, commandBufferAllocateInfo).front());
		vk::raii::Queue queue(device, 0, 0);
		vk::BufferCreateInfo bufferCreateInfo(
			{}, memorySize, vk::BufferUsageFlagBits::eStorageBuffer,
			vk::SharingMode::eExclusive);
		vk::ExternalMemoryBufferCreateInfo externalMemoryBufferCreateInfo(
			vkExternalMemoryHandleType());
		bufferCreateInfo.setPNext(&externalMemoryBufferCreateInfo);
		vk::raii::Buffer deviceBuffer(device, bufferCreateInfo);
		vk::raii::DeviceMemory deviceMemory =
			vkExternalMalloc(device, physicalDevice.getMemoryProperties(),
				deviceBuffer.getMemoryRequirements(),
				vk::MemoryPropertyFlagBits::eDeviceLocal);
		deviceBuffer.bindMemory(*deviceMemory, 0);
		vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding(
			0, vk::DescriptorType::eStorageBuffer, 1,
			vk::ShaderStageFlagBits::eCompute);
		vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
			{}, descriptorSetLayoutBinding);
		vk::raii::DescriptorSetLayout descriptorSetLayout(
			device, descriptorSetLayoutCreateInfo);
		vk::DescriptorPoolSize poolSize(vk::DescriptorType::eStorageBuffer, 1);
		vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSize);
		vk::raii::DescriptorPool descriptorPool(device, descriptorPoolCreateInfo);
		vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
			*descriptorPool, *descriptorSetLayout);
		vk::raii::DescriptorSet descriptorSet = std::move(
			vk::raii::DescriptorSets(device, descriptorSetAllocateInfo).front());
		vk::DescriptorBufferInfo descriptorBufferInfo(*deviceBuffer, 0, memorySize);
		vk::WriteDescriptorSet writeDescriptorSet(
			*descriptorSet, 0, 0, vk::DescriptorType::eStorageBuffer, {},
			descriptorBufferInfo);
		device.updateDescriptorSets(writeDescriptorSet, nullptr);
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({},
			*descriptorSetLayout);
		vk::raii::PipelineLayout pipelineLayout(device, pipelineLayoutCreateInfo);
		std::vector<uint32_t> computeShaderSPV = vkReadSPV("../Test/VulkanComputeSimple/main.comp.spv");
		vk::ShaderModuleCreateInfo computeShaderModuleCreateInfo({},
			computeShaderSPV);
		vk::raii::ShaderModule computeShaderModule(device,
			computeShaderModuleCreateInfo);
		vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo =
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
				*computeShaderModule, "main");
		vk::ComputePipelineCreateInfo computePipelineCreateInfo(
			{}, pipelineShaderStageCreateInfo, *pipelineLayout);
		vk::raii::Pipeline pipeline(device, nullptr, computePipelineCreateInfo);
		commandBuffer.begin({});
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
			*pipelineLayout, 0, {*descriptorSet},
			nullptr);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		commandBuffer.dispatch(localSize, 1, 1);
		commandBuffer.end();
		vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
		queue.submit(submitInfo);
		device.waitIdle();
		oroExternalMemory externalMemory;
		void *deviceMemoryPp;
		importPpExternalMemory(&deviceMemoryPp, externalMemory, device, *device,
			*deviceMemory, memorySize,
			static_cast<VkExternalMemoryHandleTypeFlagBits>(
				vkExternalMemoryHandleType()));
		std::vector<float> hostMemory{};
		hostMemory.resize(memorySize / sizeof(float));
		oroMemcpyDtoH((void *)hostMemory.data(), (oroDeviceptr)deviceMemoryPp, memorySize);
		bool pass = true;
		for (uint32_t i = 0; i < localSize * localSize; ++i) 
		{
			//      std::cout << i << ": " << hostMemory[i] << "\n";
			if( i!=hostMemory[i] )
				pass = false;
		}
		oroDestroyExternalMemory(externalMemory);
		if( pass )
			std::cout << "PASS\n";
		else
		{
			std::cout << "FAIL\n";
			for (uint32_t i = 0; i < localSize * localSize; ++i) 
			{
				std::cout << i << ": " << hostMemory[i] << "\n";
			}
		}
		std::cout << physicalDevice.getProperties().deviceName << "\n";
	} catch (vk::SystemError &err) {
		std::cout << "vk::SystemError: " << err.what() << std::endl;
		exit(-1);
	} catch (std::exception &err) {
		std::cout << "std::exception: " << err.what() << std::endl;
		exit(-1);
	} catch (...) {
		std::cout << "unknown error\n";
		exit(-1);
	}

	return 0;
}
