//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include <Orochi/OrochiUtils.h>
#include <utility>

namespace Oro
{

/// @brief A helper function that casts an address of a pointer to the device memory to a void pointer to be used as an argument for kernel calls.
/// @tparam T The type of the element stored in the device memory.
/// @param ptr The address of a pointer to the device memory.
/// @return A void pointer.
template<typename T>
void* arg_cast( T* const* ptr ) noexcept
{
	return reinterpret_cast<void*>( const_cast<T**>( ptr ) );
}

template<typename T>
class GpuMemory final
{
  public:
	GpuMemory() = default;

	/// @brief Allocate the elements on the device memory.
	/// @param init_size The initial container size which represents the number of elements.
	explicit GpuMemory( const size_t init_size )
	{
		OrochiUtils::malloc( m_data, init_size );

		m_size = init_size;
		m_capacity = init_size;
	}

	GpuMemory( const GpuMemory& ) = delete;
	GpuMemory& operator=( const GpuMemory& other ) = delete;

	GpuMemory( GpuMemory&& other ) noexcept : m_data{ std::exchange( other.m_data, nullptr ) }, m_size{ std::exchange( other.m_size, 0ULL ) }, m_capacity{ std::exchange( other.m_capacity, 0ULL ) } {}

	GpuMemory& operator=( GpuMemory&& other ) noexcept
	{
		GpuMemory tmp( std::move( other ) );

		swap( *this, tmp );

		return *this;
	}

	~GpuMemory()
	{
		if( m_data )
		{
			OrochiUtils::free( m_data );
			m_data = nullptr;
		}
		m_size = 0ULL;
		m_capacity = 0ULL;
	}

	/// @brief  Get the container size which represents the number of elements.
	/// @return The container size which represents the number of elements.
	size_t size() const noexcept { return m_size; }

	/// @brief Get the pointer to the device memory.
	/// @return The pointer to the device memory.
	T* ptr() const noexcept { return m_data; }

	/// @brief Get the address of the pointer to the device memory. Useful for passing arguments to the kernel call.
	/// @return The address of the pointer to the device memory.
	T* const* address() const noexcept { return &m_data; }

	/// @brief Resize the container. Its capacity is unchanged if the new size is smaller than the current one.
	/// The old data should be considered invalid to be used after the function is called unless @c copy is set to True.
	/// @param new_size The new container size which represents the number of elements after the function is called.
	/// @param copy If true, the function will copy the data to the newly created memory space as well.
	void resize( const size_t new_size, const bool copy = false ) noexcept
	{
		if( new_size <= m_capacity )
		{
			m_size = new_size;
			return;
		}

		GpuMemory tmp( new_size );

		if( copy )
		{
			OrochiUtils::copyDtoD( tmp.m_data, m_data, m_size );
		}

		*this = std::move( tmp );
	}

	/// @brief Asynchronous version of @c resize using a given Orochi stream.
	/// @param new_size The new container size which represents the number of elements after the function is called.
	/// @param copy If true, the function will copy the data to the newly created memory space as well.
	/// @param stream The Orochi stream used for the underlying operations.
	void resizeAsync( const size_t new_size, const bool copy = false, oroStream stream = 0 ) noexcept
	{
		if( new_size <= m_capacity )
		{
			m_size = new_size;
			return;
		}

		GpuMemory tmp( new_size );

		if( copy )
		{
			OrochiUtils::copyDtoDAsync( tmp.m_data, m_data, m_size, stream );
		}

		*this = std::move( tmp );
	}

	/// @brief Reset the memory space so that all bits inside are cleared to zero.
	void reset() noexcept { OrochiUtils::memset( m_data, 0, m_size * sizeof( T ) ); }

	/// @brief Asynchronous version of @c reset using a given Orochi stream.
	/// @param stream The Orochi stream used for the underlying operations.
	void resetAsync( oroStream stream = 0 ) noexcept { OrochiUtils::memsetAsync( m_data, 0, m_size * sizeof( T ), stream ); }

	/// @brief Copy the data from device memory to host.
	/// @param host_ptr The host pointer.
	/// @param host_data_size The size of the host memory which represents the number of elements.
	void copyFromHost( const T* host_ptr, const size_t host_data_size ) noexcept
	{
		resize( host_data_size );
		OrochiUtils::copyHtoD( m_data, host_ptr, host_data_size );
	}

	/// @brief Get the content of the first element stored in the device memory.
	/// @return The content of the first element in the device memory.
	T getSingle() const noexcept
	{
		T result{};

		OrochiUtils::copyDtoH( &result, m_data, 1ULL );

		return result;
	}

	/// @brief Get all the data stored in the device memory.
	/// @return A vector which contains all the data stored in the device memory.
	std::vector<T> getData() const noexcept
	{
		std::vector<T> result{};
		result.resize( m_size );

		OrochiUtils::copyDtoH( result.data(), m_data, m_size );

		return result;
	}

  private:
	static void swap( GpuMemory& lhs, GpuMemory& rhs ) noexcept
	{
		std::swap( lhs.m_data, rhs.m_data );
		std::swap( lhs.m_size, rhs.m_size );
		std::swap( lhs.m_capacity, rhs.m_capacity );
	}

	T* m_data{ nullptr };
	size_t m_size{ 0ULL };
	size_t m_capacity{ 0ULL };
};

} // namespace Oro
