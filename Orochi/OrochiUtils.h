#pragma once
#include <Orochi/Orochi.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined( GNUC )
#include <signal.h>
#endif

template<typename T, typename U>
constexpr void OROASSERT( T&& exp, [[maybe_unused]] U&& placeholder ) noexcept
{
	if( static_cast<bool>( std::forward<T>( exp ) ) != true )
	{

#if defined( _WIN32 )
		__debugbreak();
#elif defined( GNUC )
		raise( SIGTRAP );
#else
		;
#endif
	}
}

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	OrochiUtils() = default;
	//make class non copyable and movable
	OrochiUtils(const OrochiUtils&) = delete; 
    OrochiUtils& operator=(const OrochiUtils&) = delete;
    OrochiUtils(OrochiUtils&&) = delete; 
    OrochiUtils& operator=(OrochiUtils&&) = delete;
	~OrochiUtils() = default;

	oroFunction getFunctionFromPrecompiledBinary( const std::string& path, const std::string& funcName );

	oroFunction getFunctionFromFile( oroDevice device, const char* path, const char* funcName, std::vector<const char*>* opts );
	oroFunction getFunctionFromString( oroDevice device, const char* source, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders, const char** headers, const char** includeNames );
	oroFunction getFunction( oroDevice device, const char* code, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders = 0, const char** headers = 0, const char** includeNames = 0 );

	static bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	static void getData( oroDevice device, const char* code, const char* path, std::vector<const char*>* opts, std::vector<char>& dst );
	static void getProgram( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, orortcProgram* prog );
	static void getModule( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, oroModule* moduleOut );
	static void launch1D( oroFunction func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0, oroStream stream = 0 );
	static void launch2D( oroFunction func, int nx, int ny, const void** args, int wgSizeX = 8, int wgSizeY = 8, unsigned int sharedMemBytes = 0, oroStream stream = 0 );

	template<typename T>
	static void malloc( T*& ptr, size_t n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void mallocManaged( T*& ptr, size_t n, oroManagedMemoryAttachFlags flags )
	{
#if defined( _WIN32 )
#else
		oroError e = oroMallocManaged( (oroDeviceptr*)&ptr, sizeof( T ) * n, flags );
		OROASSERT( e == oroSuccess, 0 );
#endif
	}

	template<typename T>
	static void free( T* ptr )
	{
		oroFree( (oroDeviceptr)ptr );
	}

	static void memset( void* ptr, int val, size_t n )
	{
		oroError e = oroMemset( (oroDeviceptr)ptr, val, n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void memsetAsync( void* ptr, int val, size_t n, oroStream stream )
	{
		oroError e = oroMemsetD8Async( (oroDeviceptr)ptr, val, n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoD( T* dst, const T* src, size_t n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoH( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoD( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyHtoDAsync( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoHAsync( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoDAsync( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void waitForCompletion( oroStream stream = 0 )
	{
		auto e = oroStreamSynchronize( stream );
		OROASSERT( e == oroSuccess, 0 );
	}

  public:
	std::string m_cacheDirectory = "./cache/";
	std::recursive_mutex m_mutex;
	std::unordered_map<std::string, oroFunction> m_kernelMap;
};

class OroStopwatch
{
  public:
	OroStopwatch( oroStream stream ) 
	{ 
		m_stream = stream;
		oroEventCreateWithFlags( &m_start, oroEventDefault );
		oroEventCreateWithFlags( &m_stop, oroEventDefault );
	}
	~OroStopwatch() 
	{
		oroEventDestroy( m_start );
		oroEventDestroy( m_stop );
	}

	void start() { oroEventRecord( m_start, m_stream ); }
	void stop() { oroEventRecord( m_stop, m_stream ); }

	float getMs() 
	{ 
		oroEventSynchronize( m_stop );
		float ms = 0;
		oroEventElapsedTime( &ms, m_start, m_stop );
		return ms;
	}

  public:
	oroStream m_stream;
	oroEvent m_start;
	oroEvent m_stop;
};

class Timer final
{
  public:
	static constexpr bool EnableTimer = false;

	using TokenType = int;
	using TimeUnit	= float;

	Timer() = default;

	Timer( const Timer& ) = default;
	Timer( Timer&& )	  = default;

	Timer& operator=( const Timer& )  = default;
	Timer& operator=( Timer&& other ) = default;

	~Timer() = default;

	class Profiler;

	/// Call the callable and measure the elapsed time using Orochi events.
	/// @param[in] token The token of the time record.
	/// @param[in] callable The callable object to be called.
	/// @param[in] args The parameters of the callable.
	/// @return The forwarded returned result of the callable.
	template <typename CallableType, typename... Args>
	decltype( auto ) measure( const TokenType token, CallableType&& callable, Args&&... args ) noexcept
	{

#define OROCHECK( x ) { oroError e = x; OROASSERT( e == ORO_SUCCESS ); }

		TimeUnit time{};
		oroEvent start{};
		oroEvent stop{};
		if constexpr ( EnableTimer )
		{
			OROCHECK( oroEventCreateWithFlags( &start, 0 ) );
			OROCHECK( oroEventCreateWithFlags( &stop, 0 ) );
			OROCHECK( oroEventRecord( start, 0 ) );
		}

		using return_type = std::invoke_result_t<CallableType, Args...>;
		if constexpr ( std::is_void_v<return_type> )
		{
			std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... );
			if constexpr ( EnableTimer )
			{
				OROCHECK( oroEventRecord( stop, 0 ) );
				OROCHECK( oroEventSynchronize( stop ) );
				OROCHECK( oroEventElapsedTime( &time, start, stop ) );
				OROCHECK( oroEventDestroy( start ) );
				OROCHECK( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return;
		}
		else
		{
			decltype( auto ) result{ std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... ) };
			if constexpr ( EnableTimer )
			{
				OROCHECK( oroEventRecord( stop, 0 ) );
				OROCHECK( oroEventSynchronize( stop ) );
				OROCHECK( oroEventElapsedTime( &time, start, stop ) );
				OROCHECK( oroEventDestroy( start ) );
				OROCHECK( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return result;
		}
#undef OROCHECK
	}

	[[nodiscard]] TimeUnit getTimeRecord( const TokenType token ) const noexcept
	{
		if ( timeRecord.find( token ) != timeRecord.end() ) return timeRecord.at( token );
		return TimeUnit{};
	}

	void reset( const TokenType token ) noexcept
	{
		if ( timeRecord.count( token ) > 0UL )
		{
			timeRecord[token] = TimeUnit{};
		}
	}

	void clear() noexcept { timeRecord.clear(); }

  private:
	using TimeRecord = std::unordered_map<TokenType, TimeUnit>;
	TimeRecord timeRecord;
};