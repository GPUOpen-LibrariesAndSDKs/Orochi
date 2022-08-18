/*
AMD copyrights (Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved) 
*/
#pragma once

#if defined(__WINDOWS__)
	#define NOMINMAX
	#include <windows.h>
	
	#define TIME_TYPE LARGE_INTEGER
	#define QUERY_FREQ(f) QueryPerformanceFrequency(&f)
	#define RECORD(t) QueryPerformanceCounter(&t)
	#define GET_TIME(t) (t).QuadPart*1000.0
	#define GET_FREQ(f)	(f).QuadPart
#else
	#include <sys/time.h> 
	
	#define TIME_TYPE timeval
	#define QUERY_FREQ(f) f.tv_sec = 1
	#define RECORD(t) gettimeofday(&t, 0)
	#define GET_TIME(t) ((t).tv_sec*1000.0+(t).tv_usec/1000.0)
	#define GET_FREQ(f)	1.0
#endif


class Stopwatch
{
	public:
		__inline
		Stopwatch();
		__inline
		void init();
		__inline
		void start();
		__inline
		void split();
		__inline
		float getCurrent();
		__inline
		void stop();
		__inline
		float getMs();
		__inline
		void getMs( float* times, int capacity );

	private:
		enum 
		{
			CAPACITY = 12,
		};
		int m_idx;

		TIME_TYPE m_frequency;
		TIME_TYPE m_t[CAPACITY];
};

__inline
Stopwatch::Stopwatch()
{
//	QueryPerformanceFrequency( &m_frequency );
	QUERY_FREQ( m_frequency );
}

__inline
void Stopwatch::start()
{
	m_idx = 0;
//	QueryPerformanceCounter(&m_t[m_idx++]);
	RECORD( m_t[m_idx++] );
}

__inline
void Stopwatch::split()
{
//	QueryPerformanceCounter(&m_t[m_idx++]);
	RECORD( m_t[m_idx++] );
}

__inline
float Stopwatch::getCurrent()
{
	TIME_TYPE t;
	RECORD( t );
	return (float)( GET_TIME(t) - GET_TIME(m_t[0]) )/GET_FREQ(m_frequency);
}

__inline
void Stopwatch::stop()
{
	split();
}

__inline
float Stopwatch::getMs()
{
//	return (float)(1000*(m_t[1].QuadPart - m_t[0].QuadPart))/m_frequency.QuadPart;
	return (float)( GET_TIME( m_t[1] ) - GET_TIME( m_t[0] ) )/GET_FREQ( m_frequency );
}

__inline
void Stopwatch::getMs(float* times, int capacity)
{
	for(int i=0; i<capacity; i++) times[i] = 0.f;

	for(int i=0; i<std::min(capacity, m_idx-1); i++)
	{
//		times[i] = (float)(1000*(m_t[i+1].QuadPart - m_t[i].QuadPart))/m_frequency.QuadPart;
		times[i] = (float)( GET_TIME( m_t[i+1] ) - GET_TIME( m_t[i] ) )/GET_FREQ( m_frequency );
	}
}

