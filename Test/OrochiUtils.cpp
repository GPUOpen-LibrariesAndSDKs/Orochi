#include <Test/OrochiUtils.h>
#include <string>
#include <iostream>
#include <fstream>


struct OrochiUtilsImpl
{
	static
	void readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes )
	{
		std::fstream f( path );
		if( f.is_open() )
		{
			size_t sizeFile;
			f.seekg( 0, std::fstream::end );
			size_t size = sizeFile = (size_t)f.tellg();
			f.seekg( 0, std::fstream::beg );
			if( includes )
			{
				sourceCode.clear();
				std::string line;
				char buf[512];
				while( std::getline( f, line ) )
				{
					if( strstr( line.c_str(), "#include" ) != 0 )
					{
						const char* a = strstr( line.c_str(), "<" );
						const char* b = strstr( line.c_str(), ">" );
						int n = b - a - 1;
						memcpy( buf, a + 1, n );
						buf[n] = '\0';
						includes->push_back( buf );
						sourceCode += line + '\n';
					}
					else
					{
						sourceCode += line + '\n';
					}
				}
			}
			else
			{
				sourceCode.resize( size, ' ' );
				f.read( &sourceCode[0], size );
			}
			f.close();
		}
	}

};

oroFunction OrochiUtils::getFunctionFromFile( const char* path, const char* funcName, std::vector<const char*>* optsIn )
{ 
	std::string source;
	OrochiUtilsImpl::readSourceCode( path, source, 0 );

	return getFunction( source.c_str(), path, funcName, optsIn );
/*
	const char* code = source.c_str();
	oroFunction function;

	orortcProgram prog;
	orortcResult e;
	e = orortcCreateProgram( &prog, code, path, 0, 0, 0 );
	std::vector<const char*> opts;
	opts.push_back( "-I ../" );
	opts.push_back( "-G" );

	e = orortcCompileProgram( prog, opts.size(), opts.data() );
	if( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		orortcGetProgramLogSize( prog, &logSize );
		if( logSize )
		{
			std::string log( logSize, '\0' );
			orortcGetProgramLog( prog, &log[0] );
			std::cout << log << '\n';
		};
	}
	size_t codeSize;
	e = orortcGetCodeSize( prog, &codeSize );

	std::vector<char> codec( codeSize );
	e = orortcGetCode( prog, codec.data() );
	e = orortcDestroyProgram( &prog );
	oroModule module;
	oroError ee = oroModuleLoadData( &module, codec.data() );
	ee = oroModuleGetFunction( &function, module, funcName );

	return function;
*/
}

oroFunction OrochiUtils::getFunction( const char* code, const char* path, const char* funcName, std::vector<const char*>* optsIn )
{
	oroFunction function;

	orortcProgram prog;
	orortcResult e;
	e = orortcCreateProgram( &prog, code, path, 0, 0, 0 );
	std::vector<const char*> opts;
	opts.push_back( "-I ../" );
	opts.push_back( "-G" );

	e = orortcCompileProgram( prog, opts.size(), opts.data() );
	if( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		orortcGetProgramLogSize( prog, &logSize );
		if( logSize )
		{
			std::string log( logSize, '\0' );
			orortcGetProgramLog( prog, &log[0] );
			std::cout << log << '\n';
		};
	}
	size_t codeSize;
	e = orortcGetCodeSize( prog, &codeSize );

	std::vector<char> codec( codeSize );
	e = orortcGetCode( prog, codec.data() );
	e = orortcDestroyProgram( &prog );
	oroModule module;
	oroError ee = oroModuleLoadData( &module, codec.data() );
	ee = oroModuleGetFunction( &function, module, funcName );

	return function;
}

void OrochiUtils::launch1D( oroFunction func, int nx, const void** args, int wgSize, unsigned int sharedMemBytes ) 
{
	int4 tpb = { wgSize, 1, 0 };
	int4 nb = { ( nx + tpb.x - 1 ) / tpb.x, 1, 0 };
	oroError e = oroModuleLaunchKernel( func, nb.x, nb.y, 1, tpb.x, tpb.y, 1, sharedMemBytes, 0, (void**)args, 0 );
	OROASSERT( e == oroSuccess, 0 );
}