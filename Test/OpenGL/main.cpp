#define GLEW_STATIC
#include <UnitTest/contrib/glew/include/glew/glew.h>
#include <UnitTest/contrib/glfw/include/GLFW/glfw3.h>



int main( int argc, char** argv ) 
{ 
	GLFWwindow* window;
	if( !glfwInit() ) return 0;

	glfwWindowHint( GLFW_RED_BITS, 32 );
	glfwWindowHint( GLFW_GREEN_BITS, 32 );
	glfwWindowHint( GLFW_BLUE_BITS, 32 );

	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );

	window = glfwCreateWindow( 1280, 720, "orochiOglInterop", NULL, NULL );
	if( !window )
	{
		glfwTerminate();
		return 0;
	}
	glfwMakeContextCurrent( window );

	if( glewInit() != GLEW_OK )
	{
		glfwTerminate();
		return 0;
	}

	int a = 0;
	a++;


}