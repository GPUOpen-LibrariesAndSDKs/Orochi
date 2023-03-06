import json
import os
import subprocess
import re
from enumArch import enumArch

#for powershell, $ENV:Path+=";..\..\hipsdk\bin"

def getGpuList():
  f = open("amdGpuList.json")
  gpus = json.load(f)
  f.close()
  return gpus

ps = []
def compile( index ):
	if index == 0 :
		command = [
			"hipcc",
			"-x", "hip", "..\ParallelPrimitives\RadixSortKernels.h", "-O3", "-std=c++17", "-ffast-math", "--cuda-device-only", "--genco", "-I../", "-include", "hip/hip_runtime.h", "-parallel-jobs=15"]
		#command.append( "--offload-arch=gfx1100" )
		for i in enumArch( "gfx900" ):
			command.append( "--offload-arch=" + i )
		command.append( "-o" )
		command.append( "../bitcodes/oro_compiled_kernels.hipfb" )
	else:
		command = [
			'nvcc', '-x','cu','..\ParallelPrimitives\RadixSortKernels.h','-O3', '-std=c++17', '--use_fast_math', '-fatbin', '-arch=all', 
			'-I../', '-include', 'cuda_runtime.h' ]
		command.append( '-o' )
		command.append('../bitcodes/oro_compiled_kernels.fatbin')

	print( " ".join( command ) )

	if os.name == 'nt':
		ps.append( subprocess.Popen( command, shell=True ) )
	else:
		ps.append( subprocess.Popen( command ) )

compile( 0 )
compile( 1 )

for p in ps:
	p.wait()

print( "compile done." )