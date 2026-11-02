import json
import os
import subprocess
import re
from enumArch import enumArch, amdclang

# Paths below are relative to this directory.
os.chdir( os.path.dirname( os.path.abspath( __file__ ) ) )

def getGpuList():
  f = open("amdGpuList.json")
  gpus = json.load(f)
  f.close()
  return gpus

ps = []
def compile( index ):
	if index == 0 :
		# -cuid: a fixed id instead of a random one keeps the output reproducible.
		command = amdclang() + [
			"-x", "hip", "../ParallelPrimitives/RadixSortKernels.h", "-O3", "-std=c++17", "-ffast-math", "--cuda-device-only", "-I../", "-include", "hip/hip_runtime.h",
			"-cuid=RadixSortKernels", "--offload-compress", "--offload-arch=" + ",".join( enumArch( "gfx900" ) ) ]
		command.append( "-o" )
		command.append( "../bitcodes/oro_compiled_kernels.hipfb" )
	else:
		command = [
			'nvcc', '-x','cu','../ParallelPrimitives/RadixSortKernels.h','-O3', '-std=c++17', '--use_fast_math', '-fatbin', '-arch=all', '--threads', '4',
			'-I../', '-include', 'cuda_runtime.h' ]
		command.append( '-o' )
		command.append('../bitcodes/oro_compiled_kernels.fatbin')

	print( " ".join( command ) )

	try:
		ps.append( subprocess.Popen( command ) )
	except FileNotFoundError:
		print( "warning: " + command[0] + " not found, skipped" )

compile( 0 )
compile( 1 )

for p in ps:
	p.wait()

print( "compile done." )