import os
import subprocess
import re

def toNumber( arch ):
    return int(arch[3:], 16)

def rocmRoot():
    root = os.environ.get( "ROCM_PATH" ) or os.environ.get( "HIP_PATH" )
    if root:
        return root
    return "/opt/rocm" if os.name != "nt" else ""

def rocmTool( name ):
    # Resolve from the ROCm root, not PATH: a non-AMD clang or a pip-installed ROCm may come first there.
    root = rocmRoot()
    exe = name + ".exe" if os.name == "nt" else name
    for d in ( os.path.join( "lib", "llvm", "bin" ), "bin", os.path.join( "llvm", "bin" ) ):
        path = os.path.join( root, d, exe )
        if os.path.isfile( path ):
            return path
    raise FileNotFoundError( f"{exe} not found under ROCm root '{root}' (set ROCM_PATH or HIP_PATH)" )

def amdclang():
    try:
        command = [ rocmTool( "amdclang++" ) ]
    except FileNotFoundError:
        # Older HIP SDKs on Windows ship only clang++, the same driver under another name.
        command = [ rocmTool( "clang++" ) ]
    root = rocmRoot()
    # pip-installed ROCm (TheRock) keeps the device libraries only under lib/llvm.
    devLib = os.path.join( root, "lib", "llvm", "amdgcn", "bitcode" )
    if not os.path.isdir( os.path.join( root, "amdgcn", "bitcode" ) ) and os.path.isdir( devLib ):
        command.append( "--hip-device-lib-path=" + devLib )
    return command

def enumArch( minArch ):
    # llvm-mc rather than llc: the ROCm SDK produced by TheRock no longer ships llc,
    # and llc prints the list then blocks reading stdin. Both print the same list.
    process = subprocess.Popen([rocmTool('llvm-mc'), '-triple=amdgcn', '-mcpu=help'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    output, errors = process.communicate()
    lines = output.decode('utf-8').splitlines() + errors.decode('utf-8').splitlines()

    arches = []
    for line in lines:
        # \s after the name skips generic targets (gfx9-generic) and variants (gfx1250-strict).
        result = re.match(r"\s+(gfx[0-9a-f]+)\s.*processor\.", line)
        if result:
            arch = result.group(1)
            if toNumber(minArch) <= toNumber(arch):
                arches.append( arch )
    if not arches:
        print( "warning: llvm-mc may not working" )
    return arches
