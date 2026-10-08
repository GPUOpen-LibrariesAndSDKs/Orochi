import os
import subprocess
import re

def toNumber( arch ):
    return int(arch[3:], 16)

def enumArch( minArch ):
    # llvm-mc rather than llc: the ROCm SDK produced by TheRock no longer ships llc,
    # and llc prints the list then blocks reading stdin. Both print the same list.
    process = subprocess.Popen(['llvm-mc', '-triple=amdgcn', '-mcpu=help'], shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    output, errors = process.communicate()
    lines = output.decode('utf-8').splitlines() + errors.decode('utf-8').splitlines()

    arches = []
    for line in lines:
        result = re.match(r"\s+(gfx[0-9a-f]+).*processor.", line)
        if result:
            arch = result.group(1)
            if toNumber(minArch) <= toNumber(arch):
                arches.append( arch )
    if not arches: 
        print( "warning: llvm-mc may not working" )
    return arches
