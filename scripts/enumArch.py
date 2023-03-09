import os
import subprocess
import re

def toNumber( arch ):
    return int(arch[3:], 16)

def enumArch( minArch ):
    process = subprocess.Popen(['llc', '-march=amdgcn', '-mcpu=help'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    output, errors = process.communicate()
    lines = output.decode('utf-8').splitlines() + errors.decode('utf-8').splitlines()

    arches = []
    for line in lines:
        result = re.match("\s+(gfx[0-9a-f]+).*processor.", line)
        if result:
            arch = result.group(1)
            if toNumber(minArch) <= toNumber(arch):
                arches.append( arch )
    if not arches: 
        print( "warning: llc may not working" )
    return arches
