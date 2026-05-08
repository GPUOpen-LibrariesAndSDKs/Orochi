"""Enumerate AMD GPU architectures available via LLVM's llc tool.

Provides enumArch(minArch) which returns a list of supported gfx targets
at or above the given minimum architecture.
"""
import os
import re
import subprocess


def to_number(arch):
    """Convert a gfx architecture string (e.g. 'gfx900') to an integer."""
    return int(arch[3:], 16)


def enumArch(min_arch):
    """Return list of AMD GPU architectures >= min_arch using llc."""
    process = subprocess.Popen(
        ['llc', '-march=amdgcn', '-mcpu=help'],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    output, errors = process.communicate()
    lines = output.decode('utf-8').splitlines() + errors.decode('utf-8').splitlines()

    min_value = to_number(min_arch)
    arches = []

    for line in lines:
        match = re.match(r"\s+(gfx[0-9a-f]+).*processor.", line)
        if match:
            arch = match.group(1)
            if min_value <= to_number(arch):
                arches.append(arch)

    if not arches:
        print("warning: llc may not be working")

    return arches
