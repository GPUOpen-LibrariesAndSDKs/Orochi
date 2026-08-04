"""Enumerate AMD GPU architectures supported by the installed LLVM toolchain.

Provides enumArch(minArch) which returns a list of supported gfx targets
at or above the given minimum architecture.

Queries ROCm's clang first; llc is only a fallback because ROCm stopped
shipping it as of 7.14.
"""
import os
import re
import shutil
import subprocess

# Matches the leading "gfxNNN" column of both clang's and llc's -mcpu=help
# listing, rejecting the "gfx10-3-generic" style pseudo targets.
_ARCH_PATTERN = re.compile(r"^\s+(gfx[0-9a-f]+)(?=\s|$)")

# Candidate command lines, in priority order. Each yields a -mcpu=help listing.
_ENUM_COMMANDS = (
    ('clang', ['--target=amdgcn-amd-amdhsa', '-mcpu=help']),
    ('llc', ['-march=amdgcn', '-mcpu=help']),
)


def to_number(arch):
    """Convert a gfx architecture string (e.g. 'gfx900') to an integer."""
    return int(arch[3:], 16)


def _rocm_llvm_bin():
    """Return ROCm's LLVM bin directory, or None if ROCm is not installed."""
    for var in ('ROCM_PATH', 'HIP_PATH'):
        root = os.environ.get(var)
        if root:
            return os.path.join(root, 'lib', 'llvm', 'bin')

    hipconfig = shutil.which('hipconfig')
    if hipconfig:
        result = subprocess.run(
            [hipconfig, '-l'], capture_output=True, text=True, check=False
        )
        path = result.stdout.strip()
        if result.returncode == 0 and path:
            return path

    for root in ('/opt/rocm', 'C:/Program Files/AMD/ROCm'):
        candidate = os.path.join(root, 'lib', 'llvm', 'bin')
        if os.path.isdir(candidate):
            return candidate

    return None


def _find_tool(name):
    """Locate a toolchain executable, preferring ROCm's own copy over PATH."""
    rocm_bin = _rocm_llvm_bin()
    if rocm_bin:
        found = shutil.which(name, path=rocm_bin)
        if found:
            return found

    return shutil.which(name)


def _parse_arches(lines):
    """Extract gfx target names from a -mcpu=help listing."""
    arches = []
    for line in lines:
        match = _ARCH_PATTERN.match(line)
        # Reject truncated names such as "gfx9", left over from llc's
        # generic-target rows; a real target has at least three digits.
        if match and len(match.group(1)) >= 6:
            arches.append(match.group(1))

    return arches


# camelCase is kept because external build scripts import this name.
def enumArch(min_arch):
    """Return list of AMD GPU architectures >= min_arch, or [] if none found."""
    min_value = to_number(min_arch)

    for name, args in _ENUM_COMMANDS:
        tool = _find_tool(name)
        if not tool:
            continue

        try:
            result = subprocess.run(
                [tool] + args, capture_output=True, text=True, check=False
            )
        except OSError as exc:
            print(f"warning: could not run {tool}: {exc}")
            continue

        # Both tools print the listing to stderr on some versions.
        lines = result.stdout.splitlines() + result.stderr.splitlines()
        arches = [a for a in _parse_arches(lines) if min_value <= to_number(a)]

        if arches:
            return arches

        print(f"warning: {tool} listed no architecture >= {min_arch}")

    print("warning: no working LLVM toolchain found to enumerate architectures")

    return []
