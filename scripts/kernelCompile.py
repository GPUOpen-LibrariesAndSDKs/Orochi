"""Compile GPU kernels for both AMD (hipcc) and NVIDIA (nvcc) targets.

Compiles ParallelPrimitives radix sort kernels into fatbin/hipfb files
for use with precompiled kernel loading.
"""
import json
import subprocess
import sys
from enumArch import enumArch


def get_gpu_list():
    """Load the AMD GPU list from the JSON configuration file."""
    with open("amdGpuList.json") as f:
        return json.load(f)


def get_amd_arches(min_arch):
    """Return the AMD arches to build, falling back to the JSON list."""
    arches = enumArch(min_arch)
    if not arches:
        print("architecture enumeration unavailable; falling back to amdGpuList.json")
        arches = get_gpu_list()["amd"]
    return arches


def compile_kernels(target_index):
    """Compile kernels for the specified target (0=AMD/HIP, 1=NVIDIA/CUDA)."""
    if target_index == 0:
        command = [
            "hipcc",
            "-x", "hip",
            "../ParallelPrimitives/RadixSortKernels.h",
            "-O3", "-std=c++17", "-ffast-math",
            "--cuda-device-only", "--genco",
            "-I../", "-include", "hip/hip_runtime.h",
            "-parallel-jobs=15"
        ]
        for arch in get_amd_arches("gfx900"):
            command.append("--offload-arch=" + arch)
        command += ["-o", "../bitcodes/oro_compiled_kernels.hipfb"]
    else:
        command = [
            "nvcc",
            "-x", "cu",
            "../ParallelPrimitives/RadixSortKernels.h",
            "-O3", "-std=c++17", "--use_fast_math",
            "-fatbin", "-arch=all",
            "-I../", "-include", "cuda_runtime.h",
            "-o", "../bitcodes/oro_compiled_kernels.fatbin"
        ]

    print(" ".join(command))

    return subprocess.Popen(command)


def main():
    targets = {0: "hipcc", 1: "nvcc"}
    processes = [(name, compile_kernels(index)) for index, name in targets.items()]

    failed = []
    for name, proc in processes:
        if proc.wait() != 0:
            failed.append(f"{name} (exit {proc.returncode})")

    if failed:
        print("compile failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)

    print("compile done.")


if __name__ == '__main__':
    main()
