"""Compile GPU kernels for both AMD (hipcc) and NVIDIA (nvcc) targets.

Compiles ParallelPrimitives radix sort kernels into fatbin/hipfb files
for use with precompiled kernel loading.
"""
import json
import os
import subprocess
from enumArch import enumArch


def get_gpu_list():
    """Load the AMD GPU list from the JSON configuration file."""
    with open("amdGpuList.json") as f:
        return json.load(f)


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
        for arch in enumArch("gfx900"):
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

    use_shell = (os.name == 'nt')
    return subprocess.Popen(command, shell=use_shell)


def main():
    processes = [
        compile_kernels(0),
        compile_kernels(1),
    ]

    for proc in processes:
        proc.wait()

    print("compile done.")


if __name__ == '__main__':
    main()