import os
import subprocess
import sys
from pathlib import Path

TARGET = "cnn.exe"

NVCC = "nvcc"
CL   = "cl" 

INCLUDE_DIR = "include"
SRC_DIR = "src"
BUILD_DIR = "build"

CUDA_ARCH = "sm_89"

# Source files 
CU_SRCS = [
    "main.cu",
    "train.cu",
    "optimize/conv_tile.cu",
    "layers/pool_forward.cu",
    "layers/pool_backward.cu",
    "layers/fc_forward.cu",
    "layers/fc_backward.cu",
    "ops/softmax_loss.cu",
    "ops/sgd_update.cu",
    "layers/conv_backward.cu",
    "ops/relu_activate.cu",
    "utils/cuda_utils.cu",
    "utils/timer.cu",
]

CPP_SRCS = [
    "mnist_loader.cpp",
]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def run(cmd, *, shell=False):

    if isinstance(cmd, list):
        print(" ".join(cmd))
    else:
        print(cmd)
    r = subprocess.run(cmd, shell=shell)
    if r.returncode != 0:
        sys.exit(r.returncode)

def cu_obj_path(src_rel: str) -> Path:
    return Path(BUILD_DIR) / src_rel.replace(".cu", ".obj")

def cpp_obj_path(src_rel: str) -> Path:
    return Path(BUILD_DIR) / src_rel.replace(".cpp", ".obj")

def compile_cuda():
    objs = []
    for rel in CU_SRCS:
        src = Path(SRC_DIR) / rel
        obj = cu_obj_path(rel)
        ensure_dir(obj.parent)

        cmd = [
            NVCC,
            "-O3", "-std=c++14",
            f"-I{INCLUDE_DIR}",
            f"-arch={CUDA_ARCH}",
            '-Xcompiler=/MD',
            '-Xcompiler=/EHsc',
            "-c", str(src),
            "-o", str(obj),
        ]

        run(cmd)
        objs.append(str(obj))
    return objs

def compile_cpp():
    objs = []
    for rel in CPP_SRCS:
        src = Path(SRC_DIR) / rel
        obj = cpp_obj_path(rel)
        ensure_dir(obj.parent)

        cmd = [
            CL,
            "/nologo",
            "/O2",
            "/std:c++14",
            "/EHsc",
            "/MD",
            f'/I{INCLUDE_DIR}',
            "/c", str(src),
            f"/Fo{obj}",
        ]
        run(cmd)
        objs.append(str(obj))
    return objs

def link(all_objs):
    cmd = [NVCC, f"-arch={CUDA_ARCH}", "-o", TARGET, '-Xcompiler=/MD'] + all_objs

    run(cmd)

def clean():
    if Path(BUILD_DIR).exists():
        run(f'rmdir /S /Q "{BUILD_DIR}"', shell=True)
    if Path(TARGET).exists():
        os.remove(TARGET)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "clean":
        clean()
        sys.exit(0)

    cu_objs = compile_cuda()
    cpp_objs = compile_cpp()
    link(cu_objs + cpp_objs)

    print("\nBuild successful:", TARGET)
