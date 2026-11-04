"""把 C++ 引擎打包成 Python 包：构建期用 g++ 编出 DLL，再随包分发。

用法：
    pip install .
    python setup.py build_native      # 只构建原生库，便于本地调试
"""

import os
import shutil
import subprocess

from setuptools import Command, setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(HERE, "tiny_autograd")
SOURCES = ("Tensor.cpp", "Net.cpp", "FCLayer.cpp", "Graph.cpp")
RUNTIME_DEPS = ("libgomp-1.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll", "libdl.dll")


def build_native():
    cxx = os.environ.get("CXX", "g++")
    out = os.path.join(PKG, "tiny_autograd.dll")
    cmd = [cxx, "-O2", "-std=c++17", "-fopenmp", "-shared",
           "-static-libgcc", "-static-libstdc++", "-I", ROOT,
           os.path.join(HERE, "capi.cpp")]
    cmd.extend(os.path.join(ROOT, s) for s in SOURCES)
    cmd.extend(["-o", out])
    print("building native library:", " ".join(cmd))
    subprocess.check_call(cmd)

    # OpenMP 与 C++ 运行时是动态依赖，随包一起拷过去
    bin_dir = os.path.dirname(shutil.which(cxx) or "")
    for dep in RUNTIME_DEPS:
        src = os.path.join(bin_dir, dep)
        if os.path.exists(src):
            shutil.copy2(src, PKG)
    print("native library ready:", out)


class BuildNative(Command):
    description = "build the C++ engine into tiny_autograd.dll"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_native()


class BuildPy(build_py):
    def run(self):
        build_native()
        super().run()


setup(
    name="tiny-autograd",
    version="0.2.0",
    description="A tiny C++ autograd engine exposed to Python through a C ABI",
    packages=["tiny_autograd"],
    package_data={"tiny_autograd": ["*.dll"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy"],
    cmdclass={"build_py": BuildPy, "build_native": BuildNative},
)
