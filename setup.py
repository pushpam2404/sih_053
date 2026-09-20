import os
from setuptools import setup, find_packages

setup(
    name="drdo_lidar_mapping",
    version="0.7.0",
    description="Adaptive 2.5D Elevation & Semantic Mapping for Off-Road Autonomous Vehicles",
    author="DRDO ID26053 Autonomous Systems Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "deploy": ["onnx>=1.12.0", "pycuda"],
        "test": ["pytest>=6.0.0"],
    },
    # No console_scripts. There used to be four (drdo-train, drdo-eval, drdo-export-onnx,
    # drdo-benchmark) and every one of them was broken: they pointed at a `scripts` module, but
    # scripts/ has no __init__.py and is not picked up by find_packages(), so `pip install -e .`
    # installed four commands that raised ModuleNotFoundError on invocation. Two of the target
    # functions (run_e2e_benchmark, export_onnx.main) no longer exist either.
    # The scripts are run directly and are documented that way in README and CLAUDE.md:
    #     .venv/bin/python scripts/benchmark.py --frames 200
    # Re-add entry points only alongside packaging scripts/ properly.
)
