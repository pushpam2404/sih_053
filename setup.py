import os
from setuptools import setup, find_packages

setup(
    name="drdo_lidar_mapping",
    version="0.6.0",
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
    entry_points={
        "console_scripts": [
            "drdo-train=scripts.train:train",
            "drdo-eval=scripts.eval:main",
            "drdo-export-onnx=scripts.export_onnx:main",
            "drdo-benchmark=scripts.benchmark:run_e2e_benchmark",
        ],
    },
)
