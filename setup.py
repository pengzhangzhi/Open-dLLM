import importlib.metadata
import importlib.util
import os
import re
from typing import List

from setuptools import find_packages, setup


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_torch_npu_available() -> bool:
    return _is_package_available("torch_npu")


def _is_torch_available() -> bool:
    return _is_package_available("torch")


def _is_torch_cuda_available() -> bool:
    if _is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def get_version() -> str:
    with open(os.path.join("veomni", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


CUDA_REQUIRE = ["liger-kernel>=0.4.1,<1.0"]

NPU_REQUIRE = ["torchvision>=0.16.0,<0.16.1"]

EXTRAS_REQUIRE = {"dev": ["pre-commit>=4.0.0,<5.0", "ruff>=0.7.0,<1.0", "pytest>=6.0.0,<8.0", "expecttest>=0.3.0"]}

BASE_REQUIRE = [
    "byted-hdfs-io",
    "diffusers>=0.30.0,<=0.31.0",
    "tiktoken>=0.9.0",
    "blobfile>=3.0.0",
    "bytecheckpoint",
    "transformers==4.54.1",
    "accelerate",
    "datasets",
    "peft",
    "hf-transfer",
    "codetiming",
    "hydra-core",
    "pandas",
    "pyarrow>=15.0.0",
    "pylatexenc",
    "wandb",
    "ninja",
    "packaging",
]


def main():
    # Update install_requires and extras_require
    install_requires = BASE_REQUIRE

    if _is_torch_npu_available():
        install_requires.extend(NPU_REQUIRE)
    elif _is_torch_cuda_available():
        install_requires.extend(CUDA_REQUIRE)

    setup(
        name="open-dcoder",
        version=get_version(),
        python_requires=">=3.8.0",
        packages=find_packages(exclude=["scripts", "tasks", "tests"]),
        url="https://github.com/pengzhangzhi/dLLM-training",
        license="Apache 2.0",
        author="Fred Z. Peng",
        author_email="zp70@duke.edu",
        description="Open-dCoder: Open Diffusion Large Language Model for Code Generation",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        install_requires=install_requires,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=False,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Code Generators",
        ],
        keywords="diffusion, language model, code generation, machine learning, pytorch",
    )


if __name__ == "__main__":
    main()
