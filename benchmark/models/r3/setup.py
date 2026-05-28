from pathlib import Path

from setuptools import find_namespace_packages, setup


def read_requirements():
    req_path = Path(__file__).with_name("requirements.txt")
    requirements = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="R3",
    version="1.0",
    python_requires=">=3.10",
    packages=find_namespace_packages(
        include=["R3", "R3.*", "depth_anything_3", "depth_anything_3.*"]
    ),
    install_requires=read_requirements(),
    package_data={
        "R3": ["configs/*.yaml"],
        "R3.configs": ["*.yaml"],
        "depth_anything_3": ["configs/*.yaml"],
        "depth_anything_3.configs": ["*.yaml"],
    },
)
