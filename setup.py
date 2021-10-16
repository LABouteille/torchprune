from setuptools import find_packages, setup

setup(
    name="torchcompress",
    version="0.1",
    author="Ferdinand Mom",
    description="",
    packages=find_packages(),
    classifiers=["License :: OSI Approved :: MIT License"],
    install_requires=["torch"],
    python_requires=">=3.6",
)
