from setuptools import setup

setup(
    name="juliacall",
    version="0.1.0",
    description="Stub for juliacall — provides Main, seval, AnyValue, VectorValue in the pure-Python backend",
    packages=["juliacall"],
    package_dir={"juliacall": "."},
    install_requires=["numpy>=1.13.0"],
    python_requires=">=3.9",
)
