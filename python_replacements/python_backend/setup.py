from setuptools import setup

setup(
    name="python_backend",
    version="0.1.0",
    description="Pure-Python symbolic regression backend for PySR_custom",
    packages=["python_backend"],
    package_dir={"python_backend": "."},
    include_package_data=True,
    package_data={
        "python_backend": ["data/*.yaml"],
    },
    install_requires=["numpy", "pandas", "sympy"],
    python_requires=">=3.9",
)
