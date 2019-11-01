from setuptools import setup, find_packages

setup(
    name="self-supervised-3d-tasks",
    version="0.0.1",
    packages=find_packages(),

    package_data={
        'permutations': ['*.bin'],
    },
)
