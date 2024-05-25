from setuptools import setup, find_packages

setup(
    name='custom_transforms',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'torchvision'
    ]
)
