from setuptools import setup, find_packages

setup(
    name='tbccsi',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'tbccsi = tbccsi.cli:main',
        ],
    },
)
