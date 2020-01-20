from setuptools import find_packages, setup
from grtr import __version__

INSTALL_REQUIRES = [
    'torch==1.3.1',
    'pytorch-transformers==1.2.0',
    'mldc@git+https://github.com/microsoft/dstc8-meta-dialog.git@28b0406c28806e97093b0c3d6391ce48e1d16261',
    'tb-nightly',
]


setup(
    name='grtr',
    description=('dialogue transformer model which is trained to dynamically select between retrieved and generated responses'),
    url='http://tiny.cc/grtr',
    author='Igor Shalyminov <is33@hw.ac.uk>, Hannes Schulz <hannes.schulz@microsoft.com>, Adam Atkinson <adam.atkinson@microsoft.com>',
    author_email='is33@hw.ac.uk, hannes.schulz@microsoft.com, adam.atkinson@microsoft.com',
    version=__version__,
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,   # pytorch, but it needs to be installed through conda anyways
)
