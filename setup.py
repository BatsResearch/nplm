from setuptools import setup, find_packages

setup(
    name='nplm',
    version='0.0.1',
    url='https://github.com/BatsResearch/nplm.git',
    author='Peilin Yu, Tiffany Ding, Stephen H. Bach',
    author_email='peilin_yu@brown.edu, sbach@cs.brown.edu',
    description='Programmatic weak supervision for partial labelers (PLFs)',
    packages=find_packages(),
    install_requires=['numpy >= 1.11', 'scipy >= 1.5', 'torch >= 1.4', 'tqdm >= 4.62.3', 'torchvision >= 0.10'],
)
