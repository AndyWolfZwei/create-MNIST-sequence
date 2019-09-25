from setuptools import setup,find_packages

setup(
    name='create_sequence',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'opencv-python',
        'numpy',
        'six'
    ],
    description='Create a series of digits sequence randomly choice from MNIST dataset with multiprocessing',
    author='Zhiwei Zhuang',
    author_email='254391187@qq.com',
)