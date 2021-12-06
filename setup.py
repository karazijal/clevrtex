import setuptools
import os
import logging

logger = logging.getLogger('ool')

reqs = []
# if os.path.exists('other_should_be_deleted/requirements.txt'):
#     with open('other_should_be_deleted/requirements.txt') as reqf:
#         reqs = reqf.readlines()

setuptools.setup(
    name="ool",
    version="0.1.0",
    author="Laurynas Karazija",
    author_email="laurynas@robots.ox.ac.uk",
    description="OOL UMOS",
    long_description="Code for ClevrTex Unsupervised Multi-Object Segmentation Benchmark",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=reqs  # see conda environment.yml
)

