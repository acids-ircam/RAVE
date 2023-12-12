import os
import subprocess

import setuptools

# imports __version__
exec(open('rave/version.py').read())

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="acids-rave",
    version=__version__,  # type: ignore
    author="Antoine CAILLON",
    author_email="caillon@ircam.fr",
    description="RAVE: a Realtime Audio Variatione autoEncoder",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        'rave/configs': ['*.gin'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": [
        "rave = scripts.main_cli:main",
    ]},
    install_requires=requirements.split("\n"),
    python_requires='>=3.9',
    include_package_data=True,
)
