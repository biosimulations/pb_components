import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="pb_multiscale_actin",
    version="1.3.0",
    packages=[
        "pb_multiscale_actin",
        "pb_multiscale_actin.processes",
        "pb_multiscale_actin.composites",
        "pb_multiscale_actin.experiments",
    ],
    author="Ezequiel Valencia",
    author_email="ezq.valencia@pm.me",
    url="",
    license="Apache Software License 2.0",
    entry_points={"console_scripts": []},
    short_description="Simulate actin and membranes multiscale with Vivarium as connector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={},
    include_package_data=True,
    install_requires=[
        "vivarium-core",
        "process-bigraph",
        "bigraph-schema",
        "simularium-readdy-models",
    ],
)
