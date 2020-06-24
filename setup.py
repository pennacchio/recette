import codecs
import os.path

import setuptools


# Functions used to parse the package version from its top __init__.py following
# the approach in:
# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="recette",
    version=get_version("recette/__init__.py"),
    author="Alan Pennacchio",
    author_email="alanpennacchio@icloud.com",
    description="Data preprocessing utilities on top of pandas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pennacchio/recette",
    packages=["recette"],
    python_requires=">=3.6",
    install_requires=["pandas >= 1.0.3", "toolz >= 0.10.0"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
