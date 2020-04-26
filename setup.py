import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="recette",
    version="0.1.0dev",
    author="Alan Pennacchio",
    author_email="alanpennacchio@icloud.com",
    description="Data preprocessing utilities on top of pandas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pennacchio/recette",
    packages=["recette"],
    python_requires=">=3.6",
    install_requires=["pandas >= 1.0.3"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
