import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spacy-extensions",
    version="0.0.1",
    author="XXXXX",
    author_email="XXXXX",
    description="A set of Python SpaCy extension.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XXXX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
