import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skybright",
    version="0.0.3",
    author="Eric H. Neilsen, Jr.",
    author_email="neilsen@fnal.gov",
    description="Utility for calculating sky brightness from airglow and moonlight",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ehneilsen/skybright",
    packages=setuptools.find_packages(),
    install_requires=[
        "palpy >= 1.8.0",
        "numpy >= 1.11.0",
        "numexpr >= 2.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
