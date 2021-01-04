import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GAopt", # Replace with your own username
    version="1.0",
    author="Omar Zaki",
    author_email="OmarZaki9696@gmail.com",
    description="A genetic algorithm package for optimisation problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OmarZaki96/GAopt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
