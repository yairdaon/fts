from setuptools import find_packages, setup

setup(
    name="fts",
    version="1",
    description="Fast two-strain model",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    #url="https://github.com/ArjanCodes/2023-package",
    author="Yair Daon",
    author_email="firstname.lastname@gmail.com",
    license="MIT",
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3.10",
    #     "Operating System :: OS Independent",
    # ],
    #install_requires=["bson >= 0.5.10"],
    #extras_require={
    #    "dev": ["pytest>=7.0", "twine>=4.0.2"],
    #},
    #python_requires=">=3.10",
)
