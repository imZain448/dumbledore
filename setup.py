import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dumbledore", # Replace with your own username
    version="0.0.2",
    author="imzain448",
    author_email="ahmadzain.448@gmail.com",
    description="a python package to visvalize features of data and observe relation without writing lots of codes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imZain448/dumbledore",
    packages=setuptools.find_packages(),
    data_files = [('images',['images/*'])],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['matplotlib' , 'seaborn' , 'pandas' , 'numpy']
)