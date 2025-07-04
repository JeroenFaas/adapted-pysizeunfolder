# 
# setup.py : adapted-pysizeunfolder
#

from setuptools import setup, Extension


extensions = [
    Extension("pysizeunfolder.interface",
              sources=["pysizeunfolder/interface.c", ],
              )
]


setup(
    name="adapted-pysizeunfolder",
    version="1.0",
    description="Functions for stereological estimation of particle size distributions and taking random sections of polytopes",
    author="Jeroen Faas",
    packages=["pysizeunfolder", ],
    package_dir={"pysizeunfolder": "pysizeunfolder"},
    install_requires=['cython', 'numpy', 'scipy', 'KDEpy', 'statsmodels', 'scikit-learn', ],
    ext_modules=extensions,
    keywords=["geometry", "mathematics", ],
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    test_suite="test",
)
