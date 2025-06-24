# adapted-pysizeunfolder
A python repository for estimating the 3D particle size distribution based on areas and numbers of vertices 
in 2D section profiles of the particles. This procedure is referred to as 'unfolding'. 
Much of the code is adapted or directly reused from https://github.com/thomasvdj/pysizeunfolder.

## Installation and dependencies
The repository may be installed by running:

```
pip install git+https://github.com/JeroenFaas/adapted-pysizeunfolder
```

Dependencies for this library are installed automatically. Note that some depend on Python versions up to 
and including Python 3.11.

## Code examples
The files `github examples 2d.py`, `github examples 3d.py`, `github estimation example.py` and 
`github examples application.py` in the `examples` folder contain code examples that showcase how to use 
the various functions in this library. Some use the previously run results stored in `.pkl` files. These files are 
located in the many folders in `examples`, zipped in order to allow the uploading of large files to GitHub. 
It is therefore advised to unzip these folders and to place their content files in the `examples` folder.
