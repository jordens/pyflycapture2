try:
    from setuptools import setup, Extension
except ImportError:
    from distutils import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext

setup(
    name="pyflycapture2",
    version="0.1",
    description="python wrapper for libflycapture2 (C-API)",
    author="Robert Jordens",
    author_email="jordens@phys.ethz.ch",
    url="http://launchpad.net/pyflycapture2",
    #packages=["flycapture2"],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("flycapture2",
        sources = ["src/flycapture2.pyx"],
        libraries = ["flycapture-c"],
        include_dirs = ["/usr/include/flycapture/C"])]
)
