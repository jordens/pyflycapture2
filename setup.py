#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   pyflycapture2 - python bindings for libflycapture2_c
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy
import os

if os.name == "posix":
    libname = "flycapture-c"
else:
    libname = "flycapture2_c"

if os.path.exists("C:/Program Files (x86)/Point Grey Research/FlyCapture2"):
    pointgrey_win = "C:/Program Files (x86)/Point Grey Research/FlyCapture2"
else:
    pointgrey_win = "C:/Program Files/Point Grey Research/FlyCapture2"

if os.path.exists(pointgrey_win+"/lib/C"):
    libfolder = "/lib/C"
else:
    libfolder = "/lib64/C"

setup(
    name="pyflycapture2",
    description="python wrapper for libflycapture2 (C-API)",
    long_description=
"""The library itself is available from PointGrey:
http://www.ptgrey.com/support/downloads/download.asp (login required)
API docs:
http://www.ptgrey.com/support/downloads/documents/flycapture/Doxygen/C/html/index.html
(C API)
http://www.ptgrey.com/support/downloads/documents/flycapture/Doxygen/html/index.html
(C++ API)""",
    version="0.1+dev",
    author="Robert Jordens",
    author_email="jordens@phys.ethz.ch",
    url="http://launchpad.net/pyflycapture2",
    license="GPLv3+",
    install_requires=["numpy"],
    #packages=["flycapture2"],
    cmdclass = {'build_ext': build_ext},
    #"test_flycapture2.py", "convert.py"
    ext_modules = [Extension("flycapture2",
        sources = ["src/flycapture2.pyx", "src/flycapture2_enums.pxi",
            "src/_FlyCapture2Defs_C.pxd", "src/_FlyCapture2_C.pxd",],
        libraries = [libname],
        library_dirs = ["%s%s" % (pointgrey_win, libfolder)],
        include_dirs = ["/usr/include/flycapture/C",
            "%s/include/C" % pointgrey_win,
            numpy.get_include(), ],
        ),]
)
