pyflycapture2
==

*Cython/Cwrap based python bindings for the FlyCapture v2 C API from PointGrey.*

License: GPLv3+

NOTE: There is "Python support for FlyCapture2 SDK" available from PointGrey now. You may want to try and use those instead of these bindings.
==


Install Point Grey FlyCapture2 Library
--

Links and instructions for downloading and installing the latest
FlyCapture 2.x library from Point Grey for Linux can be found here:

http://www.ptgrey.com/support/downloads

Download Linux (64-bit, 32-bit, or ARM, whichever is appropriate).
This requires registration. Follow the instructions in those packages to
install the libraries.

Install pyflycapture2
--

Ensure that you have installed ``cython`` and ``numpy``.
Use ``virtualenv`` to your advantage.

::
        pip install git+https://github.com/jordens/pyflycapture2.git

Test pyflycapture2
--

::
        python test_flycapture2.py
