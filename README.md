pyflycapture2
=============

Authors:

    Robert Jordens

License:

    GPLv3+

[Original README](./README.rst)

##Ubuntu Install Instructions

###Install Point Grey FlyCapture2 Library

Links and instructions for downloading and installing the latest
FlyCapture 2.x library from Point Grey for Linux can be found here:

<http://www.ptgrey.com/support/downloads>

Download Linux (64-bit, 32-bit, or ARM, whichever is appropriate).
Requires registration.

```shell
cd ~/Downloads
tar -zxvf flycapture*
cd flycapture*
cat README
# follow the instructions that the script takes you through
sudo reboot
```

###Test Point Grey FlyCapture2 Library

```shell
# plug in Flea3 camera into USB3 port
flycap
```

### Install pyflycapture2

```shell
mkdir ~/git
cd ~/git
git clone https://github.com/peterpolidoro/pyflycapture2.git
sudo apt-get install python-pip python-virtualenv -y
mkdir ~/virtualenvs/
virtualenv ~/virtualenvs/flycapture2
source ~/virtualenvs/flycapture2/bin/activate
pip install cython
pip install numpy
cd ~/git/pyflycapture2/
python setup.py install
```

###Test pyflycapture2

```shell
# plug in Flea3 camera into USB3 port
source ~/virtualenvs/flycapture2/bin/activate
cd ~/git/pyflycapture2/
python test_flycapture2.py
sudo apt-get install ipython -y
```

