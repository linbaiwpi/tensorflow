#!/bin/bash

# run the following command before running this script
# docker run -it -v <PATH TO tensorflowON YOUR MACHINE>:/tensorflow --name tflite_soc_v0 ubuntu:20.04 /bin/bash
# cd /tensorflow && ./docker_build.sh

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install build-essential -y
apt-get install openjdk-11-jdk -y
apt-get install zip -y
apt-get install unzip -y
apt-get install nano -y
apt-get install wget -y
apt-get install git-core -y

wget https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-installer-linux-x86_64.sh
chmod a+x bazel-4.2.1-installer-linux-x86_64.sh
./bazel-4.2.1-installer-linux-x86_64.sh

cd /opt
mkdir cmake-3.22.2-linux-x86_64
cd cmake-3.22.2-linux-x86_64
wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.sh
chmod a+x cmake-3.22.2-linux-x86_64.sh
./cmake-3.22.2-linux-x86_64.sh --skip-license
ln -s /opt/cmake-3.22.2-linux-x86_64/bin/* /usr/bin/

echo '=========> apt install python3.8 -y'
apt install -y python3.8
echo '=========> ln -s /usr/bin/python3.8 /usr/bin/python'
ln -s /usr/bin/python3.8 /usr/bin/python3
ln -s /usr/bin/python3.8 /usr/bin/python
echo '=========> apt-get install python3-pip -y'
apt-get install -y python3-pip
echo '=========> ln -s /usr/bin/pip3 /usr/bin/pip'
ln -s /usr/bin/pip3 /usr/bin/pip
echo '=========> wget https://bootstrap.pypa.io/get-pip.py'
wget https://bootstrap.pypa.io/get-pip.py
echo '=========> python3.8 get-pip.py'
python3 get-pip.py
echo '=========> python -m pip install numpy'
# python -m pip install numpy
pip install numpy

#yes "" | ./configure; python -c "import numpy as np"
apt-get install gdb -y

cd /tensorflow
