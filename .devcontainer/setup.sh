#!/bin/bash
name=$(basename "$PWD")
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    psmisc \
    iproute2 \
    libjpeg-dev \
    libpng-dev \
    tmux \
    libgl1-mesa-glx \
    x11-apps \
    xorg \
    xterm
if [ ! -f "/opt/conda/envs/${name}/bin/python" ]; then
    mamba create -n ${name} python=3.11 -y
fi
source activate ${name}
pip install isort flake8 black ipython
pip install -r requirements.txt
