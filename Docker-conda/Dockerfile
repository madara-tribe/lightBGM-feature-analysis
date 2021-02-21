FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV CUDA_VERSION 10.0
ENV TF_VERSION 1.13.1
ENV PYTHON_VERSION 3.7.0
WORKDIR /root
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev
# Media I/O:
RUN sudo apt install -y \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libgtk2.0-dev

# Video I/O:
RUN sudo apt install -y \
    libdc1394-22-dev libavcodec-dev libavformat-dev \
    libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev \
    libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN sudo apt install -y libtbb-dev libeigen3-dev


RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall

RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    bash Anaconda3-2019.10-Linux-x86_64.sh -b && \
    rm Anaconda3-2019.10-Linux-x86_64.sh

ENV PATH $PATH:/root/anaconda3/bin
RUN conda install -c conda-forge lightgbm

RUN /root/anaconda3/bin/pip install --upgrade pip
RUN /root/anaconda3/bin/pip install opencv-python
#CMD [ "/bin/bash" ]

