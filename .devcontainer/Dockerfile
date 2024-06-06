FROM nvcr.io/nvidia/pytorch:24.03-py3

ARG DEBIAN_FRONTEND=noninteractive
#ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    build-essential \
    cmake \
    libeigen3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfmt-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    sudo make install && \
    sudo ldconfig

RUN git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    mkdir build && \
    cd build && \
    cmake .. && make && sudo make install
RUN pip3 install pybind11

WORKDIR /app
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    mkdir build && \
    cd build && \
    cmake .. && make && sudo make install
RUN pip3 install sophuspy

RUN apt update && apt install locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

RUN apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg\
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends ros-humble-ros-base && \
    apt install -y --no-install-recommends ros-dev-tools

CMD ["bash"]
