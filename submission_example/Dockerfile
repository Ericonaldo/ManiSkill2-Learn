# Before buliding this docker, move this docker file to the parent directory of ManiSkill2-Learn,
# and move user_solution.py to be directly under ManiSkill2-Learn/ 

FROM haosulab/mani-skill2:latest

# Install os-level packages
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
# 	--allow-change-held-packages --no-install-recommends \
#     libglvnd0 \
#     libgl1 \
#     libglx0 \
#     libgles2  \
#     libxcb1-dev \
#     tzdata \
#     build-essential \
#     ffmpeg \
#     libssl-dev \
#     net-tools \
#     vim \
#     libgl1-mesa-glx \
#     libosmesa6 \
#     libosmesa6-dev \
#     libglew-dev \
#     libglfw3 \
#     libglfw3-dev \
#     sudo \
#     libcurl4-openssl-dev \
#     xorg \
#     openbox \
#     patchelf \
#     librdmacm1 \
#     libibverbs1 \
#     bash-completion \
#     ca-certificates \
#     cmake \
#     curl \
#     git \
#     htop \
#     libegl1 \
#     libxext6 \
#     libjpeg-dev \
#     libpng-dev  \
#     libvulkan1 \
#     rsync \
#     tmux \
#     unzip \
#     vim \
#     vulkan-utils \
#     wget \
#     xvfb

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Install additional python packages you need
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

# Copy your codes and model weights
COPY ManiSkill2-Learn /root/ManiSkill2-Learn
ENV PYTHONPATH /root/ManiSkill2-Learn:$PYTHONPATH

# Install ManiSkill2-Learn
RUN pip install pytorch3d -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    # && pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/ \
    # && pip install torchvision==0.13.1 torchaudio==0.12.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install pytransform3d -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install zarr -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install protobuf==3.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && cd /root/ManiSkill2-Learn \
    && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/  \
    && conda clean -ya && pip cache purge

WORKDIR /root/ManiSkill2-Learn
# Run the following if you need to use SparseConvNet
# Building SparseConvNet requires access to local GPUs, even though it essentially uses CUDA
# In order to enable local GPU access during docker building, see https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime

# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#     libsparsehash-dev \
#     && rm -rf /var/lib/apt/lists/*
# ENV CUDA_HOME /usr/local/cuda-11.3
# RUN pip install torchsparse@git+https://github.com/lz1oceani/torchsparse.git



# To run the docker locally (not for submission),
# docker run -it --rm -v SOURCE_MOUNT_DIRECTORY:TARGET_PATH --gpus all THIS_DOCKER_NAME:latest  bash
# You might also need to "export PYTHONPATH=SOME_PATH:$PYTHONPATH"

# Follow https://haosulab.github.io/ManiSkill2/benchmark/submission.html for docker testing instructions.