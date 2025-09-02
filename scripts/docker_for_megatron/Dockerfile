# Build and run the docker image with the following command:
#
# cd <Trinity-RFT root dir>
# docker build -f scripts/docker_for_megatron/Dockerfile -t trinity-rft-megatron:latest .
# docker run -it --gpus all --shm-size="64g" --rm -v $PWD:/workspace -v <root_path_of_data_and_checkpoints>:/data trinity-rft-megatron:latest


FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

WORKDIR /workspace

# copy the Trinity-RFT dir into the workspace
COPY . .

RUN apt update && apt install -y \
    build-essential \
    curl git wget vim tmux net-tools \
    python3 python3-pip python3-dev python3-packaging \
    libomp-dev infiniband-diags libibverbs-dev librdmacm-dev rdma-core perftest \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# For Aliyun users: update pip mirror to aliyun to speed up pip install
RUN pip config set global.index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
   && pip config set install.trusted-host mirrors.cloud.aliyuncs.com

# Install Trinity-RFT with Megatron
RUN pip install --upgrade pip \
    && pip install -e .[megatron] \
    && pip install flash-attn==2.8.1 \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        --resume-retries 999 git+https://github.com/NVIDIA/apex.git

# Set Env variables

# WANDB
# ENV WANDB_API_KEY=
# ENV WANDB_BASE_URL=

# LLM API
# ENV OPENAI_API_KEY=
# ENV DASH_API_KEY=
