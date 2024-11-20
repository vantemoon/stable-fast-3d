# Use the NVIDIA CUDA base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Copy the Hugging Face token into the image
COPY hf_token.txt /.huggingface/token/hf_token.txt

# Copy the Hugging Face login script
COPY hf_login.py /.huggingface/hf_login.py

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:$PATH

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Update PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Update Conda
RUN conda update -n base -c defaults conda -y

# Create a new Conda environment
RUN conda create -n sf3d-env python=3.9.15 -y

# Activate the environment in the PATH
ENV PATH=$CONDA_DIR/envs/sf3d-env/bin:$CONDA_DIR/bin:$PATH

# Install PyTorch and CUDA with Conda
RUN conda install -n sf3d-env pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Remove libstdc++.so.6 from the Conda environment
RUN rm -f $CONDA_DIR/envs/sf3d-env/lib/libstdc++.so.6

# Clone the repository directly into /app
RUN git clone https://github.com/vantemoon/stable-fast-3d.git /app

# Install Python dependencies
RUN conda run -n sf3d-env pip install --upgrade pip setuptools==69.5.1 wheel

# Install required Python packages
RUN conda run -n sf3d-env pip install -r requirements.txt
RUN conda run -n sf3d-env pip install -r requirements-demo.txt

# Update required packages
RUN conda run -n sf3d-env pip install --upgrade pydantic gradio

# Install huggingface_hub
RUN conda run -n sf3d-env pip install huggingface_hub

# Copy entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
