FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
#RUN wget https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
#RUN pip install -r FastSAM/requirements.txt

RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install -U diffusers==0.11.1 transformers ftfy gradio accelerate
RUN pip install torch torchvision
RUN pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
RUN pip install simple-lama-inpainting

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install -r requirements.txt

# Entrypoint bash
CMD ["/bin/bash"]
