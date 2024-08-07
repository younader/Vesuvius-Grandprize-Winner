# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Install gdown to handle Google Drive file downloads
RUN pip install gdown

# Download the canonical timesformer weights from Google Drive using gdown
RUN gdown --id 13Iu-dR-1sKq_oGJfNa86LcBSv1o4XA37 -O /workspace/timesformer_weights.ckpt

# Git clone the ThaumatoAnakalyptor repository into the workspace as base
RUN apt-get update \
    && apt-get -y install git
RUN git clone https://github.com/younader/Vesuvius-Grandprize-Winner /Vesuvius-Grandprize-Winner
RUN mv /Vesuvius-Grandprize-Winner/* /workspace && rm -rf /Vesuvius-Grandprize-Winner

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to bash so the command line is opened
ENTRYPOINT ["bash"]
