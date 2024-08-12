# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Install gdown to handle Google Drive file downloads
RUN pip install gdown

# Download the canonical timesformer weights from Google Drive using gdown
RUN gdown --id 13Iu-dR-1sKq_oGJfNa86LcBSv1o4XA37 -O /workspace/timesformer_weights.ckpt

RUN apt-get update \
    && apt-get -y install git
# Copy the Vesuvius GP Ink Detection code into the workspace
COPY . /workspace

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to bash so the command line is opened
ENTRYPOINT ["bash"]
