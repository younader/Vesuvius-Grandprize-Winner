# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Download the canonical timesformer weights
RUN apt-get update && \
    apt-get install -y wget && \
    wget -O timesformer_weights.ckpt https://drive.google.com/uc?export=download&id=13Iu-dR-1sKq_oGJfNa86LcBSv1o4XA37

# Set the entrypoint to bash so the command line is opened
ENTRYPOINT ["bash"]
