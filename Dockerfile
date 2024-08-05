# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown to handle Google Drive file downloads
RUN pip install gdown

# Copy the rest of the project files into the container
COPY . .

# Download the canonical timesformer weights from Google Drive using gdown
RUN gdown --id 13Iu-dR-1sKq_oGJfNa86LcBSv1o4XA37 -O /workspace/timesformer_weights.ckpt

# Set the entrypoint to bash so the command line is opened
ENTRYPOINT ["bash"]
