# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch

# Set the working directory
WORKDIR /nnUNet

# Copy the current directory contents into the container
COPY . /nnUNet

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run when the container launches
CMD ["bash"]