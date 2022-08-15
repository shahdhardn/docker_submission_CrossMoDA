## Pull from existing image
# FROM nvcr.io/nvidia/pytorch:21.05-py3
FROM pytorch/pytorch

WORKDIR /

COPY . /

## Install Python packages in Docker image
RUN pip install -r requirements.txt

## Execute the inference command 
CMD ["python", "run_inference.py"]