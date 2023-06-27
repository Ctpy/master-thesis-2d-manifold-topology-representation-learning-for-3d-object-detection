FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

RUN apt-get update && apt-get install git libgl1 -y

# Copy the requirements file to the container
COPY requirements.txt /workspace/requirements.txt

# Install the requirements
RUN pip install -r requirements.txt

RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html