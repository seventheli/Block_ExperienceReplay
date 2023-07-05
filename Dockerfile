# Use an official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Update the conda packages
RUN conda update -n base -c defaults conda

# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "hpc", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN python -c "import ray.rllib; print(ray.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"

