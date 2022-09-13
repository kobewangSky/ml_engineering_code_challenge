FROM python:3.9.12-slim

# Add Directory
ADD . /model_module
WORKDIR /model_module

COPY requirements.txt ./
RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip install -r requirements.txt
