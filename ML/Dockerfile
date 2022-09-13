#FROM python:3.9.12-slim
#FROM apache/spark-py

ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}


# Add Directory
ADD . /model_module
WORKDIR /model_module

COPY requirements.txt ./
RUN apt-get update -y
RUN apt-get upgrade -y

#RUN apt-get install software-properties-common -y
#RUN apt-get update -y


#RUN add-apt-repository ppa:webupd8team/java -y
#RUN apt-get update -y
#RUN apt-get install oracle-java8-installer -y
#RUN apt-get install openjdk-8-jdk-headless -qq -y

RUN pip install -r requirements.txt


ENTRYPOINT ["/bin/bash"]