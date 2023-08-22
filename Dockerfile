# Use an official pytorch runtime as a parent image
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

MAINTAINER Youssef Nashed "ynashed@slac.stanford.edu"

ENV PYTHONUNBUFFERED=1

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
ENV DEBIAN_FRONTEND=noninteractive
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

WORKDIR /work
ADD requirements.txt /work/requirements.txt

RUN apt-get update && \
    apt-get install -y git wget build-essential libtool autoconf unzip libssl-dev

RUN pip install --no-use-pep517 --no-cache-dir -r requirements.txt
