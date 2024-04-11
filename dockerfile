FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --upgrade python3-pip 
RUN apt install -y ffmpeg libgl1-mesa-glx libglib2.0-0 vim

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN chmod 777 predict.sh
RUN chmod 777 start_jupyter.sh

ENV TOKENIZERS_PARALLELISM false
