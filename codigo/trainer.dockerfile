FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get -y install gcc && apt-get install -y build-essential && apt-get install -y libsndfile1

WORKDIR /root

ADD . code

RUN mkdir -p code/tmp

RUN pip install -r code/requirements.txt

WORKDIR code
