FROM ubuntu:latest

RUN apt-get clean && apt-get update \
    && apt-get update \
    && apt-get install python3.9 python3-pip -y

RUN pip install \
       numpy==1.21.6 \
       nibabel==4.0.2 \
       matplotlib==3.6.2 \
       pandas==1.5.2 \
       scikit_image==0.19.3 \
       scipy==1.9.3 \
       SimpleITK==2.2.0 \
       tqdm==4.64.1
       
RUN pip install pyradiomics==3.0.1

WORKDIR /usr/src/app

COPY . .