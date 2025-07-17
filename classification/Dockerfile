FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# dependencies (python, pip, requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev curl libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar -xf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-3.10.12 Python-3.10.12.tgz

RUN python3.10 -m ensurepip && python3.10 -m pip install --upgrade pip

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1
RUN pip install --upgrade pip

WORKDIR /classification

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy the rest of the code + data
COPY *.py .
COPY *.ipynb .

COPY configs/ configs
COPY nvidia-embed/ nvidia-embed
COPY fine_tuned/ fine_tuned
COPY datasets/ datasets
COPY local/ local
COPY artifact_eval/ artifact_eval
COPY .darkbert_token .

RUN chmod +x artifact_eval/*.sh

ARG RUN_DEVICE 
ENV DEVICE=${RUN_DEVICE:-cuda}

CMD ["bash", "./artifact_eval/minimal.sh"]
