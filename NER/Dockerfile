# Base ubuntu image
FROM ubuntu:24.04

# Install python and git
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip git libmagic1 -y
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /ner

# Copy contentv
COPY models/ models/
COPY spacy_extensions/ spacy_extensions/
COPY README.md .
COPY requirements.txt .
COPY setup.py .

# Install ner tools
RUN pip3 install -e /ner/
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_trf
RUN python3 -m nltk.downloader wordnet
RUN git clone https://github.com/mitre/cti.git
WORKDIR /ner/cti
RUN ["git", "checkout", "ATT&CK-v14.1"]
WORKDIR /ner

# Run experiments
COPY experiments/ experiments/
COPY data/ data/
WORKDIR /ner/experiments/

RUN chmod +x ablation.sh
RUN chmod +x run_pipelines.sh
RUN chmod +x evaluate_pipelines.sh
RUN chmod +x create_pipelines.sh

CMD ["/bin/bash", "ablation.sh"]