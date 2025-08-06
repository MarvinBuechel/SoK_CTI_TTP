sudo docker build -t minimal -f Dockerfile.minimal .
sudo docker run --volume ./experiments:/workspace/finetuning/experiments --gpus all --rm minimal