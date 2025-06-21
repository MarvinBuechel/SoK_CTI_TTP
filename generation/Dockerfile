FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY . .

RUN apt-get update
RUN apt-get install -y build-essential
RUN pip3 install -r requirements.txt
#RUN pip3 uninstall trl -y && pip3 install --no-cache-dir --force-reinstall --no-deps "trl<0.15.0"
#RUN pip3 install --no-cache-dir --upgrade --no-deps unsloth 
#RUN pip3 install --no-cache-dir --upgrade --no-deps unsloth_zoo

CMD ["python3", "supervised_finetuning.py"]