FROM rayproject/ray:1.11.1-gpu
USER root
COPY system-requirements.txt /app/

RUN rm /etc/apt/sources.list.d/nvidia-ml.list \
  && apt-get clean \
  && sudo apt-key del 7fa2af80 \
  && sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN sudo apt-get update --fix-missing \
  && sudo apt-get -y install git curl \
  && sudo apt-get install -y $(cat /app/system-requirements.txt | tr "\n" " ") \
  && sudo apt-get clean \
  && sudo apt-get install awscli -y \
  && sudo rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/

RUN pip3 install -r requirements.txt \
  && pip3 install git+https://github.com/openai/CLIP.git --no-deps