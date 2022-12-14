# AI Caption Model
This repo generates captions and caption embeddings for each video

# Set-up .env file
There needs to be a `.env` file with following parameters.
```
AWS_ACCESS_KEY_ID=YOUR AWS ACCESS KEY
AWS_SECRET_ACCESS_KEY=YOUR AWS SECRET KEY
AWS_REGION=us-east-2

## FOR DEV
SnowflakeResultsQueue=ai_caption-results_dev
RawResultsQueue=ai_caption_raw-results_dev
AWS_BUCKET_NAME=lomotif-datalake-dev

## FOR PROD
SnowflakeResultsQueue=ai_caption-results_prod
RawResultsQueue=ai_caption_raw-results_prod
AWS_BUCKET_NAME=lomotif-datalake-prod

version=1
sentence_transformers=all-MiniLM-L6-v2
clip=ViT-B-32
tokenizer=gpt_tokenizer

DownloadNumCPUPerReplica=0.2
DownloadNumReplicas=1
DownloadMaxCon=100

PreprocessNumCPUPerReplica=1
PreprocessNumReplicas=1
PreprocessMaxCon=100

CaptionNumCPUPerReplica=0.2
CaptionNumGPUPerReplica=0.2
CaptionNumReplicas=4
CaptionMaxCon=100

EmbedNumCPUPerReplica=0.2
EmbedNumGPUPerReplica=0.1
EmbedNumReplicas=1
EmbedMaxCon=100

PipelineNumCPUPerReplica=0.2
PipelineNumReplicas=1
PipelineMaxCon=100
```

# Instructions (Docker)
1) Ensure there are environment variables or `.env` file, see section above for environment variables.
2) Ensure GPU for docker is enabled. See section below.
3) Once the container is able to detect the GPU, we can follow the normal process of

```
docker-compose build
docker-compose up
```

# Enabling GPU for Docker
To enable the GPU for Docker, make sure Nvidia drivers for the system are installed. [Refer link for details](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)

Commands which can help install Nvidia drivers are:
```
unbuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

Then nvidia-docker2 tools needs to be installed.
To install follow the below instructions.
[Refer link for details](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

# Pytest
To be updated.
