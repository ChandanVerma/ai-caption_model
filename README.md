# AI Caption Model
This repo generates captions and caption embeddings for each lomotif. If any questions, please reach out to Data Science team (Sze Chi, Thulasiram, Chandan).

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

# More details about the output
<!-- The output will be written to this table on Snowflake: `DS_CONTENT_MODERATION_TAGGING_1ST_LAYER` (In production). -->
Example output (for snowflake table) upon sending a request to the deployment service:
```python
{'LOMOTIF_ID': '64ac40f7-b4c6-4246-84cb-d1d0875eb084', 'VIDEO': 'https://lomotif-staging.s3.amazonaws.com/lomotifs/2022/1/10/64ac40f7b4c6424684cbd1d0875eb084/64ac40f7b4c6424684cbd1d0875eb084-20220110-0623-video-vs.mp4', 'COUNTRY': 'IN', 'CREATION_TIME': '2022-01-10T06:23:42.712750', 'MESSAGE_RECEIVE_TIME': '2022-03-31 04:58:59.701529+00:00', 'KEY_FRAMES': '1', 'NUM_FRAMES': 750, 'FPS': 25.033377837116156, 'CAPTION_PROCESS_START_TIME': '2022-03-31 04:59:02.337941+00:00', 'CAPTION_PREDICTION_TIME': '2022-03-31 04:59:02.635645+00:00', 'CAPTION_STATUS': 0, 'EMBED_PROCESS_START_TIME': '2022-03-31 04:59:02.639520+00:00', 'EMBED_PREDICTION_TIME': '2022-03-31 04:59:02.646745+00:00', 'EMBED_STATUS': 0, 'TOTAL_TIME_TO_PROCESS': 2.95, 'DATA_ATTRIBUTES': {'LOMOTIF_ID': '64ac40f7-b4c6-4246-84cb-d1d0875eb084', 'VIDEO': 'https://lomotif-staging.s3.amazonaws.com/lomotifs/2022/1/10/64ac40f7b4c6424684cbd1d0875eb084/64ac40f7b4c6424684cbd1d0875eb084-20220110-0623-video-vs.mp4', 'COUNTRY': 'IN', 'CREATION_TIME': '2022-01-10T06:23:42.712750', 'MESSAGE_RECEIVE_TIME': '2022-03-31 04:58:59.701529+00:00', 'KEY_FRAMES': '1', 'NUM_FRAMES': 750, 'FPS': 25.033377837116156, 'CAPTION_PROCESS_START_TIME': '2022-03-31 04:59:02.337941+00:00', 'CAPTION_PREDICTION_TIME': '2022-03-31 04:59:02.635645+00:00', 'CAPTION_STATUS': 0, 'EMBED_PROCESS_START_TIME': '2022-03-31 04:59:02.639520+00:00', 'EMBED_PREDICTION_TIME': '2022-03-31 04:59:02.646745+00:00', 'EMBED_STATUS': 0, 'TOTAL_TIME_TO_PROCESS': 2.95, 'VERSION': '1', 'SENTENCE_TRANSFORMERS': 'all-MiniLM-L6-v2', 'CLIP_MODEL_VERSION': 'ViT-B-32', 'TOKENIZER': 'gpt_tokenizer'}}

```
- LOMOTIF_ID: As per MAIN_DB definition.
- VIDEO: S3 video link to the lomotif
- COUNTRY: As per MAIN_DB definition.
- CREATION_TIME: As per MAIN_DB definition.
- MESSAGE_RECEIVE_TIME: UTC time where kinesis message is received by the deployment service.
- KEY_FRAMES: 0-index key frames of the lomotif that has been predicted on.
- NUM_FRAMES: Number of frames of the lomotif.
- FPS: Number of frames per second.
- (CAPTION/EMBED)_PROCESS_START_TIME: UTC time when model inference begins.
- (CAPTION/EMBED)_PREDICTION_TIME: UTC time when prediction has completed.
- (CAPTION/EMBED)_STATUS: 
    - 0: Prediction successful. 
    - 1: Not a video or image, prediction unsuccesful. 
    - 403: Video clip file not found, prediction unsuccessful. Or Lomotif does not exist on S3, cannot be downloaded after retries, prediction unsuccessful.
    - 4: Some unknown error in the model that was caught by the try...except... loop. Prediction unsucessful.
    - 5: No key frames selected. Prediction unsucessful.
    - 3: Captions of length 0 generated and is still being sent to generate embeddings. Prediction unsucessful.
- TOTAL_TIME_TO_PROCESS: Total time taken to process the lomotif (in seconds).
- DATA_ATTRIBUTES: json which aggregates entire record into a single json (consists of additional attributes such as version, models used etc)
- S3PATH: Path to s3 bucket where the embeddings are stored


Example output (for S3) upon sending a request to the deployment service:

```python
{"LOMOTIF_ID": "64ac40f7-b4c6-4246-84cb-d1d0875eb084", "CAPTIONS": "digital art selected for the #.", "EMBEDDINGS": "[-0.029329456, 0.056086775, -0.018746564, -0.061756812, -0.020087374, 0.060802646, 0.0735421, 0.010757019, 0.07087296, -0.020473685, -0.043756343, 0.0920817, 0.019895816, -0.0012363319, -0.043176133, -0.017089382, 0.0068991105, -0.0819297, 0.044248406, -0.027904136, 0.08391811, 0.03857358, 0.016691744, -0.06506832, 0.035201143, -0.023211502, -0.0731792, -0.0208119, 0.12510842, -0.06472002, -0.066685356, 0.13624546, 0.08878095, -0.0054058745, -0.016445188, -0.0819071, 0.03396594, 0.0609585, -0.018270615, 0.078752555, -0.00150437, -0.09444106, -0.015846403, 0.01859659, 0.002006371, 0.060744513, -0.08441609, 0.07374056, -0.017350962, 0.06886388, 0.016620956, -0.030518027, -0.06774422, 0.009764364, 0.022360103, -0.07189827, 0.049707126, -0.06798088, 0.010923173, 0.031393185, -0.009886811, 0.005077043, -0.04316019, 0.051903598, 0.0466253, 0.03826074, -0.025633443, -0.06350477, -0.033054397, -0.08577073, 0.07957804, -0.011961815, 0.056926195, 0.032848295, 0.052309904, -0.026625441, -0.04578722, -0.06572729, 0.012065738, -0.017988447, 0.018443625, -0.045484707, -0.0304672, 0.03769031, -0.022249913, 0.02791989, -0.04160192, 0.0045483997, 0.061207417, -0.012901609, -0.041659314, 0.021036431, -0.041719, 0.029463816, -0.02866475, 0.0031562333, -0.005524478, -0.008836522, -0.024090227, 0.05995288, 0.04626031, -0.030651083, 0.023403795, -0.067668855, -0.04297634, -0.08795152, 0.058076344, 0.0032498252, 0.03331584, -0.014487371, -0.015083483, 0.012516371, -0.011991876, 0.0105655715, -0.03441346, -0.08930332, -0.050697874, -0.00452252, 0.07947601, -0.11984155, 0.027082495, -0.014428789, -0.07990599, -0.032270454, -0.06576792, -0.036303025, -0.09349414, -4.797083e-33, 0.023979988, 0.02665842, 0.08820927, 0.007005781, 0.04767684, -0.02398392, -0.030753048, 0.0075660385, -0.0004589884, 0.019575283, 0.005674641, 0.008052464, -0.05028386, 0.097233854, 0.040650755, -0.07556635, 0.013094984, 0.0404577, -0.035029054, -0.015965354, 0.015782507, 0.06301881, 0.02494267, 0.021142527, -0.029294955, 0.15365419, -0.02924808, -0.044526987, 0.072585516, 0.049729776, -0.047918092, 0.024009438, 0.0076507255, -0.048928022, -0.028734203, -0.0050402842, -0.03847573, -0.04947415, 0.014013909, 0.038844734, 0.07016217, 0.039052576, -0.016075596, 0.006304892, 0.0068881516, 0.1159055, 0.056412473, 0.11197463, -0.01995749, 0.08812499, -0.0017787553, 0.102669045, -0.06456959, -0.034607414, 0.05775847, -0.037083305, 0.011655874, -0.037549824, -0.021312838, -0.04299604, 0.065967396, 0.050414145, 0.0494164, 0.027905859, 0.010278554, 0.018931054, 0.041042045, -0.03419675, -0.010109619, -0.010864059, -0.088618256, 0.049682304, 0.025007349, -0.05184398, -0.032868043, -0.00021771918, 0.017033517, -0.0023336445, -0.038797114, -0.0071823904, -0.111727886, 0.011911983, -0.0009726313, -0.07615726, 0.036571316, 0.07440818, 0.051307116, -0.049676392, -0.012797021, 0.0033599432, 0.002085736, -0.011019599, -0.06519387, 0.060765214, -0.050634038, 4.5592948e-33, -0.06091393, 0.017198605, -0.015893994, 0.09350388, 0.03629992, -0.06445921, 0.012347261, 0.028093247, 0.11334458, 0.018826686, 0.11587608, -0.01837495, -0.063752435, 0.0008685829, 0.017543305, -0.030351257, 0.010522285, -0.012525979, -0.05856912, 0.022400007, 0.040513713, 0.0008066779, -0.0768021, 0.026975058, -0.035282828, 0.122038126, 0.022949908, -0.0072373184, 0.059388425, 0.03833534, -0.010420285, -0.05991283, 0.005679974, 0.07488474, -0.036419682, -0.031091621, 0.14673397, -0.008054492, 0.02371118, 0.0879113, 0.040542755, 0.009325476, 0.030569466, 0.12180967, -0.101044744, -0.0067432574, 0.018244429, 0.014899921, 0.038433585, -0.013480258, 0.05545174, -0.01941476, -0.03972745, -0.042325947, -0.030304099, 0.09042363, -0.05400195, 0.04361602, 0.005290369, 0.10120425, -0.021733867, 0.020023575, -0.043065373, -0.0331121, -0.026816199, -0.010860388, -0.030815752, 0.0052705193, -0.16335264, 0.04761554, 0.03567543, 0.050981887, 0.030595066, -0.028838094, -0.089564554, -0.051191386, 0.026241964, 0.08891005, 0.043917194, 0.021947913, -0.058256462, -0.04036436, -0.05408361, 0.059982818, 0.050154656, 0.08762502, 0.0046061357, -0.055460542, -0.03927013, -0.008604929, 0.0020997834, 0.018757196, 0.039835542, -0.016353881, -0.075301945, -1.7622902e-08, 0.02226524, -0.022852793, 0.028314935, -0.098091975, 0.04900286, 0.040549833, 0.008466917, -0.006054385, -0.045610674, -0.0055938694, 0.010350606, -0.071832635, -0.01181233, 0.012662506, 0.09920328, -0.07545119, -0.017868245, -0.044169478, -0.05119116, -0.06214855, -0.004590545, -0.038637783, 0.047885165, -0.07602872, -0.06552941, 0.043439377, -0.037897274, 0.03185527, -0.014160613, 0.07279262, 0.021434937, 0.08783644, -0.014222776, 0.028474316, -0.030348094, -0.021460991, -0.035280995, -0.0049518025, 0.006331063, -0.09401116, 0.009036386, -0.024448542, -0.059168838, 0.025294805, 0.029776363, -0.0060176426, 0.09132141, -0.070320554, -0.040334694, 0.007263757, -0.080931045, -0.026002688, 0.09646749, 0.03279913, 0.03474122, -0.063590825, 0.050083652, 0.03720692, -0.024421262, 0.090531625, 0.07660246, -0.06495616, -0.013710193, 0.012788264]"}
```
- LOMOTIF_ID: As per MAIN_DB definition.
- CAPTIONS: Captions generated for the lomotif
- EMBEDDINGS: Vector representation for the captions
